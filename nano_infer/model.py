"""GPT-2 decoder forward pass with paged attention.

The math matches the original GPT-2 block: token plus learned position
embeddings, pre-norm residual attention, pre-norm MLP with the tanh GELU
approximation, and a tied language-model head. Dropout is omitted because
serving always runs in eval.

Attention reads keys and values out of the paged cache. On CPU there is no
custom kernel: slots are gathered into a dense workspace, then a standard
causal matmul runs. The block table is still the source of truth for where
each token's KV lives, which is what the allocator and the scheduler manage.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from nano_infer.cache import BlockAllocator, KVCache, stack_block_tables
from nano_infer.config import ModelConfig


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """OpenAI GPT-2 GELU (tanh approximation), matching `gelu_new`."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def _write_kv(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    block_tables: torch.Tensor,
    write_starts: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter new K/V into paged slots.

    `k_cache` is `[num_blocks, block_size, heads, dim]`.
    `k_new` is `[batch, seq, heads, dim]`.
    """
    batch, seqlen, n_head, head_dim = k_new.shape
    positions = write_starts[:, None] + torch.arange(seqlen, device=k_new.device)[None, :]
    logical = torch.div(positions, block_size, rounding_mode="floor")
    offset = positions - logical * block_size
    physical = torch.gather(block_tables, 1, logical)
    flat_blocks = physical.reshape(-1)
    flat_offsets = offset.reshape(-1)
    k_cache[flat_blocks, flat_offsets] = k_new.reshape(batch * seqlen, n_head, head_dim)
    v_cache[flat_blocks, flat_offsets] = v_new.reshape(batch * seqlen, n_head, head_dim)


def _paged_attention(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    write_starts: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Causal attention for a batch of sequences with distinct lengths.

    `query` is `[batch, seq, heads, dim]` for the tokens just written.
    Keys at absolute positions `0 .. context_len-1` are gathered from the
    block table. Query position `write_start + s` may attend to keys `<=` that
    position. Padding past each sequence's context is masked.
    """
    batch, seqlen, n_head, head_dim = query.shape
    max_len = int(context_lens.max().item())
    if max_len <= 0:
        raise RuntimeError("context length must be positive")
    device = query.device
    positions = torch.arange(max_len, device=device)
    logical = torch.div(positions, block_size, rounding_mode="floor")
    logical = torch.clamp(logical, max=block_tables.shape[1] - 1)
    offset = positions - torch.div(positions, block_size, rounding_mode="floor") * block_size
    physical = block_tables[:, logical]
    offset_bt = offset.unsqueeze(0).expand(batch, max_len)
    keys = k_cache[physical, offset_bt]
    values = v_cache[physical, offset_bt]

    q = query.permute(0, 2, 1, 3).float()
    k = keys.permute(0, 2, 1, 3).float()
    v = values.permute(0, 2, 1, 3).float()
    scale = head_dim**-0.5
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale

    query_pos = write_starts[:, None] + torch.arange(seqlen, device=device)[None, :]
    key_pos = positions[None, :]
    causal = key_pos.unsqueeze(1) <= query_pos.unsqueeze(-1)
    within = key_pos.unsqueeze(1) < context_lens[:, None, None]
    mask = causal & within
    scores = scores.masked_fill(~mask[:, None, :, :], torch.finfo(torch.float32).min)
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    return out.permute(0, 2, 1, 3).to(dtype=query.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(
        self,
        hidden: torch.Tensor,
        layer_k: torch.Tensor,
        layer_v: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        write_starts: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        batch, seqlen, channels = hidden.shape
        qkv = self.c_attn(hidden)
        query, key, value = qkv.split(channels, dim=-1)
        query = query.view(batch, seqlen, self.n_head, self.head_dim)
        key = key.view(batch, seqlen, self.n_head, self.head_dim)
        value = value.view(batch, seqlen, self.n_head, self.head_dim)
        _write_kv(layer_k, layer_v, key, value, block_tables, write_starts, block_size)
        attended = _paged_attention(
            query, layer_k, layer_v, block_tables, context_lens, write_starts, block_size
        )
        merged = attended.contiguous().view(batch, seqlen, channels)
        return self.c_proj(merged)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.c_proj(gelu_new(self.c_fc(hidden)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        layer_k: torch.Tensor,
        layer_v: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        write_starts: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        hidden = hidden + self.attn(
            self.ln_1(hidden), layer_k, layer_v, block_tables, context_lens, write_starts, block_size
        )
        hidden = hidden + self.mlp(self.ln_2(hidden))
        return hidden


class GPT2LM(nn.Module):
    """Decoder-only GPT-2 language model with a paged KV cache interface."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.h = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        write_starts: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits `[batch, seq, vocab]` and write the new K/V into `kv_cache`.

        `input_ids[:, 0]` of each row is the token at absolute position
        `write_starts[row]`. `context_lens[row]` is the length of the prefix
        after this write, which is `write_starts[row] + seq`.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, seq]")
        batch, seqlen = input_ids.shape
        if write_starts.shape != (batch,) or context_lens.shape != (batch,):
            raise ValueError("write_starts and context_lens must be [batch]")
        device = input_ids.device
        positions = write_starts[:, None] + torch.arange(seqlen, device=device)[None, :]
        if int(positions.max()) >= self.config.n_positions:
            raise ValueError(
                f"position {int(positions.max())} exceeds n_positions {self.config.n_positions}"
            )
        hidden = self.wte(input_ids) + self.wpe(positions)
        for layer, block in enumerate(self.h):
            hidden = block(
                hidden,
                kv_cache.k[layer],
                kv_cache.v[layer],
                block_tables,
                context_lens,
                write_starts,
                kv_cache.block_size,
            )
        hidden = self.ln_f(hidden)
        return self.lm_head(hidden)


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def prefill_logits(
    model: GPT2LM,
    token_ids: list[int],
    *,
    block_size: int = 16,
) -> torch.Tensor:
    """Logits for a single full sequence, using a throwaway paged cache.

    Shape is `[seq, vocab]`. Used by tests and by the recompute baseline,
    which calls this on the growing prefix every step.
    """
    if not token_ids:
        raise ValueError("empty token list")
    device = model_device(model)
    needed = (len(token_ids) + block_size - 1) // block_size
    cache = KVCache.allocate(model.config, needed, block_size, device=device, dtype=torch.float32)
    blocks = BlockAllocator(needed).allocate(needed)
    tables = stack_block_tables([blocks], device)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    write_starts = torch.zeros(1, dtype=torch.long, device=device)
    context_lens = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
    logits = model(input_ids, cache, tables, context_lens, write_starts)
    return logits[0]
