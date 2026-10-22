"""Shared fixtures for logit and generation tests."""

from __future__ import annotations

import torch

from nano_infer.config import ModelConfig
from nano_infer.model import GPT2LM
from nano_infer.sampling import SamplingParams


def small_config(**overrides) -> ModelConfig:
    raw = dict(
        vocab_size=64,
        n_positions=48,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=64,
        layer_norm_epsilon=1e-5,
        bos_token_id=0,
        eos_token_id=1,
    )
    raw.update(overrides)
    return ModelConfig(**raw)


def make_model(config: ModelConfig | None = None, seed: int = 0) -> GPT2LM:
    torch.manual_seed(seed)
    model = GPT2LM(config or small_config())
    model.eval()
    return model


def greedy(max_tokens: int = 6) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, stop_token_ids=())


def hf_from_model(model: GPT2LM):
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = model.config
    hf_cfg = GPT2Config(
        vocab_size=cfg.vocab_size,
        n_positions=cfg.n_positions,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_inner=cfg.n_inner,
        layer_norm_epsilon=cfg.layer_norm_epsilon,
        bos_token_id=cfg.bos_token_id,
        eos_token_id=cfg.eos_token_id,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    )
    hf_cfg._attn_implementation = "eager"
    hf = GPT2LMHeadModel(hf_cfg)
    hf.eval()
    own = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}

    def put(key: str, tensor: torch.Tensor, transpose: bool = False) -> None:
        value = tensor.detach()
        mapped[key] = value.t().contiguous() if transpose else value.clone()

    put("transformer.wte.weight", own["wte.weight"])
    put("transformer.wpe.weight", own["wpe.weight"])
    put("transformer.ln_f.weight", own["ln_f.weight"])
    put("transformer.ln_f.bias", own["ln_f.bias"])
    put("lm_head.weight", own["wte.weight"])
    for layer in range(cfg.n_layer):
        src = f"h.{layer}."
        dst = f"transformer.h.{layer}."
        put(dst + "ln_1.weight", own[src + "ln_1.weight"])
        put(dst + "ln_1.bias", own[src + "ln_1.bias"])
        put(dst + "ln_2.weight", own[src + "ln_2.weight"])
        put(dst + "ln_2.bias", own[src + "ln_2.bias"])
        put(dst + "attn.c_attn.weight", own[src + "attn.c_attn.weight"], transpose=True)
        put(dst + "attn.c_attn.bias", own[src + "attn.c_attn.bias"])
        put(dst + "attn.c_proj.weight", own[src + "attn.c_proj.weight"], transpose=True)
        put(dst + "attn.c_proj.bias", own[src + "attn.c_proj.bias"])
        put(dst + "mlp.c_fc.weight", own[src + "mlp.c_fc.weight"], transpose=True)
        put(dst + "mlp.c_fc.bias", own[src + "mlp.c_fc.bias"])
        put(dst + "mlp.c_proj.weight", own[src + "mlp.c_proj.weight"], transpose=True)
        put(dst + "mlp.c_proj.bias", own[src + "mlp.c_proj.bias"])
    missing, unexpected = hf.load_state_dict(mapped, strict=False)
    missing = [key for key in missing if "attn.bias" not in key and "masked_bias" not in key]
    if missing or unexpected:
        raise RuntimeError(f"hf load mismatch missing={missing} unexpected={unexpected}")
    return hf
