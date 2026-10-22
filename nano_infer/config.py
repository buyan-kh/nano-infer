"""Model hyperparameters for a GPT-2-style decoder."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """GPT-2 configuration.

    Field names follow the public GPT-2 checkpoint (`n_embd`, `n_layer`, ...)
    so a `config.json` from the Hugging Face hub loads without translation.
    """

    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    n_inner: int
    layer_norm_epsilon: float = 1e-5
    bos_token_id: int = 50256
    eos_token_id: int = 50256

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_layer < 1 or self.n_head < 1 or self.vocab_size < 1:
            raise ValueError("invalid model dimensions")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_dict(cls, raw: dict) -> ModelConfig:
        n_embd = int(raw.get("n_embd", raw.get("hidden_size")))
        n_layer = int(raw.get("n_layer", raw.get("num_hidden_layers")))
        n_head = int(raw.get("n_head", raw.get("num_attention_heads")))
        n_positions = int(raw.get("n_positions", raw.get("n_ctx", raw.get("max_position_embeddings"))))
        vocab = int(raw["vocab_size"])
        n_inner = raw.get("n_inner")
        if n_inner is None:
            n_inner = 4 * n_embd
        return cls(
            vocab_size=vocab,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=int(n_inner),
            layer_norm_epsilon=float(raw.get("layer_norm_epsilon", 1e-5)),
            bos_token_id=int(raw.get("bos_token_id", 50256) or 50256),
            eos_token_id=int(raw.get("eos_token_id", 50256) or 50256),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> ModelConfig:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))
