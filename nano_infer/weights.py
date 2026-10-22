"""Load GPT-2 checkpoints written by the reference implementation.

Runtime loading uses `safetensors` or `torch.load` plus `huggingface_hub`.
Transformers is not imported here and `generate()` is never called.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from nano_infer.config import ModelConfig
from nano_infer.model import GPT2LM

_CONV_WEIGHTS = (
    "attn.c_attn.weight",
    "attn.c_proj.weight",
    "mlp.c_fc.weight",
    "mlp.c_proj.weight",
)


def _strip_prefix(key: str) -> str:
    if key.startswith("transformer."):
        return key[len("transformer.") :]
    return key


def _needs_transpose(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in _CONV_WEIGHTS)


def apply_hf_weights(model: GPT2LM, tensors: dict[str, torch.Tensor]) -> None:
    """Copy a Hugging Face GPT-2 state dict into this module.

    Conv1D weights are stored as `[in, out]`. `nn.Linear` stores `[out, in]`,
    so those four matrices are transposed. The language-model head stays tied
    to the token embedding.
    """
    own = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        stripped = _strip_prefix(key)
        if stripped == "lm_head.weight" or stripped not in own:
            continue
        if _needs_transpose(stripped):
            mapped[stripped] = value.detach().t().contiguous()
        else:
            mapped[stripped] = value.detach()
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    real_missing = [key for key in missing if key != "lm_head.weight"]
    if real_missing or unexpected:
        raise RuntimeError(f"weight mismatch, missing={real_missing}, unexpected={unexpected}")
    model.lm_head.weight = model.wte.weight


def _load_tensors(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        loaded = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded, dict):
        raise RuntimeError(f"unrecognized checkpoint at {path}")
    return loaded


def resolve_checkpoint(source: str | Path) -> Path:
    """Return a directory that contains `config.json` and a weight file."""
    path = Path(source)
    if path.exists():
        if path.is_file():
            return path.parent
        return path
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        repo_id=str(source),
        allow_patterns=["config.json", "model.safetensors", "pytorch_model.bin"],
    )
    return Path(snapshot)


def load_model(source: str | Path, device: str = "cpu") -> tuple[ModelConfig, GPT2LM]:
    """Load a GPT-2 checkpoint from a local directory or a hub repo id."""
    directory = resolve_checkpoint(source)
    config_path = directory / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"no config.json in {directory}")
    config = ModelConfig.from_json(config_path)
    weight_path = None
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = directory / name
        if candidate.exists():
            weight_path = candidate
            break
    if weight_path is None:
        raise FileNotFoundError(f"no weights in {directory}")
    model = GPT2LM(config)
    apply_hf_weights(model, _load_tensors(weight_path))
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return config, model
