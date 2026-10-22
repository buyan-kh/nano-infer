"""Temperature, top-k, and top-p sampling, plus speculative rejection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = ()
    stop: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        self.stop_token_ids = tuple(int(t) for t in self.stop_token_ids)
        self.stop = tuple(self.stop)


def make_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def sample(logits: torch.Tensor, params: SamplingParams, generator: torch.Generator | None = None) -> int:
    """Draw one token from a 1-D logit vector.

    `temperature == 0` is greedy argmax and does not consume the generator.
    top-k keeps every logit tied with the k-th. top-p keeps the smallest
    prefix of the sorted distribution whose mass reaches `top_p`, including
    the token that crosses the threshold.
    """
    if logits.ndim != 1:
        raise ValueError("sample() expects a 1-D logit vector")
    if params.temperature == 0:
        return int(torch.argmax(logits).item())

    scores = logits.float() / params.temperature
    if params.top_k > 0:
        k = min(params.top_k, scores.shape[-1])
        cutoff = torch.topk(scores, k).values[-1]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))
    if params.top_p < 1:
        ordered, indices = torch.sort(scores, descending=True)
        probs = torch.softmax(ordered, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # Drop tokens strictly after the nucleus. The token that crosses
        # top_p has (cumulative - prob) <= top_p and stays.
        remove = (cumulative - probs) > params.top_p
        ordered = ordered.masked_fill(remove, float("-inf"))
        restored = torch.full_like(scores, float("-inf"))
        restored.scatter_(0, indices, ordered)
        scores = restored
    probs = torch.softmax(scores, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
        return int(torch.argmax(logits).item())
    drawn = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(drawn.item())


def speculative_accept(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    token: int,
    generator: torch.Generator | None,
) -> bool:
    """Leviathan et al. accept test: keep the draft token with probability min(1, p/q)."""
    p = float(target_probs[token])
    q = float(draft_probs[token])
    if q <= p:
        return True
    threshold = p / q
    uniform = torch.rand((), generator=generator)
    return float(uniform) <= threshold


def rejection_distribution(target_probs: torch.Tensor, draft_probs: torch.Tensor) -> torch.Tensor:
    """Normalized positive part of p - q, the residual law used on a rejection."""
    residual = torch.clamp(target_probs - draft_probs, min=0)
    total = residual.sum()
    if float(total) <= 0:
        return target_probs
    return residual / total
