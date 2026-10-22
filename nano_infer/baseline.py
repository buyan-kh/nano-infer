"""Serial baselines built on the same forward pass.

`recompute` runs the full prefix on every new token and throws the cache away.
`serial_kv` keeps a paged cache but never batches: each request runs to
completion before the next one starts. Neither baseline is a second model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from nano_infer.engine import Engine
from nano_infer.model import GPT2LM, prefill_logits
from nano_infer.sampling import SamplingParams, sample


@dataclass
class WaveResult:
    request_id: str
    tokens: list[int]
    ttft_s: float
    latency_s: float


def _generator(sampling: SamplingParams) -> torch.Generator | None:
    if sampling.seed is None:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sampling.seed)
    return generator


@torch.inference_mode()
def generate_recompute(
    model: GPT2LM,
    prompt: list[int],
    sampling: SamplingParams,
    *,
    block_size: int = 16,
    wave_start: float | None = None,
    request_id: str = "naive",
) -> WaveResult:
    start = wave_start if wave_start is not None else time.perf_counter()
    ids = list(prompt)
    generated: list[int] = []
    first: float | None = None
    generator = _generator(sampling)
    while len(generated) < sampling.max_tokens:
        logits = prefill_logits(model, ids, block_size=block_size)
        token = sample(logits[-1], sampling, generator)
        now = time.perf_counter()
        if first is None:
            first = now
        generated.append(token)
        ids.append(token)
        if token in sampling.stop_token_ids:
            break
    end = time.perf_counter()
    return WaveResult(
        request_id=request_id,
        tokens=generated,
        ttft_s=(first or end) - start,
        latency_s=end - start,
    )


def generate_serial_kv(
    model: GPT2LM,
    prompts: list[list[int]],
    sampling: SamplingParams,
    *,
    block_size: int,
    num_blocks: int,
    max_prefill_tokens: int = 128,
    wave_start: float | None = None,
) -> list[WaveResult]:
    """One request at a time, paged cache, batch size 1."""
    start = wave_start if wave_start is not None else time.perf_counter()
    results: list[WaveResult] = []
    for index, prompt in enumerate(prompts):
        engine = Engine(
            model,
            block_size=block_size,
            num_blocks=num_blocks,
            max_batch_size=1,
            max_prefill_tokens=max_prefill_tokens,
            watermark_blocks=0,
        )
        request_id = f"serial-{index}"
        engine.add_request(prompt, sampling, request_id)
        # Stamp the wave start so queue wait is part of latency.
        for seq in list(engine.scheduler.waiting):
            seq.arrival_time = start
        while engine.has_work():
            engine.step()
        seq = engine.scheduler.finished[-1]
        results.append(
            WaveResult(
                request_id=request_id,
                tokens=list(seq.generated_ids),
                ttft_s=(seq.first_token_time or time.perf_counter()) - start,
                latency_s=(seq.finish_time or time.perf_counter()) - start,
            )
        )
    return results


def generate_recompute_wave(
    model: GPT2LM,
    prompts: list[list[int]],
    sampling: SamplingParams,
    *,
    block_size: int = 16,
) -> list[WaveResult]:
    start = time.perf_counter()
    return [
        generate_recompute(
            model,
            prompt,
            sampling,
            block_size=block_size,
            wave_start=start,
            request_id=f"recompute-{index}",
        )
        for index, prompt in enumerate(prompts)
    ]
