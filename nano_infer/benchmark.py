"""Measure logit error, throughput, latency, and speculative decoding.

Numbers written by this module are wall-clock measurements on the machine
that ran it. Nothing here is estimated.

Usage:
    python -m nano_infer.benchmark
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from nano_infer.baseline import generate_recompute_wave, generate_serial_kv
from nano_infer.config import ModelConfig
from nano_infer.engine import Engine
from nano_infer.model import GPT2LM, prefill_logits
from nano_infer.sampling import SamplingParams
from nano_infer.weights import load_model

ROOT = Path(__file__).resolve().parents[1]
DISTIL = Path(
    os.environ.get(
        "NANO_INFER_DISTIL_PATH",
        "/home/ubuntu/.cache/huggingface/hub/models--distilgpt2/snapshots/2290a62682d06624634c1f46a6ad5be0f47f38aa",
    )
)
GPT2 = Path(
    os.environ.get(
        "NANO_INFER_GPT2_PATH",
        "/home/ubuntu/.cache/huggingface/hub/models--openai-community--gpt2/snapshots",
    )
)
from nano_infer.weights import TINY_DIR as TINY  # noqa: E402

PROMPT_TOKENS = 32
NEW_TOKENS = 16
REQUESTS_PER_LEVEL = 16
CONCURRENCIES = (1, 2, 4, 8)
BLOCK_SIZE = 16
POOL_BLOCKS = 128


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def hardware() -> dict:
    cpu = "unknown"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        cpu = platform.processor() or "unknown"
    pages = os.sysconf("SC_PHYS_PAGES")
    page = os.sysconf("SC_PAGE_SIZE")
    return {
        "cpu": cpu,
        "vcpu": os.cpu_count() or 1,
        "ram_gib": round(pages * page / (1024**3), 2),
        "device": "cpu",
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "dtype": "float32",
    }


def prompts_for(count: int, length: int, salt: int) -> list[list[int]]:
    rows = []
    for index in range(count):
        rows.append([((salt + 1) * 17 + index * 13 + step * 3) % 4000 + 100 for step in range(length)])
    return rows


def logit_error(source: Path, label: str, n_tokens: int = 16) -> dict:
    from transformers import GPT2LMHeadModel

    print(f"logit check {label}", flush=True)
    config, model = load_model(source)
    hf = GPT2LMHeadModel.from_pretrained(source, attn_implementation="eager")
    hf.eval()
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, min(config.vocab_size, 50000), (n_tokens,), generator=generator).tolist()
    ours = prefill_logits(model, tokens, block_size=BLOCK_SIZE)
    with torch.no_grad():
        reference = hf(torch.tensor([tokens], dtype=torch.long)).logits[0]
    error = float((ours - reference).abs().max())
    argmax_match = int(torch.argmax(ours[-1])) == int(torch.argmax(reference[-1]))
    del hf, model, ours, reference
    print(f"  max abs error {error:.3e} argmax_match={argmax_match}", flush=True)
    return {
        "model": label,
        "prompt_tokens": n_tokens,
        "max_abs_logit_error": error,
        "final_token_argmax_match": argmax_match,
        "hf_attention": "eager",
    }


def _summarize(rows: list[dict], wall_s: float) -> dict:
    tokens = sum(row["tokens"] for row in rows)
    return {
        "requests": len(rows),
        "generated_tokens": tokens,
        "wall_s": wall_s,
        "throughput_tps": tokens / wall_s if wall_s > 0 else 0.0,
        "ttft_p50_ms": percentile([row["ttft_s"] for row in rows], 50) * 1000,
        "ttft_p99_ms": percentile([row["ttft_s"] for row in rows], 99) * 1000,
        "latency_p50_ms": percentile([row["latency_s"] for row in rows], 50) * 1000,
        "latency_p99_ms": percentile([row["latency_s"] for row in rows], 99) * 1000,
    }


def _batched_wave(
    model: GPT2LM,
    prompts: list[list[int]],
    sampling: SamplingParams,
    *,
    num_blocks: int,
    max_batch_size: int,
    watermark_blocks: int = 0,
    reserve_full: bool = False,
) -> tuple[list[dict], float]:
    engine = Engine(
        model,
        block_size=BLOCK_SIZE,
        num_blocks=num_blocks,
        max_batch_size=max_batch_size,
        max_prefill_tokens=512,
        watermark_blocks=watermark_blocks,
        reserve_full=reserve_full,
    )
    start = time.perf_counter()
    for index, prompt in enumerate(prompts):
        engine.add_request(prompt, sampling, f"r{index}")
    for seq in engine.scheduler.waiting:
        seq.arrival_time = start
    while engine.has_work():
        for event in engine.step():
            if event.error:
                raise RuntimeError(event.error)
    wall = time.perf_counter() - start
    rows = []
    for seq in engine.scheduler.finished:
        if seq.finish_reason == "error":
            raise RuntimeError(seq.error)
        rows.append(
            {
                "id": seq.request_id,
                "tokens": len(seq.generated_ids),
                "ids": list(seq.generated_ids),
                "ttft_s": (seq.first_token_time or start) - start,
                "latency_s": (seq.finish_time or time.perf_counter()) - start,
            }
        )
    rows.sort(key=lambda row: row["id"])
    return rows, wall


def _baseline_rows(results) -> list[dict]:
    rows = []
    for result in results:
        rows.append(
            {
                "id": result.request_id,
                "tokens": len(result.tokens),
                "ids": list(result.tokens),
                "ttft_s": result.ttft_s,
                "latency_s": result.latency_s,
            }
        )
    return rows


def throughput(model: GPT2LM) -> dict:
    sampling = SamplingParams(max_tokens=NEW_TOKENS, temperature=0.0, stop_token_ids=())
    catalog = prompts_for(REQUESTS_PER_LEVEL, PROMPT_TOKENS, salt=5)
    print("warmup", flush=True)
    _batched_wave(model, catalog[:1], sampling, num_blocks=POOL_BLOCKS, max_batch_size=1)
    generate_recompute_wave(model, catalog[:1], sampling, block_size=BLOCK_SIZE)

    systems = [
        ("nano-infer", "Paged KV + continuous batching"),
        ("serial_kv", "Serial KV cache"),
        ("recompute", "Serial recompute"),
    ]
    report = []
    reference_ids: dict[int, list[list[int]]] = {}
    for system_id, label in systems:
        levels = []
        print(f"throughput {system_id}", flush=True)
        for concurrency in CONCURRENCIES:
            rows: list[dict] = []
            wall = 0.0
            for wave_index in range(0, REQUESTS_PER_LEVEL, concurrency):
                wave = catalog[wave_index : wave_index + concurrency]
                if system_id == "recompute":
                    started = time.perf_counter()
                    produced = generate_recompute_wave(model, wave, sampling, block_size=BLOCK_SIZE)
                    wall += time.perf_counter() - started
                    # generate_recompute_wave stamps its own wave start; recompute the
                    # wall from the helper's own clock by using the returned latencies'
                    # span. The explicit timer above is the wave wall, and TTFT/latency
                    # inside the helper are measured from the same wave start.
                    chunk = _baseline_rows(produced)
                elif system_id == "serial_kv":
                    started = time.perf_counter()
                    produced = generate_serial_kv(
                        model,
                        wave,
                        sampling,
                        block_size=BLOCK_SIZE,
                        num_blocks=POOL_BLOCKS,
                        max_prefill_tokens=512,
                    )
                    wall += time.perf_counter() - started
                    chunk = _baseline_rows(produced)
                else:
                    chunk, wave_wall = _batched_wave(
                        model,
                        wave,
                        sampling,
                        num_blocks=POOL_BLOCKS,
                        max_batch_size=max(concurrency, 1),
                    )
                    wall += wave_wall
                if system_id == "nano-infer":
                    reference_ids.setdefault(concurrency, []).extend(row["ids"] for row in chunk)
                else:
                    expected = reference_ids[concurrency][len(rows) : len(rows) + len(chunk)]
                    got = [row["ids"] for row in chunk]
                    if got != expected:
                        raise RuntimeError(f"{system_id} diverged at concurrency {concurrency}")
                rows.extend(chunk)
            summary = _summarize(rows, wall)
            summary["concurrency"] = concurrency
            levels.append(summary)
            print(
                f"  c={concurrency} {summary['throughput_tps']:.1f} tok/s "
                f"ttft_p50={summary['ttft_p50_ms']:.1f}ms lat_p50={summary['latency_p50_ms']:.1f}ms",
                flush=True,
            )
        report.append({"id": system_id, "label": label, "levels": levels})
    return {
        "model": "distilgpt2",
        "prompt_tokens": PROMPT_TOKENS,
        "max_new_tokens": NEW_TOKENS,
        "temperature": 0,
        "requests_per_level": REQUESTS_PER_LEVEL,
        "block_size": BLOCK_SIZE,
        "num_blocks": POOL_BLOCKS,
        "arrival": "Each wave's requests are submitted together before the first token. Latency includes queue wait inside the wave.",
        "outputs_match_across_systems": True,
        "systems": report,
    }


def memory_pressure(model: GPT2LM) -> dict:
    """Same slot budget. Reservation holds prompt+max_tokens up front; paging grows."""
    sampling = SamplingParams(max_tokens=32, temperature=0.0, stop_token_ids=())
    catalog = prompts_for(8, 32, salt=9)
    num_blocks = 20
    print("memory pressure", flush=True)
    systems = []
    for reserve in (False, True):
        name = "reserve" if reserve else "paged"
        rows, wall = _batched_wave(
            model,
            catalog,
            sampling,
            num_blocks=num_blocks,
            max_batch_size=8,
            reserve_full=reserve,
        )
        # A second pass records the high-water running count.
        engine = Engine(
            model,
            block_size=BLOCK_SIZE,
            num_blocks=num_blocks,
            max_batch_size=8,
            max_prefill_tokens=512,
            watermark_blocks=0,
            reserve_full=reserve,
        )
        engine.trace = []
        for index, prompt in enumerate(catalog):
            engine.add_request(prompt, sampling, f"m{index}")
        while engine.has_work():
            engine.step()
        high = max((len(snap["running"]) for snap in engine.trace), default=0)
        summary = _summarize(rows, wall)
        summary["id"] = name
        summary["max_running"] = high
        summary["label"] = "Up-front reservation" if reserve else "Paged incremental"
        systems.append(summary)
        print(f"  {name} {summary['throughput_tps']:.1f} tok/s max_running={high}", flush=True)
    return {
        "model": "distilgpt2",
        "block_size": BLOCK_SIZE,
        "num_blocks": num_blocks,
        "slot_budget": num_blocks * BLOCK_SIZE,
        "prompt_tokens": 32,
        "max_new_tokens": 32,
        "requests": 8,
        "note": "Each sequence needs 32 prompt tokens plus 32 generated tokens (64 tokens, 4 blocks). The pool has 20 blocks, so a full reservation of 4 blocks admits 5 sequences; paging admits on the 2-block prompt and grows.",
        "systems": systems,
    }


def speculative(target: GPT2LM, draft: GPT2LM) -> dict:
    prompt = prompts_for(1, 32, salt=3)[0]
    sampling = SamplingParams(max_tokens=32, temperature=0.0, stop_token_ids=())
    print("speculative decoding", flush=True)
    alone_engine = Engine(target, block_size=BLOCK_SIZE, num_blocks=64, max_batch_size=1, max_prefill_tokens=128)
    started = time.perf_counter()
    alone = alone_engine.generate(prompt, sampling)
    alone_wall = time.perf_counter() - started

    spec_engine = Engine(
        target,
        block_size=BLOCK_SIZE,
        num_blocks=64,
        max_batch_size=1,
        max_prefill_tokens=128,
        draft=draft,
        gamma=4,
    )
    started = time.perf_counter()
    speculated = spec_engine.generate(prompt, sampling, speculative=True)
    spec_wall = time.perf_counter() - started
    if speculated != alone:
        raise RuntimeError("speculative greedy output diverged from the target")
    stats = spec_engine.last_spec_stats or {}
    tokens = len(alone)
    print(
        f"  accept={stats.get('acceptance_rate', 0):.3f} "
        f"target={tokens / alone_wall:.1f} tok/s spec={tokens / spec_wall:.1f} tok/s",
        flush=True,
    )
    return {
        "target": "openai-community/gpt2",
        "draft": "distilgpt2",
        "gamma": 4,
        "prompt_tokens": len(prompt),
        "max_new_tokens": 32,
        "temperature": 0,
        "greedy_match": True,
        "generated_tokens": tokens,
        "target_only_wall_s": alone_wall,
        "speculative_wall_s": spec_wall,
        "target_only_tps": tokens / alone_wall,
        "speculative_tps": tokens / spec_wall,
        **stats,
    }


def build_trace() -> dict:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=64,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        n_inner=64,
        eos_token_id=1,
    )
    model = GPT2LM(config)
    model.eval()
    engine = Engine(
        model,
        block_size=4,
        num_blocks=8,
        max_batch_size=4,
        max_prefill_tokens=8,
        watermark_blocks=0,
    )
    prompts = {
        "A": [4, 5, 6, 7, 8, 9, 10, 11],
        "B": [12, 13, 14, 15, 16, 17, 18, 19],
        "C": [3, 3, 4, 4, 5, 5, 6, 6],
        "D": [9, 8, 7, 6],
    }
    arrivals = {0: ["A", "B"], 3: ["C"], 6: ["D"]}
    sampling = SamplingParams(max_tokens=10, temperature=0.0, stop_token_ids=())
    steps = []
    for index in range(80):
        arrived = []
        for name in arrivals.get(index, []):
            engine.add_request(prompts[name], sampling, name)
            arrived.append(name)
        if not engine.has_work():
            if index > max(arrivals):
                break
            continue
        engine.step()
        snap = dict(engine.last_snapshot or {})
        snap["index"] = len(steps)
        snap["arrived"] = arrived
        snap["note"] = _note(snap)
        steps.append(snap)
    return {
        "block_size": 4,
        "num_blocks": 8,
        "watermark_blocks": 0,
        "max_batch_size": 4,
        "max_prefill_tokens": 8,
        "max_new_tokens": 10,
        "model": {
            "family": "gpt2",
            "n_layer": 2,
            "n_embd": 32,
            "n_head": 4,
            "vocab_size": 64,
            "weights": "seeded random, greedy",
        },
        "prompts": {name: len(ids) for name, ids in prompts.items()},
        "steps": steps,
    }


def _note(snap: dict) -> str:
    phases = [phase for phase in snap.get("phases", []) if phase != "idle"]
    bits = []
    if snap.get("arrived"):
        bits.append("arrived " + ", ".join(snap["arrived"]))
    if phases:
        bits.append("+".join(phases))
    if snap.get("preempted"):
        bits.append("preempted " + ", ".join(snap["preempted"]))
    events = snap.get("events") or []
    produced = [event["id"] for event in events if event.get("token") is not None]
    if produced:
        bits.append("token " + ", ".join(produced))
    return " · ".join(bits) if bits else "idle"


def find_gpt2() -> Path:
    if (GPT2 / "config.json").exists():
        return GPT2
    snapshots = sorted(GPT2.glob("*/config.json"))
    if not snapshots:
        raise FileNotFoundError(f"no gpt2 snapshot under {GPT2}")
    return snapshots[-1].parent


def main() -> None:
    torch.set_num_threads(int(os.environ.get("NANO_INFER_THREADS", "4")))
    torch.manual_seed(0)
    started = time.perf_counter()
    gpt2_path = find_gpt2()
    correctness = [
        logit_error(TINY, "sshleifer/tiny-gpt2"),
        logit_error(DISTIL, "distilgpt2"),
        logit_error(gpt2_path, "openai-community/gpt2"),
    ]
    print("loading distilgpt2 for throughput", flush=True)
    _, distil = load_model(DISTIL)
    throughput_report = throughput(distil)
    memory_report = memory_pressure(distil)
    del distil
    print("loading gpt2 target and distil draft", flush=True)
    _, target = load_model(gpt2_path)
    _, draft = load_model(DISTIL)
    spec_report = speculative(target, draft)
    payload = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware(),
        "correctness": correctness,
        "throughput": throughput_report,
        "memory": memory_report,
        "speculative": spec_report,
        "elapsed_s": time.perf_counter() - started,
    }
    out = ROOT / "benchmarks" / "results.json"
    web = ROOT / "web" / "data" / "benchmarks.json"
    trace_path = ROOT / "web" / "data" / "trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    web.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    out.write_text(text + "\n", encoding="utf-8")
    web.write_text(text + "\n", encoding="utf-8")
    trace = build_trace()
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print(f"wrote {trace_path}", flush=True)


if __name__ == "__main__":
    main()
