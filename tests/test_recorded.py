"""Guard the recorded benchmark file without remeasuring it."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_recorded_benchmark_matches_the_console_copy():
    recorded = _load(ROOT / "benchmarks" / "results.json")
    console = _load(ROOT / "web" / "data" / "benchmarks.json")
    assert console == recorded

    assert recorded["hardware"]["device"] == "cpu"
    assert recorded["hardware"]["vcpu"] == 4
    assert recorded["throughput"]["outputs_match_across_systems"] is True

    for row in recorded["correctness"]:
        assert row["max_abs_logit_error"] == 0.0
        assert row["final_token_argmax_match"] is True
        assert row["hf_attention"] == "eager"

    systems = {system["id"]: system for system in recorded["throughput"]["systems"]}
    assert set(systems) == {"nano-infer", "serial_kv", "recompute"}
    for system in systems.values():
        assert [level["concurrency"] for level in system["levels"]] == [1, 2, 4, 8]
        for level in system["levels"]:
            assert level["generated_tokens"] == 256
            assert level["throughput_tps"] > 0
            assert level["ttft_p50_ms"] > 0
            assert level["latency_p99_ms"] >= level["latency_p50_ms"]

    batched8 = next(level for level in systems["nano-infer"]["levels"] if level["concurrency"] == 8)
    recompute8 = next(level for level in systems["recompute"]["levels"] if level["concurrency"] == 8)
    assert batched8["throughput_tps"] > recompute8["throughput_tps"]

    memory = {system["id"]: system for system in recorded["memory"]["systems"]}
    assert memory["paged"]["max_running"] > memory["reserve"]["max_running"]
    assert memory["reserve"]["throughput_tps"] > memory["paged"]["throughput_tps"]

    spec = recorded["speculative"]
    assert spec["greedy_match"] is True
    assert spec["proposed"] > 0
    assert spec["accepted_draft"] <= spec["proposed"]
    assert spec["acceptance_rate"] == spec["accepted_draft"] / spec["proposed"]
    assert spec["target_only_tps"] > spec["speculative_tps"] > 0


def test_scheduler_trace_includes_preemption():
    trace = _load(ROOT / "web" / "data" / "trace.json")
    assert trace["num_blocks"] > 0
    assert trace["block_size"] > 0
    assert trace["steps"]
    assert any(step["preempted"] for step in trace["steps"])
    for step in trace["steps"]:
        assert len(step["occupancy"]) == trace["num_blocks"]
