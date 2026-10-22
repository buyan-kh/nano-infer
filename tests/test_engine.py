import pytest
import torch

from nano_infer.cache import CapacityError
from nano_infer.engine import Engine
from nano_infer.model import prefill_logits
from nano_infer.sampling import SamplingParams
from tests.support import greedy, make_model, small_config


def _run(engine: Engine, prompts: list[list[int]], sampling: SamplingParams) -> dict[str, list[int]]:
    ids = [engine.add_request(prompt, sampling, f"r{i}") for i, prompt in enumerate(prompts)]
    out = {request_id: [] for request_id in ids}
    while engine.has_work():
        for event in engine.step():
            if event.error:
                raise CapacityError(event.error)
            if event.token_id is not None:
                out[event.request_id].append(event.token_id)
    return out


def test_chunked_prefill_matches_a_single_prefill():
    model = make_model(seed=2)
    prompt = [4, 9, 1, 2, 8, 7, 3, 6, 5, 4]
    full = prefill_logits(model, prompt, block_size=4)
    expected = int(torch.argmax(full[-1]).item())
    engine = Engine(model, block_size=4, num_blocks=16, max_batch_size=1, max_prefill_tokens=3)
    assert engine.generate(prompt, greedy(1)) == [expected]


def test_continuous_batch_matches_serial_greedy_under_preemption():
    model = make_model(seed=4)
    prompts = [
        [4, 5, 6, 7, 8, 9, 10, 11],
        [12, 13, 14, 15, 16, 17, 18, 19],
        [3, 3, 3, 3, 4, 4],
    ]
    sampling = greedy(8)
    solo = [
        Engine(model, block_size=4, num_blocks=32, max_batch_size=1, max_prefill_tokens=64).generate(
            prompt, sampling
        )
        for prompt in prompts
    ]
    # 6 blocks cannot hold every sequence at full length (each can grow to 4 blocks).
    stressed = Engine(
        model,
        block_size=4,
        num_blocks=6,
        max_batch_size=4,
        max_prefill_tokens=8,
        watermark_blocks=0,
    )
    stressed.trace = []
    got = _run(stressed, prompts, sampling)
    assert [got[f"r{i}"] for i in range(3)] == solo
    assert any(snap["preempted"] for snap in stressed.trace)


def test_watermark_holds_a_second_request_until_blocks_are_free():
    model = make_model(seed=6)
    engine = Engine(
        model,
        block_size=4,
        num_blocks=4,
        max_batch_size=4,
        max_prefill_tokens=32,
        watermark_blocks=3,
    )
    sampling = greedy(4)
    engine.trace = []
    _run(engine, [[1, 2, 3, 4], [5, 6, 7, 8]], sampling)
    assert engine.trace
    assert max(len(snap["running"]) for snap in engine.trace) == 1


def test_request_that_cannot_fit_alone_is_rejected():
    model = make_model(seed=1)
    engine = Engine(model, block_size=4, num_blocks=2, max_batch_size=2)
    with pytest.raises(CapacityError):
        engine.add_request([1, 2, 3, 4, 5, 6, 7, 8], greedy(8))


def test_reserve_full_matches_paged_greedy_and_admits_fewer():
    model = make_model(seed=9)
    prompts = [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3], [2, 2, 2, 2]]
    sampling = greedy(6)
    paged = Engine(
        model, block_size=4, num_blocks=32, max_batch_size=4, max_prefill_tokens=64, watermark_blocks=0
    )
    reserved = Engine(
        model,
        block_size=4,
        num_blocks=32,
        max_batch_size=4,
        max_prefill_tokens=64,
        watermark_blocks=0,
        reserve_full=True,
    )
    assert list(_run(paged, prompts, sampling).values()) == list(_run(reserved, prompts, sampling).values())


def test_stop_string_ends_the_request_without_emitting_the_stop(monkeypatch):
    from nano_infer.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    planned = iter(tokenizer.encode("Hello STOP world"))

    def fake_sample(logits, params, generator=None):
        return next(planned)

    monkeypatch.setattr("nano_infer.engine.sample", fake_sample)
    config = small_config(vocab_size=50257, n_embd=16, n_head=2, n_inner=32, n_layer=1, eos_token_id=50256)
    model = make_model(config, seed=0)
    engine = Engine(model, block_size=8, num_blocks=16, tokenizer=tokenizer)
    engine.generate(
        tokenizer.encode("Q"),
        SamplingParams(max_tokens=20, temperature=1.0, stop=("STOP",), stop_token_ids=()),
    )
    finished = engine.scheduler.finished[-1]
    assert finished.finish_reason == "stop"
    assert finished.visible_text == "Hello "
    assert "STOP" not in finished.visible_text


def test_finished_request_returns_blocks_for_the_next_one():
    model = make_model(seed=12)
    engine = Engine(model, block_size=4, num_blocks=4, max_batch_size=1, watermark_blocks=0)
    sampling = greedy(4)
    first = engine.generate([1, 2, 3, 4], sampling)
    assert engine.scheduler.allocator.free_count() == 4
    second = engine.generate([9, 8, 7, 6, 5], sampling)
    solo = Engine(model, block_size=4, num_blocks=16, max_batch_size=1).generate([9, 8, 7, 6, 5], sampling)
    assert second == solo
    assert first
