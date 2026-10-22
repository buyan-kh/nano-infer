from nano_infer.baseline import generate_recompute_wave, generate_serial_kv
from nano_infer.engine import Engine
from tests.support import greedy, make_model


def test_recompute_and_serial_kv_match_the_engine():
    model = make_model(seed=15)
    prompts = [[4, 5, 6, 7, 8, 9], [1, 2, 3, 4]]
    sampling = greedy(5)
    batched = Engine(
        model, block_size=4, num_blocks=32, max_batch_size=4, max_prefill_tokens=32, watermark_blocks=0
    )
    ids = [batched.add_request(prompt, sampling, f"b{i}") for i, prompt in enumerate(prompts)]
    got = {request_id: [] for request_id in ids}
    while batched.has_work():
        for event in batched.step():
            if event.token_id is not None:
                got[event.request_id].append(event.token_id)
    serial = generate_serial_kv(
        model, prompts, sampling, block_size=4, num_blocks=32, max_prefill_tokens=32
    )
    recompute = generate_recompute_wave(model, prompts, sampling, block_size=4)
    assert [row.tokens for row in serial] == [got[f"b{i}"] for i in range(2)]
    assert [row.tokens for row in recompute] == [got[f"b{i}"] for i in range(2)]
