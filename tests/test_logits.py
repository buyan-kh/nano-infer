import torch

from nano_infer.engine import Engine
from nano_infer.model import prefill_logits
from nano_infer.weights import load_model
from tests.support import greedy, hf_from_model, make_model, small_config


def _hf_logits(hf, token_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        return hf(ids, use_cache=False).logits[0]


def _hf_stepwise(hf, token_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor([token_ids], dtype=torch.long)
    rows = []
    with torch.no_grad():
        out = hf(ids[:, :1], use_cache=True)
        rows.append(out.logits[0, -1])
        past = out.past_key_values
        for index in range(1, ids.shape[1]):
            out = hf(ids[:, index : index + 1], past_key_values=past, use_cache=True)
            rows.append(out.logits[0, -1])
            past = out.past_key_values
    return torch.stack(rows, dim=0)


def test_prefill_matches_huggingface_eager_attention():
    model = make_model(seed=3)
    hf = hf_from_model(model)
    tokens = [4, 8, 15, 16, 23, 42, 7, 1, 9, 11, 13]
    ours = prefill_logits(model, tokens, block_size=4)
    ref = _hf_logits(hf, tokens)
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)


def test_stepwise_decode_matches_full_prefill_and_huggingface():
    model = make_model(seed=5)
    hf = hf_from_model(model)
    tokens = [3, 6, 9, 12, 18, 21, 2, 8]
    full = prefill_logits(model, tokens, block_size=2)
    stepwise = _hf_stepwise(hf, tokens)
    torch.testing.assert_close(full, stepwise, atol=1e-4, rtol=1e-4)

    engine = Engine(model, block_size=2, num_blocks=16, max_batch_size=1, max_prefill_tokens=3)
    # Chunked prefill of the prompt, then greedy continuation, must agree with a
    # single forward on the prompt: the first sampled token is argmax(full[-1]).
    sampling = greedy(max_tokens=1)
    generated = engine.generate(tokens, sampling)
    assert generated == [int(torch.argmax(full[-1]).item())]


def test_block_size_does_not_change_logits():
    model = make_model(seed=8)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    reference = prefill_logits(model, tokens, block_size=16)
    for block_size in (1, 3, 4, 8):
        got = prefill_logits(model, tokens, block_size=block_size)
        torch.testing.assert_close(got, reference, atol=1e-5, rtol=1e-5)


def test_batched_decode_matches_solo_sequences():
    model = make_model(seed=11)
    prompts = [[4, 5, 6, 7, 8], [9, 10, 11], [1, 1, 2, 3, 5, 8, 13]]
    sampling = greedy(max_tokens=5)
    solo = []
    for prompt in prompts:
        solo.append(
            Engine(
                model, block_size=4, num_blocks=32, max_batch_size=1, max_prefill_tokens=64
            ).generate(prompt, sampling)
        )
    batched = Engine(
        model, block_size=4, num_blocks=32, max_batch_size=4, max_prefill_tokens=64, watermark_blocks=0
    )
    ids = [batched.add_request(prompt, sampling, f"p{i}") for i, prompt in enumerate(prompts)]
    seen: dict[str, list[int]] = {request_id: [] for request_id in ids}
    while batched.has_work():
        for event in batched.step():
            if event.token_id is not None:
                seen[event.request_id].append(event.token_id)
    assert [seen[request_id] for request_id in ids] == solo


def test_real_tiny_gpt2_matches_huggingface():
    from nano_infer.weights import ensure_tiny_checkpoint

    tiny = ensure_tiny_checkpoint()
    config, model = load_model(tiny)
    assert config.vocab_size == 50257
    assert config.n_layer == 2
    assert config.n_embd == 2
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(str(tiny), attn_implementation="eager")
    hf.eval()
    tokens = [15496, 11, 995, 0, 50256, 15, 88, 318, 257, 1664, 649, 198]
    ours = prefill_logits(model, tokens, block_size=8)
    with torch.no_grad():
        ref = hf(torch.tensor([tokens])).logits[0]
    torch.testing.assert_close(ours, ref, atol=1e-4, rtol=1e-4)
    # A fixed prompt pins the checkpoint, not just the architecture.
    assert int(torch.argmax(ours[-1]).item()) == int(torch.argmax(ref[-1]).item())
