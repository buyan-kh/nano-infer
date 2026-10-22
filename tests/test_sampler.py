import torch

from nano_infer.sampling import (
    SamplingParams,
    make_generator,
    rejection_distribution,
    sample,
    speculative_accept,
)


def test_temperature_zero_is_argmax_and_ignores_the_generator():
    logits = torch.tensor([0.1, 3.0, 2.5, -1.0])
    params = SamplingParams(temperature=0.0)
    generator = make_generator(0)
    before = generator.get_state().clone()
    assert sample(logits, params, generator) == 1
    assert torch.equal(generator.get_state(), before)


def test_top_k_limits_support():
    logits = torch.tensor([0.0, 5.0, 4.0, -3.0, 1.0])
    params = SamplingParams(temperature=1.0, top_k=2, seed=0)
    seen = {sample(logits, params, make_generator(seed)) for seed in range(40)}
    assert seen <= {1, 2}
    assert seen == {1, 2}


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    # Probabilities after softmax of these logits are about 0.64, 0.23, 0.09, 0.03.
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    params = SamplingParams(temperature=1.0, top_p=0.7)
    seen = {sample(logits, params, make_generator(seed)) for seed in range(50)}
    assert 0 in seen
    assert 3 not in seen


def test_seed_replays_the_same_sample():
    logits = torch.arange(16, dtype=torch.float32)
    params = SamplingParams(temperature=0.8, top_k=8, top_p=0.9)
    first = [sample(logits, params, make_generator(7)) for _ in range(1)]
    second = [sample(logits, params, make_generator(7)) for _ in range(1)]
    assert first == second


def test_rejection_distribution_is_the_positive_residual():
    target = torch.tensor([0.2, 0.8])
    draft = torch.tensor([0.5, 0.5])
    residual = rejection_distribution(target, draft)
    assert torch.allclose(residual, torch.tensor([0.0, 1.0]))


def test_speculative_accept_is_certain_when_target_mass_is_higher():
    target = torch.tensor([0.2, 0.8])
    draft = torch.tensor([0.5, 0.5])
    generator = make_generator(1)
    assert speculative_accept(target, draft, 1, generator)
    # Token 0 has p/q = 0.4, so some seeds reject it.
    decisions = [speculative_accept(target, draft, 0, make_generator(seed)) for seed in range(30)]
    assert any(decisions) and not all(decisions)
