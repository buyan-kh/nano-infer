import torch

from nano_infer.config import ModelConfig
from nano_infer.engine import Engine
from nano_infer.sampling import SamplingParams
from tests.support import make_model, small_config


def _clone(model):
    copied = type(model)(model.config)
    copied.load_state_dict(model.state_dict())
    copied.eval()
    return copied


def test_greedy_speculative_matches_target_with_a_different_draft():
    target = make_model(small_config(), seed=1)
    draft_cfg = ModelConfig(
        vocab_size=target.config.vocab_size,
        n_positions=target.config.n_positions,
        n_embd=16,
        n_layer=1,
        n_head=4,
        n_inner=32,
        eos_token_id=target.config.eos_token_id,
    )
    draft = make_model(draft_cfg, seed=99)
    prompt = [4, 10, 11, 12, 13, 2, 7]
    sampling = SamplingParams(max_tokens=9, temperature=0.0, stop_token_ids=())
    alone = Engine(target, block_size=4, num_blocks=32, max_batch_size=1).generate(prompt, sampling)
    spec = Engine(
        target,
        block_size=4,
        num_blocks=32,
        max_batch_size=1,
        draft=draft,
        gamma=4,
    ).generate(prompt, sampling, speculative=True)
    assert spec == alone


def test_identical_draft_accepts_every_proposed_token():
    target = make_model(small_config(), seed=2)
    draft = _clone(target)
    prompt = [6, 6, 7, 8, 9]
    sampling = SamplingParams(max_tokens=8, temperature=0.0, stop_token_ids=())
    engine = Engine(target, block_size=4, num_blocks=32, draft=draft, gamma=3)
    engine.trace = []
    tokens = engine.generate(prompt, sampling, speculative=True)
    alone = Engine(target, block_size=4, num_blocks=32).generate(prompt, sampling)
    assert tokens == alone
    stats = [snap["stats"] for snap in engine.trace if "stats" in snap]
    assert stats
    final = stats[-1]
    assert final["proposed"] > 0
    assert final["accepted_draft"] == final["proposed"]
    # A perfect draft emits gamma + 1 tokens per full round.
    assert final["emitted"] == len(tokens)


def test_speculative_sampling_runs_and_respects_max_tokens():
    target = make_model(small_config(), seed=3)
    draft = make_model(small_config(), seed=4)
    sampling = SamplingParams(max_tokens=5, temperature=0.7, top_k=8, top_p=0.9, seed=11, stop_token_ids=())
    tokens = Engine(target, block_size=4, num_blocks=16, draft=draft, gamma=2).generate(
        [1, 2, 3, 4], sampling, speculative=True
    )
    assert len(tokens) == 5
