import asyncio

from fastapi.testclient import TestClient

from nano_infer.engine import Engine
from nano_infer.server import AsyncEngineAdapter, create_app
from nano_infer.sampling import SamplingParams
from nano_infer.tokenizer import Tokenizer
from tests.support import make_model, small_config


def _app():
    # Full GPT-2 vocab so tiktoken ids are in range. The hidden size stays tiny.
    config = small_config(vocab_size=50257, n_embd=16, n_head=2, n_inner=32, n_layer=1, eos_token_id=50256)
    model = make_model(config, seed=0)
    tokenizer = Tokenizer()
    engine = Engine(
        model,
        block_size=8,
        num_blocks=32,
        max_batch_size=4,
        max_prefill_tokens=64,
        watermark_blocks=0,
        tokenizer=tokenizer,
        draft=make_model(config, seed=1),
        gamma=2,
    )
    return create_app(engine, model_name="tiny", eos_token_id=50256, tokenizer=tokenizer), engine, tokenizer


def test_health_and_models():
    app, _, _ = _app()
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        payload = client.get("/v1/models").json()
        assert payload["data"][0]["id"] == "tiny"


def test_completion_and_chat_round_trip():
    app, engine, tokenizer = _app()
    with TestClient(app) as client:
        refused = client.post("/v1/completions", json={"model": "tiny", "prompt": "Hi", "n": 2})
        assert refused.status_code == 400
        response = client.post(
            "/v1/completions",
            json={"model": "tiny", "prompt": "Hello", "max_tokens": 4, "temperature": 0},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "text_completion"
        assert body["usage"]["prompt_tokens"] == len(tokenizer.encode("Hello"))
        assert body["usage"]["completion_tokens"] >= 1
        assert body["choices"][0]["finish_reason"] in {"stop", "length"}

        chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "tiny",
                "messages": [{"role": "user", "content": "Say ping"}],
                "max_tokens": 3,
                "temperature": 0,
            },
        )
        assert chat.status_code == 200
        assert chat.json()["choices"][0]["message"]["role"] == "assistant"


def test_streaming_emits_sse_and_done():
    app, _, _ = _app()
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": "tiny",
                "prompt": "Stream",
                "max_tokens": 3,
                "temperature": 0.8,
                "seed": 2,
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            raw = "".join(response.iter_text())
    assert raw.startswith("data: ")
    assert "data: [DONE]" in raw
    assert "text_completion" in raw


def test_stop_field_is_accepted():
    app, _, _ = _app()
    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "tiny", "prompt": "Hello", "max_tokens": 2, "temperature": 0, "stop": ["\n"]},
        )
    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] >= 1


def test_async_batch_matches_solo_greedy():
    _, engine, tokenizer = _app()
    prompts = [tokenizer.encode("alpha "), tokenizer.encode("beta wave")]
    sampling = SamplingParams(max_tokens=4, temperature=0.0, stop_token_ids=(50256,))
    fresh = Engine(
        engine.model,
        block_size=8,
        num_blocks=32,
        max_batch_size=4,
        max_prefill_tokens=64,
        watermark_blocks=0,
        tokenizer=tokenizer,
    )

    async def collect():
        adapter = AsyncEngineAdapter(fresh)
        await adapter.start()
        try:

            async def one(prompt, request_id):
                tokens = []
                async for event in adapter.generate(prompt, sampling, request_id):
                    if event.error:
                        raise RuntimeError(event.error)
                    if event.token_id is not None:
                        tokens.append(event.token_id)
                return tokens

            return await asyncio.gather(one(prompts[0], "a"), one(prompts[1], "b"))
        finally:
            await adapter.stop()

    batched = asyncio.run(collect())
    solo = [
        Engine(
            engine.model, block_size=8, num_blocks=32, max_batch_size=1, tokenizer=tokenizer
        ).generate(prompt, sampling)
        for prompt in prompts
    ]
    assert list(batched) == solo
