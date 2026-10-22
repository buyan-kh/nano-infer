"""OpenAI-compatible HTTP API.

Supported:
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`
- streaming via `text/event-stream`
- `temperature`, `top_p`, `top_k`, `max_tokens`, `stop`, `seed`
- `speculative: true` runs the configured draft model (an extra field)

Not supported: `n > 1`, logprobs, penalties, tools, and multi-prompt batches.
Send concurrent requests to exercise continuous batching.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from nano_infer import __version__
from nano_infer.cache import CapacityError
from nano_infer.engine import Engine, StepEvent
from nano_infer.sampling import SamplingParams
from nano_infer.tokenizer import Tokenizer, render_chat


class CompletionRequest(BaseModel):
    model: str = "nano-infer"
    prompt: str | list[int]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int = 1
    logprobs: int | None = None
    speculative: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "nano-infer"
    messages: list[ChatMessage]
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int = 1
    speculative: bool = False


class AsyncEngineAdapter:
    """Background loop that batches whatever the engine has queued."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.queues: dict[str, asyncio.Queue[StepEvent]] = {}
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def _loop(self) -> None:
        while self._running:
            if not self.engine.has_work():
                await asyncio.sleep(0.002)
                continue
            events = await asyncio.to_thread(self.engine.step)
            for event in events:
                queue = self.queues.get(event.request_id)
                if queue is not None:
                    await queue.put(event)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling: SamplingParams,
        request_id: str,
        *,
        speculative: bool = False,
    ) -> AsyncIterator[StepEvent]:
        queue: asyncio.Queue[StepEvent] = asyncio.Queue()
        self.queues[request_id] = queue
        try:
            await asyncio.to_thread(
                lambda: self.engine.add_request(
                    prompt_ids, sampling, request_id, speculative=speculative
                )
            )
            while True:
                event = await queue.get()
                yield event
                if event.finished:
                    break
        finally:
            self.queues.pop(request_id, None)


def create_app(
    engine: Engine,
    *,
    model_name: str,
    eos_token_id: int,
    tokenizer: Tokenizer | None = None,
) -> FastAPI:
    tokenizer = tokenizer or Tokenizer()
    adapter = AsyncEngineAdapter(engine)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        await adapter.start()
        yield
        await adapter.stop()

    app = FastAPI(title="nano-infer", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.engine = engine
    app.state.adapter = adapter
    app.state.model_name = model_name
    app.state.eos_token_id = eos_token_id
    app.state.tokenizer = tokenizer

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models")
    def models() -> dict:
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": created,
                    "owned_by": "nano-infer",
                }
            ],
        }

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest):
        _reject_unsupported(body.n, body.logprobs)
        prompt_ids = _prompt_to_ids(body.prompt, tokenizer)
        return await _respond(
            adapter,
            tokenizer,
            model=body.model,
            prompt_ids=prompt_ids,
            sampling=_params(body, eos_token_id),
            stream=body.stream,
            kind="text",
            speculative=body.speculative,
        )

    @app.post("/v1/chat/completions")
    async def chat(body: ChatRequest):
        _reject_unsupported(body.n, None)
        if not body.messages:
            raise HTTPException(status_code=400, detail="messages is empty")
        try:
            rendered = render_chat([message.model_dump() for message in body.messages])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        prompt_ids = tokenizer.encode(rendered)
        return await _respond(
            adapter,
            tokenizer,
            model=body.model,
            prompt_ids=prompt_ids,
            sampling=_params(body, eos_token_id),
            stream=body.stream,
            kind="chat",
            speculative=body.speculative,
        )

    return app


def create_default_app() -> FastAPI:
    from nano_infer.weights import load_model

    source = os.environ.get("NANO_INFER_MODEL", "weights/tiny-gpt2")
    draft_source = os.environ.get("NANO_INFER_DRAFT") or None
    device = os.environ.get("NANO_INFER_DEVICE", "cpu")
    config, model = load_model(source, device=device)
    draft = load_model(draft_source, device=device)[1] if draft_source else None
    name = os.environ.get("NANO_INFER_MODEL_NAME", PathName.of(source))
    engine = Engine(
        model,
        block_size=int(os.environ.get("NANO_INFER_BLOCK_SIZE", "16")),
        num_blocks=int(os.environ.get("NANO_INFER_NUM_BLOCKS", "128")),
        max_batch_size=int(os.environ.get("NANO_INFER_MAX_BATCH", "8")),
        max_prefill_tokens=int(os.environ.get("NANO_INFER_MAX_PREFILL_TOKENS", "128")),
        watermark_blocks=int(os.environ.get("NANO_INFER_WATERMARK_BLOCKS", "1")),
        tokenizer=Tokenizer(),
        draft=draft,
        gamma=int(os.environ.get("NANO_INFER_GAMMA", "4")),
    )
    return create_app(engine, model_name=name, eos_token_id=config.eos_token_id, tokenizer=Tokenizer())


class PathName:
    @staticmethod
    def of(source: str) -> str:
        text = source.rstrip("/").split("/")[-1]
        return text or "nano-infer"


def main() -> None:
    import uvicorn

    port = int(os.environ.get("NANO_INFER_PORT", "8741"))
    host = os.environ.get("NANO_INFER_HOST", "0.0.0.0")
    uvicorn.run(create_default_app(), host=host, port=port)


def _reject_unsupported(n: int, logprobs: int | None) -> None:
    if n != 1:
        raise HTTPException(status_code=400, detail="only n=1 is supported")
    if logprobs:
        raise HTTPException(status_code=400, detail="logprobs are not supported")


def _prompt_to_ids(prompt: str | list[int], tokenizer: Tokenizer) -> list[int]:
    if isinstance(prompt, str):
        return tokenizer.encode(prompt)
    if isinstance(prompt, list) and all(isinstance(token, int) for token in prompt):
        return [int(token) for token in prompt]
    raise HTTPException(status_code=400, detail="prompt must be a string or a list of token ids")


def _stops(stop: str | list[str] | None) -> tuple[str, ...]:
    if stop is None:
        return ()
    if isinstance(stop, str):
        return (stop,)
    return tuple(stop)


def _params(body: CompletionRequest | ChatRequest, eos_token_id: int) -> SamplingParams:
    try:
        return SamplingParams(
            max_tokens=body.max_tokens,
            temperature=body.temperature,
            top_k=body.top_k,
            top_p=body.top_p,
            seed=body.seed,
            stop_token_ids=(eos_token_id,),
            stop=_stops(body.stop),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def _respond(
    adapter: AsyncEngineAdapter,
    tokenizer: Tokenizer,
    *,
    model: str,
    prompt_ids: list[int],
    sampling: SamplingParams,
    stream: bool,
    kind: str,
    speculative: bool,
):
    if sampling.max_tokens == 0:
        return _empty_response(model, kind, len(prompt_ids))
    request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    try:
        # Validate before the stream starts so a bad prompt is a JSON error.
        if not prompt_ids:
            raise HTTPException(status_code=400, detail="prompt is empty")
        if speculative and adapter.engine.spec is None:
            raise HTTPException(status_code=400, detail="speculative decoding is not enabled on this server")
    except HTTPException:
        raise

    async def events() -> AsyncIterator[StepEvent]:
        try:
            async for event in adapter.generate(
                prompt_ids, sampling, request_id, speculative=speculative
            ):
                yield event
        except CapacityError as exc:
            failed = StepEvent(
                request_id=request_id,
                token_id=None,
                text_delta="",
                finished=True,
                finish_reason="error",
                error=str(exc),
                phase="error",
                prompt_tokens=len(prompt_ids),
                completion_tokens=0,
            )
            yield failed
        except ValueError as exc:
            failed = StepEvent(
                request_id=request_id,
                token_id=None,
                text_delta="",
                finished=True,
                finish_reason="error",
                error=str(exc),
                phase="error",
                prompt_tokens=len(prompt_ids),
                completion_tokens=0,
            )
            yield failed

    if not stream:
        text = ""
        finish = "length"
        prompt_tokens = len(prompt_ids)
        completion_tokens = 0
        error = None
        async for event in events():
            text += event.text_delta
            prompt_tokens = event.prompt_tokens
            completion_tokens = event.completion_tokens
            if event.finished and event.finish_reason:
                finish = event.finish_reason
            if event.error:
                error = event.error
        if error and completion_tokens == 0 and finish == "error":
            raise HTTPException(status_code=400, detail=error)
        return _final_json(
            request_id, created, model, kind, text, finish, prompt_tokens, completion_tokens
        )

    async def byte_stream() -> AsyncIterator[bytes]:
        role_sent = False
        finish = "length"
        async for event in events():
            if event.error and event.completion_tokens == 0:
                payload = {"error": {"message": event.error, "type": "invalid_request_error"}}
                yield _sse(payload)
                yield b"data: [DONE]\n\n"
                return
            if kind == "chat" and not role_sent:
                role_sent = True
                yield _sse(_chat_chunk(request_id, created, model, {"role": "assistant"}, None))
            if event.text_delta:
                if kind == "chat":
                    yield _sse(_chat_chunk(request_id, created, model, {"content": event.text_delta}, None))
                else:
                    yield _sse(_text_chunk(request_id, created, model, event.text_delta, None))
            if event.finished:
                finish = event.finish_reason or finish
        if kind == "chat":
            yield _sse(_chat_chunk(request_id, created, model, {}, finish))
        else:
            yield _sse(_text_chunk(request_id, created, model, "", finish))
        yield b"data: [DONE]\n\n"

    return StreamingResponse(byte_stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


def _empty_response(model: str, kind: str, prompt_tokens: int) -> JSONResponse:
    created = int(time.time())
    request_id = f"cmpl-{uuid.uuid4().hex[:24]}"
    return JSONResponse(
        _final_json(request_id, created, model, kind, "", "length", prompt_tokens, 0)
    )


def _final_json(
    request_id: str,
    created: int,
    model: str,
    kind: str,
    text: str,
    finish: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if kind == "chat":
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                }
            ],
            "usage": usage,
        }
    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
        "usage": usage,
    }


def _chat_chunk(request_id: str, created: int, model: str, delta: dict, finish: str | None) -> dict:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _text_chunk(request_id: str, created: int, model: str, text: str, finish: str | None) -> dict:
    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": finish}],
    }


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
