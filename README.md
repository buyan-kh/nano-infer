# nano-infer

A small LLM inference server for GPT-2 family models, written from scratch on top of PyTorch tensors. It implements its own forward pass, paged KV cache, continuous-batching scheduler, sampler, and speculative decoding, behind an OpenAI-compatible streaming API.

## Features

- GPT-2 forward pass with paged attention over a block-allocated KV cache
- Continuous batching with chunked prefill, admission watermark, and recompute preemption
- Temperature, top-k, top-p sampling and stop strings
- Draft/target speculative decoding with rejection sampling
- FastAPI server: /v1/completions and /v1/chat/completions with SSE streaming
- Next.js console that charts benchmarks and replays a recorded scheduler trace

## Results

Measured on a 4 vCPU Intel Xeon, CPU only, float32 (full data in benchmarks/results.json).

- Logits match Hugging Face GPT-2 exactly: max absolute error 0.0 on tiny-gpt2, distilgpt2, and gpt2
- distilgpt2 at concurrency 8: 458.5 tok/s vs 137.0 for serial KV-cached decoding (3.35x) and 58.2 for serial recompute (7.88x)
- Time to first token p50 at concurrency 8: 67.9 ms vs 440.8 ms for serial KV
- Speculative decoding (gpt2 target, distilgpt2 draft): 85.7% acceptance, but slower on this CPU (56.5 vs 77.8 tok/s)

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
python3 -m pytest
NANO_INFER_MODEL=distilgpt2 python3 -m nano_infer.server

curl http://127.0.0.1:8741/v1/completions -H 'content-type: application/json' \
  -d '{"model":"distilgpt2","prompt":"Hello","max_tokens":16,"stream":true}'
```

Benchmarks: `python3 -m nano_infer.benchmark`. Console: `cd web && npm ci && npm run dev`.

## Layout

```
nano_infer/model.py        GPT-2 forward pass and paged attention
nano_infer/cache.py        block allocator and KV tensors
nano_infer/scheduler.py    continuous batching and preemption
nano_infer/sampling.py     sampling and rejection sampling
nano_infer/speculative.py  speculative decoding
nano_infer/engine.py       request lifecycle and step loop
nano_infer/server.py       OpenAI-compatible API
nano_infer/benchmark.py    benchmark harness
tests/                     unit tests and logit checks vs Hugging Face
web/                       Next.js benchmark console
docs/WALKTHROUGH.md        design notes
```
