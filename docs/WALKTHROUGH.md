# Design notes

These are the decisions behind nano-infer, in the order a design review tends to ask them. Measured numbers are copied from `benchmarks/results.json` (22 Oct 2026, Intel Xeon, 4 vCPU, 15.63 GiB, torch 2.14.1+cpu, 4 threads, float32). Tables in the README round to one decimal; ratios below use the unrounded file.

## Why GPT-2

The serving problem (paging, admission, preemption, speculation) is architecture-agnostic. The correctness problem is not. GPT-2 has a public reference implementation and small public weights, so a from-scratch forward pass can be checked token by token. A Llama-shaped model would add RoPE, grouped-query attention, SwiGLU, and RMSNorm. Those change the math inside the block. They do not change the block table, the scheduler, or the HTTP loop. Shipping GPT-2 first means a logit mismatch is a bug in this code, not an ambiguity about which kernel Hugging Face selected.

`weights/tiny-gpt2` is sshleifer/tiny-gpt2 (width 2, 2 layers, 2 heads, vocab 50257). It is not committed; it is downloaded from the hub into weights/ on first use. It is the wrong model for throughput. Throughput and the memory-pressure run use distilgpt2. Speculative decoding uses openai-community/gpt2 as the target and distilgpt2 as the draft, because the draft was distilled from GPT-2 and therefore accepts often.

## Forward pass

`GPT2LM` is the original block, eval-only:

- Token embedding plus a learned position embedding. No dropout.
- Pre-norm residual attention, then pre-norm MLP.
- GELU is the tanh approximation: `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`. A stock `gelu` does not match GPT-2.
- LayerNorm epsilon is `1e-5`.
- Attention scores are scaled by `head_dim ** -0.5`.
- The language-model head is tied to the token embedding. After a checkpoint load the tie is applied again, because the checkpoint may contain a separate `lm_head` copy.
- Hugging Face stores the four projections (`c_attn`, `c_proj`, `c_fc`, `c_proj`) as Conv1D weights of shape `[in, out]`. `nn.Linear` wants `[out, in]`, so those four matrices are transposed on load. Other keys, including the causal `attn.bias` mask and `masked_bias`, are skipped. The mask is applied in the attention code.

The comparison forces `attn_implementation="eager"`. SDPA and FlashAttention are allowed to differ in the last bits. Eager attention is the same matmul this code runs, so a mismatch is meaningful. On the recorded 16-token prompts the max absolute logit error is 0.0 for tiny-gpt2, distilgpt2, and GPT-2, and the final argmax matches. Tests also check that block size does not change logits, that a chunked prefill plus the first greedy token matches a single full forward, and that a batched decode matches three solo decodes.

## Paged KV, and why the kernel is ordinary

The cache is `[n_layer, 2, num_blocks, block_size, n_head, head_dim]` conceptually, stored as separate K and V tensors of `[n_layer, num_blocks, block_size, n_head, head_dim]`. A sequence carries a block table. Token `t` lives at physical block `table[t // block_size]`, slot `t % block_size`.

Writes scatter new K/V into those slots. Reads gather the slots for the current context into a dense workspace, then a causal matmul produces attention. There is no custom CUDA kernel. On CPU a gather plus a matmul is the honest implementation, and it keeps the numerical path identical to a dense cache, which is why block size is an invariant in the tests.

Paging still matters with a boring kernel. A reservation of `prompt + max_tokens` strands slots that a sequence has not used yet. A block allocator can hand a sequence two blocks for a 32-token prompt and a third block only when decode crosses the boundary. Finished sequences return blocks to a free list. The next request reuses them. The thing being scheduled is admission and fragmentation, not FLOPs inside attention.

`BlockAllocator` is a free list plus a set of outstanding ids. Double-free is an error. `blocks_required` is `ceil(tokens / block_size)`. `CapacityError` (a `ValueError`) is raised at admission when `prompt + max_tokens` cannot fit in the pool even with the engine otherwise empty. A request that cannot run alone fails immediately instead of sitting in the queue forever.

## Scheduler

One `schedule()` call does four things, in this order.

1. Reap sequences that already finished and return their blocks.
2. Schedule every running decode. A decode has `pending == 1`: one token has been sampled and not yet written into KV. Decodes may consume the watermark. If a decode needs a new block and the free list is empty, the newest other running sequence is preempted.
3. Continue prefills already in flight, up to `max_prefill_tokens`. If the next prompt token needs a block and none are free, the same preemption path runs, so a prefill cannot stall while holding the sequence in `running`.
4. Admit from the waiting deque while batch size and the prefill budget allow.

Decode-first is the latency choice. A token already sampled should not wait behind a new prompt. Prefill is the throughput hog, so it has an explicit token budget and is not mixed into the decode forward. The engine then runs every scheduled decode in one forward (they all have span 1) and groups prefills by span length for separate forwards. Padding a 1-token decode up to a 32-token prefill would dominate the step on CPU.

The watermark is `watermark_blocks` (default 1) left free when something is already running and a new request wants in. Without it, admission fills the pool, the next decode crosses a block boundary, and the scheduler immediately preempts someone. Decodes ignore the watermark: a running sequence is allowed to take the reserved block.

Preemption is recompute, not swap. The victim's blocks go back to the free list, `num_computed` is set to 0, and the sequence is pushed to the front of the waiting deque. Tokens already sampled stay in `generated_ids`. The prompt plus those tokens are prefilled again later. Greedy output is therefore identical to a run that never preempted; the tests check that against the serial-KV baseline and check that the trace actually contains a preemption. Sequences preempted in the current step are held out of re-admission until the next step, so a victim is not instantly pulled back into the hole it just created.

`reserve_full` is the comparison policy, not a second model. On admission it allocates `ceil((prompt + max_tokens) / block_size)` blocks up front or it waits. The pool, the forward pass, and the sampler are the same objects.

The victim is the newest other running sequence (`running` is append-on-admit, and the search walks it in reverse). Newest-first keeps the sequence that has already paid for the most prefill. It is not a fairness policy. A production scheduler would add priority and aging; this one is small enough to explain on a whiteboard.

## Sampling

`temperature == 0` is argmax and does not draw from the generator, so greedy is reproducible even when a seed is set. Otherwise logits are divided by temperature.

Top-k keeps every logit tied with the k-th. A strict "first k in an unstable order" would drop ties and make tests flake.

Top-p sorts, takes the softmax, and removes tokens whose cumulative mass *before* that token already exceeds `top_p`. The token that crosses the threshold stays. That is the usual nucleus, and it is the line people get backwards.

If the mask produces a non-finite or empty distribution, the sampler falls back to argmax of the original logits.

Stop token ids finish the sequence without appending the token's text. Stop strings are applied by `IncrementalDecoder`, which holds incomplete UTF-8 so a multibyte character split across tokens does not surface as a replacement character and then disappear. A matching stop trims the delta and sets finish reason `stop`. End of `max_tokens` is `length`. EOS is a stop id.

## Speculative decoding

The algorithm is Leviathan, Kalman, and Matias, 2023. The draft proposes `gamma` tokens. The target scores the proposal in one forward.

Greedy: accept the draft token while it equals the target argmax. On the first mismatch, emit the target argmax and drop the rest of the proposal. If the whole proposal matches, emit one bonus token from the target. The bonus is written only when that path actually completes (`took_bonus`). A reject truncates KV to the matched prefix and writes the corrected token. If EOS lands inside the accepted draft tokens, the unverified tail is discarded.

Sampling: accept the draft token with probability `min(1, p/q)`. On a reject, sample from `normalize(max(0, p - q))`. That residual is the distribution that makes the accepted token match the target's distribution. It does not reproduce a particular target-only sample, because the draft consumes its own generator. The draft generator is seeded with `target_seed + 1_000_003` so the two streams do not alias.

Greedy speculative output equals target greedy for any draft. The unit test uses a different draft and an identical draft (the identical draft accepts every proposed token). The recorded pair is GPT-2 and distilgpt2, gamma 4, 32 prompt tokens and 32 new tokens: greedy match is true, 28 proposed, 24 accepted (85.7%), 8 rounds, 4.0 tokens emitted per round. Target only ran at 77.8 tok/s. With the draft, 56.5 tok/s. The draft is half the depth of the target, not an order of magnitude cheaper, and every round still runs a target forward over the proposal. On this CPU the verification does not pay for itself. The implementation is still worth having: the acceptance test is the part interviews ask about, and the measurement says when not to turn it on.

Speculative requests do not share the continuous-batching pool. They own a side allocator and two caches (target and draft). The engine alternates a speculative round with a batched step when both queues have work, under one lock. A batch of speculative sequences is the obvious next step; one sequence is enough to test the math.

## API

FastAPI exposes `GET /health`, `GET /v1/models`, `POST /v1/completions`, and `POST /v1/chat/completions`. Streaming is `text/event-stream` with `data: {json}\n\n` and a terminal `data: [DONE]`. CORS is open so the console, or any local page, can call it.

`AsyncEngineAdapter` runs `engine.step` on a worker thread and parks the event loop for 2 ms when the engine is idle. The lock is around `add_request` and `step`, so concurrent HTTP requests join one scheduler instead of each owning a model. Lifespan starts and stops that loop.

Gaps, on purpose: `n` must be 1, logprobs are rejected, there is no tool calling, and the chat transcript is a plain role rendering rather than a model-specific template. `speculative: true` is an extra field. None of those are required to show batching, paging, or the logit match.

## Baselines and how the numbers were taken

Both baselines call the same `GPT2LM`.

- Serial recompute runs one request at a time and recomputes the full prefix for every new token, through a throwaway paged cache (`prefill_logits`). That is the "no cache" floor.
- Serial KV is batch size 1 with a real cache, one request at a time. That isolates continuous batching from the cache itself.

The throughput workload is fixed in `nano_infer/benchmark.py`: distilgpt2, prompt 32, 16 new tokens, 16 requests per concurrency in `{1, 2, 4, 8}`, temperature 0, block size 16, 128 blocks, prefill budget 512, watermark 0 for the comparison so admission policy is not the thing being measured. nano-infer runs first. Serial KV and serial recompute must emit the same token ids or the script fails. Wall time starts before the wave is submitted and stops when the last request finishes. TTFT is submit to first token, including queue wait inside the wave. p50 and p99 are percentiles of those 16 requests. With 16 samples, p99 is essentially the slow end of the wave. Say that out loud before quoting 68.3 ms against 1954 ms.

At concurrency 1, nano-infer (149.7 tok/s) and serial KV (146.7) agree, which is the sanity check that batching overhead is small when there is nothing to batch. At concurrency 8, nano-infer is 458.5 tok/s against 137.0 for serial KV (3.35×) and 58.2 for recompute (7.88×). TTFT p50 moves from 15.9 ms at c1 to 67.9 ms at c8 for nano-infer, and from 16.0 ms to 977.5 ms for recompute. The recompute line is flat in throughput because it never batches and never caches; concurrency only queues.

The memory run is the place the paged policy can lose. Eight requests, 32+32 tokens, 20 blocks. Reservation needs 4 blocks each and admits 5 (max running 5, 395.9 tok/s). Paging admits all 8 (263.2 tok/s) and then preempts. On CPU, replaying a preempted prompt is most of a prefill, and the wider batch did not pay that back. A swap-based preemption, or a watermark and a tighter admission cap, would change this. The number to defend is the one that was measured.

## What is left

A fused paged-attention kernel on GPU, prefix caching for shared prompts, batched speculative decoding, and a fairer preemption policy (priority, or preempt the sequence with the shortest computed prefix). Chunked prefill and the block table would stay. The CPU gather would not.
