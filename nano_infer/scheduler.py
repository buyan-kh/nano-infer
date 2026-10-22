"""Iteration-level scheduler for continuous batching.

Policy, in one step:

1. Reap finished sequences and return their blocks to the free list.
2. Schedule every running decode (exactly one pending token). Decodes may
   consume the watermark; a decode that cannot grow preempts the newest other
   running sequence. Preemption drops KV only. Already chosen tokens are
   replayed through prefill, so greedy outputs do not change.
3. Continue chunked prefills already in flight, up to `max_prefill_tokens`.
4. Admit waiting sequences into the leftover budget. New admissions leave
   `watermark_blocks` free whenever anything is already running, so a decode
   can cross a block boundary without immediately preempting.

`reserve_full` is the contiguous-reservation baseline: each sequence is given
`ceil((prompt + max_tokens) / block_size)` blocks up front and is not admitted
until that reservation fits. It uses the same block pool, so the comparison
with paging is about allocation policy rather than a second model.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from nano_infer.cache import BlockAllocator, CapacityError, blocks_required
from nano_infer.sampling import SamplingParams, make_generator
from nano_infer.tokenizer import IncrementalDecoder


@dataclass
class Sequence:
    request_id: str
    prompt_ids: list[int]
    sampling: SamplingParams
    generated_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    num_computed: int = 0
    status: str = "waiting"
    finish_reason: str | None = None
    error: str | None = None
    arrival_time: float = 0.0
    first_token_time: float | None = None
    finish_time: float | None = None
    generator: object | None = None
    decoder: IncrementalDecoder | None = None
    visible_text: str = ""

    def __post_init__(self) -> None:
        if self.generator is None and self.sampling.seed is not None:
            self.generator = make_generator(self.sampling.seed)

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_ids + self.generated_ids

    @property
    def pending(self) -> int:
        return len(self.token_ids) - self.num_computed


@dataclass
class ScheduledPrefill:
    seq: Sequence
    token_ids: list[int]


@dataclass
class ScheduledDecode:
    seq: Sequence
    token_ids: list[int]


@dataclass
class Schedule:
    prefills: list[ScheduledPrefill]
    decodes: list[ScheduledDecode]
    preempted: list[str]
    failed: list[Sequence]


class Scheduler:
    def __init__(
        self,
        allocator: BlockAllocator,
        *,
        block_size: int,
        max_batch_size: int,
        max_prefill_tokens: int,
        watermark_blocks: int = 1,
        reserve_full: bool = False,
    ) -> None:
        if block_size <= 0 or max_batch_size <= 0 or max_prefill_tokens <= 0:
            raise ValueError("scheduler limits must be positive")
        if watermark_blocks < 0:
            raise ValueError("watermark_blocks must be >= 0")
        self.allocator = allocator
        self.block_size = block_size
        self.max_batch_size = max_batch_size
        self.max_prefill_tokens = max_prefill_tokens
        self.watermark_blocks = watermark_blocks
        self.reserve_full = reserve_full
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        full = blocks_required(len(seq.prompt_ids) + seq.sampling.max_tokens, self.block_size)
        if full > self.allocator.num_blocks:
            raise CapacityError(
                f"request needs {full} blocks and the pool has {self.allocator.num_blocks}"
            )
        seq.arrival_time = seq.arrival_time or time.perf_counter()
        seq.status = "waiting"
        self.waiting.append(seq)

    def has_pending(self) -> bool:
        if self.waiting:
            return True
        return any(seq.finish_reason is None for seq in self.running)

    def schedule(self) -> Schedule:
        preempted: list[str] = []
        blocked = set(self._reap())
        decodes: list[ScheduledDecode] = []
        prefills: list[ScheduledPrefill] = []
        failed: list[Sequence] = []

        for seq in list(self.running):
            if seq.request_id in blocked or seq not in self.running:
                continue
            if seq.pending != 1:
                continue
            if not self._ensure_span(seq, seq.num_computed + 1, preempted):
                failed.append(self._fail(seq, "unable to allocate a KV block for decode"))
                continue
            if seq not in self.running:
                continue
            decodes.append(ScheduledDecode(seq=seq, token_ids=[seq.token_ids[seq.num_computed]]))

        decodes = [item for item in decodes if item.seq in self.running and item.seq.finish_reason is None]

        budget = self.max_prefill_tokens
        for seq in list(self.running):
            if budget <= 0 or seq not in self.running or seq.finish_reason is not None:
                continue
            if seq.pending <= 1:
                continue
            fit = self._fittable(seq, respect_watermark=False)
            if fit <= 0:
                # The next prompt token needs a new block and the free list is empty.
                # Ask ensure to preempt rather than leaving the prefill stuck.
                if not self._ensure_span(seq, seq.num_computed + 1, preempted):
                    failed.append(self._fail(seq, "unable to allocate a KV block for prefill"))
                    continue
                fit = self._fittable(seq, respect_watermark=False)
            take = min(seq.pending, budget, fit)
            if take <= 0:
                continue
            if not self._ensure_span(seq, seq.num_computed + take, preempted):
                continue
            prefills.append(
                ScheduledPrefill(seq=seq, token_ids=seq.token_ids[seq.num_computed : seq.num_computed + take])
            )
            budget -= take

        held: list[Sequence] = []
        while self.waiting and len(self.running) < self.max_batch_size and budget > 0:
            seq = self.waiting[0]
            if seq.request_id in set(preempted):
                held.append(self.waiting.popleft())
                continue
            full = blocks_required(len(seq.prompt_ids) + seq.sampling.max_tokens, self.block_size)
            if full > self.allocator.num_blocks:
                self.waiting.popleft()
                failed.append(self._fail(seq, "prompt plus max_tokens does not fit in the KV cache"))
                continue
            respect = (not self.reserve_full) and len(self.running) > 0
            if self.reserve_full and not seq.block_ids:
                if self.allocator.free_count() < full:
                    break
                seq.block_ids.extend(self.allocator.allocate(full))
            fit = self._fittable(seq, respect_watermark=respect)
            want = 1 if seq.pending == 1 else min(seq.pending, budget)
            take = min(want, fit)
            if take <= 0:
                break
            self.waiting.popleft()
            seq.status = "running"
            self.running.append(seq)
            if not self._ensure_span(seq, seq.num_computed + take, preempted):
                failed.append(self._fail(seq, "admission allocation failed"))
                continue
            if seq.pending == 1:
                decodes.append(ScheduledDecode(seq=seq, token_ids=[seq.token_ids[seq.num_computed]]))
            else:
                prefills.append(
                    ScheduledPrefill(
                        seq=seq, token_ids=seq.token_ids[seq.num_computed : seq.num_computed + take]
                    )
                )
                budget -= take

        for seq in reversed(held):
            self.waiting.appendleft(seq)

        decodes = [item for item in decodes if item.seq in self.running and item.seq.finish_reason is None]
        prefills = [item for item in prefills if item.seq in self.running and item.seq.finish_reason is None]
        return Schedule(prefills=prefills, decodes=decodes, preempted=preempted, failed=failed)

    def _reap(self) -> list[str]:
        alive: list[Sequence] = []
        for seq in self.running:
            if seq.finish_reason is not None or seq.status == "finished":
                self._release(seq)
                self.finished.append(seq)
            else:
                alive.append(seq)
        self.running = alive
        return []

    def _fittable(self, seq: Sequence, *, respect_watermark: bool) -> int:
        if self.reserve_full:
            if not seq.block_ids:
                full = blocks_required(len(seq.prompt_ids) + seq.sampling.max_tokens, self.block_size)
                if self.allocator.free_count() < full:
                    return 0
                return seq.pending
            return len(seq.block_ids) * self.block_size - seq.num_computed
        free_blocks = self.allocator.free_count()
        if respect_watermark:
            free_blocks = max(0, free_blocks - self.watermark_blocks)
        total = len(seq.block_ids) + free_blocks
        return total * self.block_size - seq.num_computed

    def _ensure_span(self, seq: Sequence, target_len: int, preempted: list[str]) -> bool:
        if self.reserve_full:
            return blocks_required(target_len, self.block_size) <= len(seq.block_ids)
        guard = 0
        while blocks_required(target_len, self.block_size) > len(seq.block_ids):
            guard += 1
            if guard > self.allocator.num_blocks + 1:
                return False
            if self.allocator.free_count() > 0:
                seq.block_ids.extend(self.allocator.allocate(1))
                continue
            victim = self._victim(seq)
            if victim is None:
                return False
            preempted.append(victim.request_id)
            self._preempt(victim)
        return True

    def _victim(self, protected: Sequence) -> Sequence | None:
        for seq in reversed(self.running):
            if seq is not protected and seq.finish_reason is None:
                return seq
        return None

    def _preempt(self, seq: Sequence) -> None:
        if seq in self.running:
            self.running.remove(seq)
        self._release(seq)
        seq.num_computed = 0
        seq.status = "waiting"
        self.waiting.appendleft(seq)

    def _release(self, seq: Sequence) -> None:
        if seq.block_ids:
            self.allocator.free(list(seq.block_ids))
            seq.block_ids = []

    def _fail(self, seq: Sequence, message: str) -> Sequence:
        if seq in self.running:
            self.running.remove(seq)
        self._release(seq)
        seq.status = "finished"
        seq.finish_reason = "error"
        seq.error = message
        seq.finish_time = time.perf_counter()
        self.finished.append(seq)
        return seq
