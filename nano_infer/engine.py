"""Engine: one model, a paged cache, and the continuous-batching scheduler.

A step runs every scheduled decode in one forward, then every scheduled prefill
in another. Prefill and decode are separate because their shapes differ: a
prefill writes a span of prompt tokens, a decode writes exactly one token per
running sequence. New requests join on a later step without waiting for the
batch to drain.

Speculative requests use a side cache so a draft model can verify against the
same target weights. They alternate with batch steps when both are busy.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field

import torch

from nano_infer.cache import BlockAllocator, CapacityError, KVCache, stack_block_tables
from nano_infer.model import GPT2LM, model_device
from nano_infer.sampling import SamplingParams, sample
from nano_infer.scheduler import Schedule, Scheduler, Sequence
from nano_infer.speculative import SpeculativeDecoder, SpecRun
from nano_infer.tokenizer import IncrementalDecoder, Tokenizer


@dataclass
class StepEvent:
    request_id: str
    token_id: int | None
    text_delta: str
    finished: bool
    finish_reason: str | None
    error: str | None
    phase: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class Engine:
    model: GPT2LM
    block_size: int = 16
    num_blocks: int = 64
    max_batch_size: int = 8
    max_prefill_tokens: int = 128
    watermark_blocks: int = 1
    reserve_full: bool = False
    tokenizer: Tokenizer | None = None
    draft: GPT2LM | None = None
    gamma: int = 4
    cache: KVCache = field(init=False)
    scheduler: Scheduler = field(init=False)
    device: torch.device = field(init=False)
    spec: SpeculativeDecoder | None = field(init=False, default=None)
    spec_queue: list[Sequence] = field(init=False, default_factory=list)
    spec_run: SpecRun | None = field(init=False, default=None)
    last_spec_stats: dict | None = field(init=False, default=None)
    completed: list[Sequence] = field(init=False, default_factory=list)
    last_snapshot: dict | None = field(init=False, default=None)
    trace: list[dict] | None = None
    _ids: itertools.count = field(init=False, default_factory=lambda: itertools.count(1))
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _prefer_spec: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.model.eval()
        self.device = model_device(self.model)
        self.cache = KVCache.allocate(
            self.model.config, self.num_blocks, self.block_size, device=self.device
        )
        allocator = BlockAllocator(self.num_blocks)
        self.scheduler = Scheduler(
            allocator,
            block_size=self.block_size,
            max_batch_size=self.max_batch_size,
            max_prefill_tokens=self.max_prefill_tokens,
            watermark_blocks=self.watermark_blocks,
            reserve_full=self.reserve_full,
        )
        if self.draft is not None:
            self.draft.eval()
            self.spec = SpeculativeDecoder(
                self.model,
                self.draft,
                block_size=self.block_size,
                num_blocks=self.num_blocks,
                gamma=self.gamma,
            )
        self.spec_queue = []
        self.completed = []

    def add_request(
        self,
        prompt_ids: list[int],
        sampling: SamplingParams,
        request_id: str | None = None,
        *,
        speculative: bool = False,
    ) -> str:
        with self._lock:
            return self._add_unlocked(prompt_ids, sampling, request_id, speculative)

    def _add_unlocked(
        self,
        prompt_ids: list[int],
        sampling: SamplingParams,
        request_id: str | None,
        speculative: bool,
    ) -> str:
        if not prompt_ids:
            raise ValueError("prompt is empty")
        if any(token < 0 or token >= self.model.config.vocab_size for token in prompt_ids):
            raise ValueError("prompt token id out of range")
        if len(prompt_ids) + sampling.max_tokens > self.model.config.n_positions:
            raise ValueError("prompt plus max_tokens exceeds the context window")
        if speculative and self.spec is None:
            raise RuntimeError("speculative decoding requires a draft model")
        if sampling.stop and self.tokenizer is None:
            raise RuntimeError("stop strings require a tokenizer")
        request_id = request_id or f"req-{next(self._ids)}"
        decoder = IncrementalDecoder(self.tokenizer) if self.tokenizer is not None else None
        seq = Sequence(
            request_id=request_id,
            prompt_ids=list(prompt_ids),
            sampling=sampling,
            decoder=decoder,
            arrival_time=time.perf_counter(),
        )
        if speculative:
            self.spec_queue.append(seq)
        else:
            self.scheduler.add(seq)
        return request_id

    def has_work(self) -> bool:
        if self.spec_run is not None or self.spec_queue:
            return True
        if self.scheduler.waiting:
            return True
        if self.scheduler.running:
            return True
        return False

    def step(self) -> list[StepEvent]:
        with self._lock:
            with torch.inference_mode():
                return self._step_unlocked()

    def _step_unlocked(self) -> list[StepEvent]:
        batch_busy = self.scheduler.has_pending() or bool(self.scheduler.running)
        spec_busy = self.spec_run is not None or bool(self.spec_queue)
        if spec_busy and (self._prefer_spec or not batch_busy):
            self._prefer_spec = False
            events = self._spec_step()
        elif batch_busy:
            self._prefer_spec = spec_busy
            events = self._batch_step()
        else:
            events = []
        return events

    def _batch_step(self) -> list[StepEvent]:
        schedule = self.scheduler.schedule()
        events: list[StepEvent] = []
        for seq in schedule.failed:
            events.append(self._event(seq, None, "", "error"))
            self.completed.append(seq)
        if schedule.decodes:
            events.extend(self._execute(schedule, decode=True))
        if schedule.prefills:
            events.extend(self._execute(schedule, decode=False))
        self._snapshot(schedule, events)
        return events

    def _execute(self, schedule: Schedule, *, decode: bool) -> list[StepEvent]:
        items = schedule.decodes if decode else schedule.prefills
        if not items:
            return []
        # Group prefills by span length so one matmul covers every row. Decodes
        # are already length 1.
        groups: dict[int, list] = {}
        for item in items:
            groups.setdefault(len(item.token_ids), []).append(item)
        events: list[StepEvent] = []
        phase = "decode" if decode else "prefill"
        for group in groups.values():
            length = len(group[0].token_ids)
            input_ids = torch.tensor([item.token_ids for item in group], dtype=torch.long, device=self.device)
            write_starts = torch.tensor(
                [item.seq.num_computed for item in group], dtype=torch.long, device=self.device
            )
            context_lens = write_starts + length
            tables = stack_block_tables([item.seq.block_ids for item in group], self.device)
            logits = self.model(input_ids, self.cache, tables, context_lens, write_starts)
            for row, item in enumerate(group):
                item.seq.num_computed += length
                if item.seq.num_computed != len(item.seq.token_ids):
                    continue
                token = sample(logits[row, -1], item.seq.sampling, _generator(item.seq))
                events.extend(self._push_token(item.seq, token, phase))
        return events

    def _spec_step(self) -> list[StepEvent]:
        assert self.spec is not None
        if self.spec_run is None:
            if not self.spec_queue:
                return []
            seq = self.spec_queue.pop(0)
            self.spec_run = self.spec.start(seq)
        run = self.spec_run
        assert run is not None
        tokens = self.spec.round(run)
        events: list[StepEvent] = []
        for token in tokens:
            if run.seq.finish_reason is not None:
                break
            pushed = self._push_token(run.seq, token, "speculative")
            events.extend(pushed)
            if run.seq.finish_reason is not None:
                break
        if run.seq.finish_reason is not None:
            stats = run.stats
            self.last_spec_stats = {
                "proposed": stats.proposed,
                "accepted_draft": stats.accepted_draft,
                "rounds": stats.rounds,
                "emitted": stats.emitted,
                "acceptance_rate": stats.acceptance_rate,
                "mean_emitted_per_round": stats.mean_emitted_per_round,
            }
            self.spec.finish(run)
            self.completed.append(run.seq)
            self.spec_run = None
        self._snapshot_spec(run, events)
        return events

    def _push_token(self, seq: Sequence, token: int, phase: str) -> list[StepEvent]:
        now = time.perf_counter()
        seq.generated_ids.append(int(token))
        if seq.first_token_time is None:
            seq.first_token_time = now
        delta = ""
        reason = None
        if int(token) in seq.sampling.stop_token_ids:
            reason = "stop"
        elif seq.decoder is not None:
            delta, stop_reason = seq.decoder.push(int(token), seq.sampling.stop)
            seq.visible_text = seq.decoder.text
            if stop_reason:
                reason = "stop"
        if reason is None and len(seq.generated_ids) >= seq.sampling.max_tokens:
            reason = "length"
        if reason is not None:
            seq.finish_reason = reason
            seq.status = "finished"
            seq.finish_time = now
        return [self._event(seq, int(token), delta, phase)]

    def _event(self, seq: Sequence, token: int | None, delta: str, phase: str) -> StepEvent:
        return StepEvent(
            request_id=seq.request_id,
            token_id=token,
            text_delta=delta,
            finished=seq.finish_reason is not None,
            finish_reason=seq.finish_reason,
            error=seq.error,
            phase=phase,
            prompt_tokens=len(seq.prompt_ids),
            completion_tokens=len(seq.generated_ids),
        )

    def generate(
        self,
        prompt_ids: list[int],
        sampling: SamplingParams,
        *,
        request_id: str | None = None,
        speculative: bool = False,
    ) -> list[int]:
        """Run until this request finishes. The engine must not already be busy."""
        if self.has_work():
            raise RuntimeError("generate() requires an idle engine")
        self.add_request(prompt_ids, sampling, request_id, speculative=speculative)
        tokens: list[int] = []
        while self.has_work():
            for event in self.step():
                if event.error:
                    raise CapacityError(event.error)
                if event.token_id is not None:
                    tokens.append(event.token_id)
        return tokens

    def occupancy(self) -> list[str | None]:
        owners: list[str | None] = [None] * self.num_blocks
        for seq in self.scheduler.running:
            for block in seq.block_ids:
                owners[block] = seq.request_id
        return owners

    def _snapshot(self, schedule: Schedule, events: list[StepEvent]) -> None:
        phases = []
        if schedule.decodes:
            phases.append("decode")
        if schedule.prefills:
            phases.append("prefill")
        if schedule.preempted:
            phases.append("preempt")
        if not phases:
            phases.append("idle")
        snap = {
            "phases": phases,
            "preempted": list(schedule.preempted),
            "waiting": [seq.request_id for seq in self.scheduler.waiting],
            "running": [_seq_view(seq) for seq in self.scheduler.running],
            "finished": [seq.request_id for seq in self.scheduler.finished],
            "occupancy": self.occupancy(),
            "events": [
                {"id": event.request_id, "token": event.token_id, "phase": event.phase, "finished": event.finished}
                for event in events
            ],
            "free_blocks": self.scheduler.allocator.free_count(),
        }
        self.last_snapshot = snap
        if self.trace is not None:
            self.trace.append(snap)

    def _snapshot_spec(self, run: SpecRun, events: list[StepEvent]) -> None:
        owners: list[str | None] = [None] * self.spec.target_alloc.num_blocks if self.spec else []
        if self.spec is not None:
            for block in run.target_blocks:
                if block < len(owners):
                    owners[block] = run.seq.request_id
        snap = {
            "phases": ["speculative"],
            "preempted": [],
            "waiting": [seq.request_id for seq in self.spec_queue],
            "running": [_seq_view(run.seq)],
            "finished": [],
            "occupancy": owners,
            "events": [
                {"id": event.request_id, "token": event.token_id, "phase": event.phase, "finished": event.finished}
                for event in events
            ],
            "free_blocks": self.spec.target_alloc.free_count() if self.spec else 0,
            "stats": {
                "proposed": run.stats.proposed,
                "accepted_draft": run.stats.accepted_draft,
                "rounds": run.stats.rounds,
                "emitted": run.stats.emitted,
            },
        }
        self.last_snapshot = snap
        if self.trace is not None:
            self.trace.append(snap)


def _generator(seq: Sequence) -> torch.Generator | None:
    if isinstance(seq.generator, torch.Generator):
        return seq.generator
    return None


def _seq_view(seq: Sequence) -> dict:
    return {
        "id": seq.request_id,
        "prompt": len(seq.prompt_ids),
        "generated": len(seq.generated_ids),
        "computed": seq.num_computed,
        "blocks": list(seq.block_ids),
        "status": seq.status,
        "finish_reason": seq.finish_reason,
    }
