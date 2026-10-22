"""Speculative decoding (Leviathan, Kalman, Matias 2023).

The draft model proposes `gamma` tokens. The target scores the whole proposal
in one forward. Greedy mode keeps the proposal while it matches the target
argmax, then emits the target argmax at the first mismatch (or a bonus token
if the whole proposal matched). Sampling mode uses the rejection test
`min(1, p/q)` and, on a reject, draws from `normalize(max(0, p - q))`.

Greedy speculative decoding is exactly target greedy, for any draft. Sampling
mode matches the target distribution; it does not reproduce a particular
target-only sample, because the draft consumes its own random stream.

Caches are paged and separate from the continuous-batching pool. A rejected
suffix is dropped by freeing the tail blocks and replaying the corrected token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from nano_infer.cache import BlockAllocator, KVCache, blocks_required, stack_block_tables
from nano_infer.model import GPT2LM, model_device
from nano_infer.sampling import (
    SamplingParams,
    rejection_distribution,
    sample,
    speculative_accept,
)
from nano_infer.scheduler import Sequence


@dataclass
class SpecStats:
    proposed: int = 0
    accepted_draft: int = 0
    rounds: int = 0
    emitted: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.proposed == 0:
            return 0.0
        return self.accepted_draft / self.proposed

    @property
    def mean_emitted_per_round(self) -> float:
        if self.rounds == 0:
            return 0.0
        return self.emitted / self.rounds


@dataclass
class SpecRun:
    seq: Sequence
    prompt_len: int
    target_blocks: list[int] = field(default_factory=list)
    draft_blocks: list[int] = field(default_factory=list)
    target_len: int = 0
    draft_len: int = 0
    target_logits: torch.Tensor | None = None
    draft_logits: torch.Tensor | None = None
    draft_generator: torch.Generator | None = None
    stats: SpecStats = field(default_factory=SpecStats)


class SpeculativeDecoder:
    def __init__(
        self,
        target: GPT2LM,
        draft: GPT2LM,
        *,
        block_size: int,
        num_blocks: int,
        gamma: int,
    ) -> None:
        if gamma < 1:
            raise ValueError("gamma must be >= 1")
        if target.config.vocab_size != draft.config.vocab_size:
            raise ValueError("draft and target vocabularies differ")
        self.target = target
        self.draft = draft
        self.block_size = block_size
        self.gamma = gamma
        self.device = model_device(target)
        self.target_cache = KVCache.allocate(target.config, num_blocks, block_size, device=self.device)
        self.draft_cache = KVCache.allocate(draft.config, num_blocks, block_size, device=self.device)
        self.target_alloc = BlockAllocator(num_blocks)
        self.draft_alloc = BlockAllocator(num_blocks)

    def start(self, seq: Sequence) -> SpecRun:
        run = SpecRun(seq=seq, prompt_len=len(seq.prompt_ids))
        if seq.sampling.seed is None:
            run.draft_generator = None
        else:
            run.draft_generator = torch.Generator(device="cpu")
            run.draft_generator.manual_seed(int(seq.sampling.seed) + 1_000_003)
        target_logits, run.target_blocks, run.target_len = self._forward(
            self.target,
            self.target_cache,
            self.target_alloc,
            run.target_blocks,
            0,
            seq.prompt_ids,
        )
        draft_logits, run.draft_blocks, run.draft_len = self._forward(
            self.draft,
            self.draft_cache,
            self.draft_alloc,
            run.draft_blocks,
            0,
            seq.prompt_ids,
        )
        run.target_logits = target_logits[-1]
        run.draft_logits = draft_logits[-1]
        return run

    @torch.inference_mode()
    def round(self, run: SpecRun) -> list[int]:
        """Run one speculation round and return the tokens to emit, in order."""
        remaining = run.seq.sampling.max_tokens - len(run.seq.generated_ids)
        if remaining <= 0:
            return []
        assert run.target_logits is not None and run.draft_logits is not None
        if remaining == 1:
            token = sample(run.target_logits, run.seq.sampling, _as_generator(run.seq.generator))
            self._advance_both(run, token)
            run.stats.rounds += 1
            run.stats.emitted += 1
            return [token]

        gamma = min(self.gamma, remaining - 1)
        proposals: list[int] = []
        draft_probs: list[torch.Tensor] = []
        draft_logits = run.draft_logits
        for _ in range(gamma):
            draft_probs.append(torch.softmax(draft_logits.float(), dim=-1))
            token = sample(draft_logits, run.seq.sampling, run.draft_generator)
            proposals.append(token)
            stepped, run.draft_blocks, run.draft_len = self._forward(
                self.draft,
                self.draft_cache,
                self.draft_alloc,
                run.draft_blocks,
                run.draft_len,
                [token],
            )
            draft_logits = stepped[-1]

        after, run.target_blocks, run.target_len = self._forward(
            self.target,
            self.target_cache,
            self.target_alloc,
            run.target_blocks,
            run.target_len,
            proposals,
        )
        target_rows = [run.target_logits] + [after[i] for i in range(gamma - 1)]
        bonus_logits = after[gamma - 1]
        origin_target = run.target_len - gamma
        origin_draft = run.draft_len - gamma

        emitted: list[int] = []
        matched = 0
        rejected = False
        took_bonus = False
        greedy = run.seq.sampling.temperature == 0
        stop_ids = set(run.seq.sampling.stop_token_ids)
        target_gen = _as_generator(run.seq.generator)

        for index, token in enumerate(proposals):
            row = target_rows[index]
            if greedy:
                chosen = int(torch.argmax(row).item())
                keep = chosen == token
            else:
                probs = torch.softmax(row.float(), dim=-1)
                keep = speculative_accept(probs, draft_probs[index], token, target_gen)
                if keep:
                    chosen = token
                else:
                    chosen = int(
                        torch.multinomial(
                            rejection_distribution(probs, draft_probs[index]),
                            num_samples=1,
                            generator=target_gen,
                        ).item()
                    )
            if keep:
                matched += 1
                emitted.append(chosen)
                if chosen in stop_ids:
                    break
            else:
                emitted.append(chosen)
                rejected = True
                break
        else:
            bonus = sample(bonus_logits, run.seq.sampling, target_gen)
            emitted.append(bonus)
            took_bonus = True

        run.stats.rounds += 1
        run.stats.proposed += gamma
        run.stats.accepted_draft += matched

        if took_bonus:
            # Cache holds the draft proposals. The bonus is not stored yet.
            self._advance_both(run, emitted[-1])
        elif rejected:
            # Keep the matched proposal prefix, then write the corrected token.
            self._truncate_target(run, origin_target + matched)
            self._truncate_draft(run, origin_draft + matched)
            self._advance_both(run, emitted[-1])
        else:
            # Stopped on an accepted draft token. Drop the unverified tail.
            self._truncate_target(run, origin_target + len(emitted))
            self._truncate_draft(run, origin_draft + len(emitted))

        run.stats.emitted += len(emitted)
        return emitted

    def finish(self, run: SpecRun) -> None:
        if run.target_blocks:
            self.target_alloc.free(list(run.target_blocks))
            run.target_blocks = []
        if run.draft_blocks:
            self.draft_alloc.free(list(run.draft_blocks))
            run.draft_blocks = []
        run.target_len = 0
        run.draft_len = 0

    def _advance_both(self, run: SpecRun, token: int) -> None:
        target_logits, run.target_blocks, run.target_len = self._forward(
            self.target,
            self.target_cache,
            self.target_alloc,
            run.target_blocks,
            run.target_len,
            [token],
        )
        draft_logits, run.draft_blocks, run.draft_len = self._forward(
            self.draft,
            self.draft_cache,
            self.draft_alloc,
            run.draft_blocks,
            run.draft_len,
            [token],
        )
        run.target_logits = target_logits[-1]
        run.draft_logits = draft_logits[-1]

    def _truncate_target(self, run: SpecRun, new_len: int) -> None:
        run.target_blocks, run.target_len = self._truncate(
            self.target_alloc, run.target_blocks, new_len
        )

    def _truncate_draft(self, run: SpecRun, new_len: int) -> None:
        run.draft_blocks, run.draft_len = self._truncate(self.draft_alloc, run.draft_blocks, new_len)

    def _truncate(self, alloc: BlockAllocator, blocks: list[int], new_len: int) -> tuple[list[int], int]:
        keep = blocks_required(new_len, self.block_size) if new_len > 0 else 0
        extra = blocks[keep:]
        if extra:
            alloc.free(extra)
        return blocks[:keep], new_len

    def _forward(
        self,
        model: GPT2LM,
        cache: KVCache,
        alloc: BlockAllocator,
        blocks: list[int],
        computed: int,
        token_ids: list[int],
    ) -> tuple[torch.Tensor, list[int], int]:
        target_len = computed + len(token_ids)
        need = blocks_required(target_len, self.block_size) - len(blocks)
        if need > 0:
            blocks = list(blocks) + alloc.allocate(need)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        write_starts = torch.tensor([computed], dtype=torch.long, device=self.device)
        context_lens = torch.tensor([target_len], dtype=torch.long, device=self.device)
        tables = stack_block_tables([blocks], self.device)
        logits = model(input_ids, cache, tables, context_lens, write_starts)
        return logits[0], blocks, target_len


def _as_generator(value: object) -> torch.Generator | None:
    if isinstance(value, torch.Generator):
        return value
    return None
