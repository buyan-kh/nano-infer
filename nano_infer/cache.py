"""Paged KV cache and the block allocator that backs it.

Physical memory is a fixed pool of blocks. A sequence never owns a contiguous
range; it owns a block table that maps logical blocks to physical ones.
Token `t` lives at physical block `table[t // block_size]`, slot `t % block_size`.
"""

from __future__ import annotations

import torch

from nano_infer.config import ModelConfig


class OutOfBlocks(RuntimeError):
    def __init__(self, requested: int, available: int) -> None:
        super().__init__(f"need {requested} KV blocks, only {available} free")
        self.requested = requested
        self.available = available


class CapacityError(ValueError):
    """Raised when a request cannot fit in the KV pool even running alone."""


def blocks_required(num_tokens: int, block_size: int) -> int:
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if num_tokens == 0:
        return 0
    return (num_tokens + block_size - 1) // block_size


class BlockAllocator:
    """Free-list allocator over a fixed set of block ids.

    Allocation pops from the free list. Freeing pushes back. A block cannot be
    freed twice. The watermark the scheduler keeps is a policy on top of this
    class; the allocator itself hands out every block it has.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))
        self._used: set[int] = set()

    def free_count(self) -> int:
        return len(self._free)

    def used_count(self) -> int:
        return len(self._used)

    def can_allocate(self, n: int) -> bool:
        return n <= len(self._free)

    def allocate(self, n: int) -> list[int]:
        if n < 0:
            raise ValueError("n must be non-negative")
        if n == 0:
            return []
        if n > len(self._free):
            raise OutOfBlocks(n, len(self._free))
        out: list[int] = []
        for _ in range(n):
            block = self._free.pop()
            self._used.add(block)
            out.append(block)
        return out

    def free(self, blocks: list[int]) -> None:
        for block in blocks:
            if block not in self._used:
                raise RuntimeError(f"double free of KV block {block}")
            self._used.remove(block)
            self._free.append(block)


class KVCache:
    """Layer-major paged key/value storage.

    Layout: `[n_layer, num_blocks, block_size, n_head, head_dim]`.
    Layer is the outer index so a forward pass can hand one layer's page
    table to attention without slicing the other layers.
    """

    def __init__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int,
    ) -> None:
        if k.shape != v.shape:
            raise ValueError("key and value cache shapes differ")
        if k.ndim != 5:
            raise ValueError("cache must be [layer, block, slot, head, dim]")
        self.k = k
        self.v = v
        self.block_size = block_size
        self.num_blocks = k.shape[1]
        self.n_layer = k.shape[0]

    @classmethod
    def allocate(
        cls,
        config: ModelConfig,
        num_blocks: int,
        block_size: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> KVCache:
        shape = (config.n_layer, num_blocks, block_size, config.n_head, config.head_dim)
        k = torch.zeros(shape, dtype=dtype, device=device)
        v = torch.zeros(shape, dtype=dtype, device=device)
        return cls(k, v, block_size)

    @property
    def device(self) -> torch.device:
        return self.k.device

    @property
    def dtype(self) -> torch.dtype:
        return self.k.dtype


def stack_block_tables(tables: list[list[int]], device: torch.device | str) -> torch.Tensor:
    """Pad per-sequence block tables into a `[batch, max_blocks]` long tensor."""
    if not tables:
        raise ValueError("empty block table list")
    width = max((len(table) for table in tables), default=0)
    width = max(width, 1)
    out = torch.zeros(len(tables), width, dtype=torch.long, device=device)
    for row, table in enumerate(tables):
        if table:
            out[row, : len(table)] = torch.tensor(table, dtype=torch.long, device=device)
    return out
