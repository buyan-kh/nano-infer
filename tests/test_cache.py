import pytest

from nano_infer.cache import BlockAllocator, CapacityError, OutOfBlocks, blocks_required


def test_blocks_required_rounds_up():
    assert blocks_required(0, 4) == 0
    assert blocks_required(1, 4) == 1
    assert blocks_required(4, 4) == 1
    assert blocks_required(5, 4) == 2


def test_allocator_reuses_freed_blocks():
    alloc = BlockAllocator(4)
    first = alloc.allocate(3)
    assert alloc.free_count() == 1
    alloc.free(first[:2])
    assert alloc.free_count() == 3
    again = alloc.allocate(2)
    assert set(again) == set(first[:2])
    assert alloc.used_count() == 3


def test_allocator_refuses_oversubscribe_and_double_free():
    alloc = BlockAllocator(2)
    held = alloc.allocate(2)
    with pytest.raises(OutOfBlocks):
        alloc.allocate(1)
    alloc.free(held)
    with pytest.raises(RuntimeError):
        alloc.free(held)


def test_capacity_error_is_a_value_error():
    assert issubclass(CapacityError, ValueError)
