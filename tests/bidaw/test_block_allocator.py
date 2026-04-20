# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Step 3: Mixed-Granularity Block Allocator.

Covers:
- Big/small block allocation
- Block type tracking
- Split/merge logic
- Allocation statistics
"""

import pytest
from unittest.mock import MagicMock, patch

from vllm.bidaw.block_allocator import (
    BidawBlockAllocator,
    BlockAllocation,
    BlockType,
)


def make_mock_block_pool(num_blocks: int, hash_block_size: int = 64):
    """Create a mock BlockPool for testing."""
    pool = MagicMock()
    pool.num_gpu_blocks = num_blocks
    pool.hash_block_size = hash_block_size

    # Create mock KVCacheBlock objects
    blocks = []
    for i in range(num_blocks):
        blk = MagicMock()
        blk.block_id = i
        blk.ref_cnt = 0
        blk.is_null = False
        blocks.append(blk)

    # Mock free_block_queue
    queue = MagicMock()
    queue.num_free_blocks = num_blocks

    def popleft_if_available(n: int = 1):
        if queue.num_free_blocks >= n:
            result = blocks[:n]
            del blocks[:n]
            queue.num_free_blocks -= n
            return result
        elif blocks:
            result = blocks[:]
            blocks.clear()
            queue.num_free_blocks = 0
            return result
        return []

    queue.popleft_if_available = popleft_if_available
    pool.free_block_queue = queue

    return pool


class TestBlockType:
    def test_block_type_enum(self):
        assert BlockType.BIG.name == "BIG"
        assert BlockType.SMALL.name == "SMALL"


class TestBidawBlockAllocator:
    def test_create_allocator(self):
        pool = make_mock_block_pool(num_blocks=100, hash_block_size=64)
        alloc = BidawBlockAllocator(
            pool,
            big_block_size=256,
            small_block_size=16,
        )
        assert alloc.big_block_size == 256
        assert alloc.small_block_size == 16
        assert alloc.num_free_blocks == 100

    def test_allocate_big_block(self):
        pool = make_mock_block_pool(num_blocks=10)
        alloc = BidawBlockAllocator(pool, big_block_size=256)

        result = alloc.allocate_big_block()
        assert result is not None
        assert result.block_type == BlockType.BIG
        assert result.total_tokens == 256
        assert len(result.blocks) == 1

    def test_allocate_small_block(self):
        pool = make_mock_block_pool(num_blocks=10)
        alloc = BidawBlockAllocator(pool, small_block_size=16)

        result = alloc.allocate_small_block()
        assert result is not None
        assert result.block_type == BlockType.SMALL
        assert result.total_tokens == 16

    def test_allocate_for_prompt_uses_big_blocks(self):
        pool = make_mock_block_pool(num_blocks=10)
        alloc = BidawBlockAllocator(pool, big_block_size=256, small_block_size=16)

        result = alloc.allocate_for_tokens(num_tokens=512, is_response=False)
        assert result is not None
        assert result.block_type == BlockType.BIG
        # 512 tokens / 256 per block = 2 blocks
        assert len(result.blocks) == 2

    def test_allocate_for_response_uses_small_blocks(self):
        pool = make_mock_block_pool(num_blocks=10)
        alloc = BidawBlockAllocator(pool, big_block_size=256, small_block_size=16)

        result = alloc.allocate_for_tokens(num_tokens=1, is_response=True)
        assert result is not None
        assert result.block_type == BlockType.SMALL

    def test_get_block_type(self):
        pool = make_mock_block_pool(num_blocks=10)
        alloc = BidawBlockAllocator(pool, big_block_size=256, small_block_size=16)

        big = alloc.allocate_big_block()
        assert alloc.get_block_type(big.blocks[0].block_id) == BlockType.BIG

        small = alloc.allocate_small_block()
        assert alloc.get_block_type(small.blocks[0].block_id) == BlockType.SMALL

    def test_get_stats(self):
        pool = make_mock_block_pool(num_blocks=10)
        alloc = BidawBlockAllocator(pool, big_block_size=256, small_block_size=16)

        alloc.allocate_big_block()
        alloc.allocate_big_block()
        alloc.allocate_small_block()

        stats = alloc.get_stats()
        assert stats["big_blocks_allocated"] == 2
        assert stats["small_blocks_allocated"] == 1
        assert stats["total_big_allocated"] == 2
        assert stats["total_small_allocated"] == 1
        assert stats["free_blocks"] == 7
        assert stats["big_block_size"] == 256
        assert stats["small_block_size"] == 16

    def test_merge_small_to_big(self):
        pool = make_mock_block_pool(num_blocks=10)
        alloc = BidawBlockAllocator(pool, big_block_size=256, small_block_size=16)

        # Allocate some small blocks
        alloc.allocate_small_block()
        alloc.allocate_small_block()
        alloc.allocate_small_block()

        # Merge them back to big
        merged = alloc.merge_small_to_big(num_small_blocks=2)
        assert merged == 2

        stats = alloc.get_stats()
        assert stats["big_blocks_allocated"] == 2
        assert stats["small_blocks_allocated"] == 1

    def test_can_split_big_to_small(self):
        pool = make_mock_block_pool(num_blocks=10)
        alloc = BidawBlockAllocator(pool, big_block_size=256, small_block_size=16)

        assert alloc.can_split_big_to_small() is True

    def test_allocate_when_pool_empty(self):
        pool = make_mock_block_pool(num_blocks=1)
        alloc = BidawBlockAllocator(pool, big_block_size=256, small_block_size=16)

        alloc.allocate_big_block()
        result = alloc.allocate_big_block()
        assert result is None

    def test_allocate_for_prompt_insufficient_blocks(self):
        pool = make_mock_block_pool(num_blocks=1)
        alloc = BidawBlockAllocator(pool, big_block_size=256, small_block_size=16)

        # Need 2 blocks but only 1 available
        result = alloc.allocate_for_tokens(num_tokens=512, is_response=False)
        # Should return None and free the partially allocated blocks
        assert result is None

    def test_free_block_removes_type(self):
        pool = make_mock_block_pool(num_blocks=10)
        alloc = BidawBlockAllocator(pool, big_block_size=256, small_block_size=16)

        big = alloc.allocate_big_block()
        block_id = big.blocks[0].block_id
        assert alloc.get_block_type(block_id) == BlockType.BIG

        alloc.free_block(block_id)
        assert alloc.get_block_type(block_id) is None
