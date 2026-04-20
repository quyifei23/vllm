# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mixed-Granularity Block Allocator for Bidaw.

Paper §4: uses big blocks (256 tokens) for history/query tokens and small
blocks (16 tokens) for response tokens with unpredictable length.

Wraps the v1 BlockPool to provide big/small block allocation with
split/merge logic.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)


class BlockType(Enum):
    """Type of block in the mixed-granularity allocator."""

    BIG = auto()
    SMALL = auto()


@dataclass
class BlockAllocation:
    """Result of a block allocation."""

    blocks: list["KVCacheBlock"]
    block_type: BlockType
    total_tokens: int


class BidawBlockAllocator:
    """
    Wraps BlockPool to provide mixed-granularity block allocation.

    Big blocks (big_block_size tokens):
      - Used for history tokens (already processed) and query tokens (prompt)
      - These are predictable-length, so large blocks reduce fragmentation

    Small blocks (small_block_size tokens):
      - Used for response tokens (generated output)
      - Unpredictable length, so small blocks reduce waste

    Split/Merge logic:
      - When big blocks run low, small block pools can be merged
      - When small blocks accumulate, they can be split from big block pool
    """

    def __init__(
        self,
        block_pool: "BlockPool",
        big_block_size: int = 256,
        small_block_size: int = 16,
    ) -> None:
        self._pool = block_pool
        self._big_block_size = big_block_size
        self._small_block_size = small_block_size
        self._hash_block_size = block_pool.hash_block_size

        # Track which blocks are big vs small
        # block_id -> BlockType
        self._block_types: dict[int, BlockType] = {}

        # Counters for monitoring
        self._total_big_allocated = 0
        self._total_small_allocated = 0

    @property
    def pool(self) -> "BlockPool":
        """Access the underlying BlockPool."""
        return self._pool

    @property
    def big_block_size(self) -> int:
        return self._big_block_size

    @property
    def small_block_size(self) -> int:
        return self._small_block_size

    @property
    def num_free_blocks(self) -> int:
        """Return the number of free blocks in the pool."""
        return self._pool.free_block_queue.num_free_blocks

    def allocate_big_block(self, num_tokens_needed: int = 0) -> BlockAllocation | None:
        """
        Allocate a big block for history/query tokens.

        Args:
            num_tokens_needed: Number of tokens to allocate (for logging).

        Returns:
            BlockAllocation with big block(s), or None if no free blocks.
        """
        blocks = self._pool.free_block_queue.popleft_if_available()
        if not blocks:
            return None

        for blk in blocks:
            self._block_types[blk.block_id] = BlockType.BIG
        self._total_big_allocated += len(blocks)

        total_tokens = len(blocks) * self._big_block_size
        return BlockAllocation(
            blocks=blocks,
            block_type=BlockType.BIG,
            total_tokens=total_tokens,
        )

    def allocate_small_block(self) -> BlockAllocation | None:
        """
        Allocate a small block for response tokens.

        Returns:
            BlockAllocation with small block(s), or None if no free blocks.
        """
        blocks = self._pool.free_block_queue.popleft_if_available()
        if not blocks:
            return None

        for blk in blocks:
            self._block_types[blk.block_id] = BlockType.SMALL
        self._total_small_allocated += len(blocks)

        total_tokens = len(blocks) * self._small_block_size
        return BlockAllocation(
            blocks=blocks,
            block_type=BlockType.SMALL,
            total_tokens=total_tokens,
        )

    def allocate_for_tokens(
        self, num_tokens: int, is_response: bool = False
    ) -> BlockAllocation | None:
        """
        Smart allocation: choose big or small block based on token type.

        Args:
            num_tokens: Number of tokens to allocate space for.
            is_response: If True, allocate small blocks (response tokens).
                        If False, allocate big blocks (prompt/history).

        Returns:
            BlockAllocation with appropriate block type.
        """
        if is_response:
            return self.allocate_small_block()
        else:
            # For prompt/history, compute how many big blocks needed
            num_big_blocks = (
                num_tokens + self._big_block_size - 1
            ) // self._big_block_size
            blocks = []
            for _ in range(num_big_blocks):
                alloc = self.allocate_big_block(num_tokens)
                if alloc is None:
                    # Free already-allocated blocks if we can't fulfill request
                    for b in blocks:
                        self.free_block(b.block_id)
                    return None
                blocks.extend(alloc.blocks)

            total_tokens = len(blocks) * self._big_block_size
            return BlockAllocation(
                blocks=blocks,
                block_type=BlockType.BIG,
                total_tokens=total_tokens,
            )

    def free_block(self, block_id: int) -> None:
        """
        Free a block from type tracking.

        Note: The underlying BlockPool handles the actual free operation
        through its free_block_queue when the block's ref_cnt drops to 0.
        This method only removes the type tracking entry.

        Args:
            block_id: The ID of the block to free.
        """
        self._block_types.pop(block_id, None)

    def get_block_type(self, block_id: int) -> BlockType | None:
        """Get the type of a block by its ID."""
        return self._block_types.get(block_id)

    def get_stats(self) -> dict:
        """Return allocation statistics."""
        big_count = sum(
            1 for t in self._block_types.values() if t == BlockType.BIG
        )
        small_count = sum(
            1 for t in self._block_types.values() if t == BlockType.SMALL
        )
        return {
            "big_blocks_allocated": big_count,
            "small_blocks_allocated": small_count,
            "total_big_allocated": self._total_big_allocated,
            "total_small_allocated": self._total_small_allocated,
            "free_blocks": self.num_free_blocks,
            "big_block_size": self._big_block_size,
            "small_block_size": self._small_block_size,
        }

    def can_split_big_to_small(self) -> bool:
        """Check if we have enough big block capacity to split into small."""
        # We always can split conceptually - it's a logical division
        return self.num_free_blocks > 0

    def merge_small_to_big(self, num_small_blocks: int) -> int:
        """
        Conceptually merge small blocks back into big block capacity.

        In v1, blocks are uniform at the hardware level. The "merge" is
        logical: we simply reassign the block type from SMALL to BIG.

        Args:
            num_small_blocks: Number of small blocks to merge.

        Returns:
            Number of blocks actually merged.
        """
        merged = 0
        for block_id, btype in self._block_types.items():
            if btype == BlockType.SMALL and merged < num_small_blocks:
                self._block_types[block_id] = BlockType.BIG
                merged += 1
        return merged
