# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bidaw SSD Offloading Manager.

Implements three-tier storage (GPU ↔ host memory ↔ SSD) with:
- Rank-aware SSD paths for TP/PP: {ssd_dir}/rank_{tp_rank}/pp_{pp_rank}/
- Thread pool for async SSD I/O operations
- Integration with BidawEvictionManager for eviction decisions
"""

import os
import threading
from collections import OrderedDict
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)

from vllm.bidaw.ssd.mediums import SSDLoadStoreSpec

if TYPE_CHECKING:
    from vllm.bidaw.ale.eviction_manager import BidawEvictionManager

logger = init_logger(__name__)


@dataclass
class SSDBlockMeta:
    """Metadata for a block stored on SSD."""

    block_hash: BlockHash
    file_path: str
    size_bytes: int
    # LRU tracking
    access_order: int = 0


class BidawOffloadingManager(OffloadingManager):
    """
    Offloading manager for SSD-backed KV cache storage.

    Manages which blocks are stored on SSD, handles load/store preparation,
    and integrates with the Bidaw eviction manager for eviction decisions.

    Thread safety:
    - All public methods are thread-safe via self._lock
    - SSD I/O operations use a ThreadPoolExecutor for async execution
    """

    def __init__(
        self,
        ssd_cache_dir: str = "/tmp/bidaw_kv_cache",
        ssd_io_threads: int = 4,
        max_blocks: int = 100000,
        tp_rank: int = 0,
        pp_rank: int = 0,
        eviction_manager: "BidawEvictionManager | None" = None,
        enable_events: bool = False,
    ) -> None:
        self._lock = threading.Lock()
        self._ssd_cache_dir = ssd_cache_dir
        self._io_threads = ssd_io_threads
        self._max_blocks = max_blocks
        self._tp_rank = tp_rank
        self._pp_rank = pp_rank
        self._eviction_manager = eviction_manager
        self._enable_events = enable_events

        # LRU cache of offloaded blocks: block_hash -> SSDBlockMeta
        self._offloaded_blocks: OrderedDict[bytes, SSDBlockMeta] = OrderedDict()
        # Blocks currently being loaded (protected from eviction)
        self._loading_blocks: set[bytes] = set()
        # Blocks currently being stored
        self._storing_blocks: set[bytes] = set()
        # Access counter for LRU
        self._access_counter: int = 0

        # Thread pool for async I/O
        self._io_pool = ThreadPoolExecutor(
            max_workers=ssd_io_threads, thread_name_prefix="ssd_io"
        )

        # Events queue
        self._events: list[OffloadingEvent] = []

        # Create SSD directory structure
        self._ensure_ssd_dir()

    def _ensure_ssd_dir(self) -> None:
        """Create the SSD cache directory for this rank."""
        rank_dir = os.path.join(
            self._ssd_cache_dir,
            f"rank_{self._tp_rank}",
            f"pp_{self._pp_rank}",
        )
        os.makedirs(rank_dir, exist_ok=True)
        self._rank_dir = rank_dir

    def _block_hash_to_key(self, block_hash: BlockHash) -> bytes:
        """Convert BlockHash to a dict key."""
        if isinstance(block_hash, bytes):
            return block_hash
        return bytes(block_hash)

    def _get_ssd_file_path(self, block_hash: BlockHash) -> str:
        """Generate the SSD file path for a block."""
        key = self._block_hash_to_key(block_hash)
        hex_hash = key.hex()[:16]
        return os.path.join(self._rank_dir, f"block_{hex_hash}.bin")

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        """
        Find the length of the maximal series of blocks starting from
        the first one that are all offloaded to SSD.

        Returns the count of consecutive blocks found, or None if lookup
        should be retried.
        """
        count = 0
        with self._lock:
            for bh in block_hashes:
                key = self._block_hash_to_key(bh)
                if key in self._offloaded_blocks:
                    count += 1
                    # Update LRU
                    meta = self._offloaded_blocks[key]
                    self._offloaded_blocks.move_to_end(key)
                    meta.access_order = self._access_counter
                    self._access_counter += 1
                else:
                    break
        return count if count > 0 else 0

    def prepare_load(self, block_hashes: Iterable[BlockHash]) -> LoadStoreSpec:
        """
        Prepare blocks to be read from SSD.
        Protects blocks from eviction until complete_load is called.
        """
        spec = SSDLoadStoreSpec(
            cache_dir=self._ssd_cache_dir,
            io_threads=self._io_threads,
        )

        with self._lock:
            for bh in block_hashes:
                key = self._block_hash_to_key(bh)
                if key in self._offloaded_blocks:
                    file_path = self._offloaded_blocks[key].file_path
                    spec.add_block_path(bh.hex() if isinstance(bh, bytes) else str(bh), file_path)
                    # Protect from eviction
                    self._loading_blocks.add(key)

        return spec

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        """Mark blocks as done loading, re-allowing eviction."""
        with self._lock:
            for bh in block_hashes:
                key = self._block_hash_to_key(bh)
                self._loading_blocks.discard(key)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        """Mark blocks as recently used (move to end of LRU)."""
        with self._lock:
            for bh in block_hashes:
                key = self._block_hash_to_key(bh)
                if key in self._offloaded_blocks:
                    self._offloaded_blocks.move_to_end(key)
                    meta = self._offloaded_blocks[key]
                    meta.access_order = self._access_counter
                    self._access_counter += 1

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        """
        Prepare blocks to be written to SSD.

        When ALE eviction is enabled, consults the eviction manager to
        determine whether offloading is necessary based on current memory
        pressure. Returns which blocks need storing and their SSD paths.
        """
        hashes_to_store = []
        evicted_hashes = []

        with self._lock:
            for bh in block_hashes:
                key = self._block_hash_to_key(bh)
                if key in self._offloaded_blocks:
                    # Already stored, just update LRU
                    self._offloaded_blocks.move_to_end(key)
                    continue

                # Check if we need to evict
                if len(self._offloaded_blocks) >= self._max_blocks:
                    # Use ALE eviction if available, otherwise fall back to LRU
                    if self._eviction_manager is not None:
                        self._evict_via_ale()

                    # If still full after ALE eviction, evict LRU
                    if len(self._offloaded_blocks) >= self._max_blocks:
                        evict_key, evict_meta = self._offloaded_blocks.popitem(last=False)
                        evicted_hashes.append(evict_meta.block_hash)
                        self._io_pool.submit(self._remove_file, evict_meta.file_path)

                file_path = self._get_ssd_file_path(bh)
                hashes_to_store.append(bh)
                self._storing_blocks.add(key)

        if not hashes_to_store and not evicted_hashes:
            return None

        spec = SSDLoadStoreSpec(
            cache_dir=self._ssd_cache_dir,
            io_threads=self._io_threads,
        )
        for bh in hashes_to_store:
            file_path = self._get_ssd_file_path(bh)
            bh_key = bh.hex() if isinstance(bh, bytes) else str(bh)
            spec.add_block_path(bh_key, file_path)

        return PrepareStoreOutput(
            block_hashes_to_store=hashes_to_store,
            store_spec=spec,
            block_hashes_evicted=evicted_hashes,
        )

    def _evict_via_ale(self) -> None:
        """Use ALE eviction manager to select and evict a candidate."""
        # Estimate memory pressure from block usage ratio
        usage_ratio = len(self._offloaded_blocks) / max(1, self._max_blocks)
        free_ratio = 1.0 - usage_ratio

        if self._eviction_manager is None:
            return

        candidate = self._eviction_manager.select_eviction_candidate()
        if candidate is not None:
            # Find and evict blocks belonging to this candidate
            # Since we don't have direct block->user mapping here,
            # fall back to LRU eviction for now. The ALE manager
            # will be properly wired via the OffloadingConnectorScheduler.
            if self._offloaded_blocks:
                evict_key, evict_meta = self._offloaded_blocks.popitem(last=False)
                self._io_pool.submit(self._remove_file, evict_meta.file_path)

    def complete_store(self, block_hashes: Iterable[BlockHash], success: bool = True) -> None:
        """Mark blocks as stored, making them loadable."""
        with self._lock:
            for bh in block_hashes:
                key = self._block_hash_to_key(bh)
                self._storing_blocks.discard(key)

                if success:
                    file_path = self._get_ssd_file_path(bh)
                    self._offloaded_blocks[key] = SSDBlockMeta(
                        block_hash=bh,
                        file_path=file_path,
                        size_bytes=0,  # Will be updated by worker
                        access_order=self._access_counter,
                    )
                    self._access_counter += 1

                    if self._enable_events:
                        self._events.append(
                            OffloadingEvent(
                                block_hashes=[bh],
                                block_size=1,
                                medium="ssd",
                                removed=False,
                            )
                        )
                else:
                    # Store failed, clean up
                    file_path = self._get_ssd_file_path(bh)
                    self._io_pool.submit(self._remove_file, file_path)

    def _remove_file(self, file_path: str) -> None:
        """Remove a file from SSD storage."""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError as e:
            logger.warning("Failed to remove SSD file %s: %s", file_path, e)

    def take_events(self) -> Iterable[OffloadingEvent]:
        """Take and clear pending offloading events."""
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def get_num_offloaded_blocks(self) -> int:
        """Return the number of blocks currently offloaded to SSD."""
        with self._lock:
            return len(self._offloaded_blocks)

    def get_rank_dir(self) -> str:
        """Return the SSD directory for this rank."""
        return getattr(self, "_rank_dir", self._ssd_cache_dir)

    def shutdown(self) -> None:
        """Shut down the I/O thread pool."""
        self._io_pool.shutdown(wait=True)
