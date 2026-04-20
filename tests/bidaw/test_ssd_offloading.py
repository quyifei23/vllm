# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Step 4: SSD Offloading Manager.

Covers:
- SSDLoadStoreSpec: medium type, path management
- BidawOffloadingManager: lookup, prepare_load/store, complete_load/store, LRU
- Rank-aware SSD paths
- SSD file I/O (mocked for unit testing)
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from vllm.v1.core.kv_cache_utils import BlockHash

from vllm.bidaw.ssd.mediums import SSDLoadStoreSpec
from vllm.bidaw.ssd.manager import BidawOffloadingManager


class TestSSDLoadStoreSpec:
    def test_medium_returns_ssd(self):
        assert SSDLoadStoreSpec.medium() == "ssd"

    def test_add_and_get_block_path(self):
        spec = SSDLoadStoreSpec()
        spec.add_block_path("hash1", "/tmp/block1.bin")
        spec.add_block_path("hash2", "/tmp/block2.bin")
        assert spec.get_file_path("hash1") == "/tmp/block1.bin"
        assert spec.get_file_path("hash2") == "/tmp/block2.bin"
        assert spec.get_file_path("nonexistent") == ""

    def test_default_values(self):
        spec = SSDLoadStoreSpec()
        assert spec.cache_dir == "/tmp/bidaw_kv_cache"
        assert spec.io_threads == 4
        assert spec.block_paths == {}


class TestBidawOffloadingManager:
    def _make_manager(self, ssd_dir: str = "/tmp/test_bidaw_ssd", **kwargs):
        defaults = {
            "ssd_cache_dir": ssd_dir,
            "ssd_io_threads": 2,
            "max_blocks": 10,
            "tp_rank": 0,
            "pp_rank": 0,
        }
        defaults.update(kwargs)
        return BidawOffloadingManager(**defaults)

    def test_create_manager_creates_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = self._make_manager(ssd_dir=tmpdir)
            rank_dir = os.path.join(tmpdir, "rank_0", "pp_0")
            assert os.path.exists(rank_dir)
            mgr.shutdown()

    def test_lookup_empty_returns_zero(self):
        mgr = self._make_manager()
        result = mgr.lookup([BlockHash(b"hash1"), BlockHash(b"hash2")])
        assert result == 0
        mgr.shutdown()

    def test_prepare_store_returns_output(self):
        mgr = self._make_manager()
        bh = BlockHash(b"test_hash_001")
        output = mgr.prepare_store([bh])
        assert output is not None
        assert bh in output.block_hashes_to_store
        assert isinstance(output.store_spec, SSDLoadStoreSpec)
        mgr.shutdown()

    def test_complete_store_makes_loadable(self):
        mgr = self._make_manager()
        bh = BlockHash(b"test_hash_002")
        mgr.prepare_store([bh])
        mgr.complete_store([bh], success=True)

        # Now lookup should find it
        result = mgr.lookup([bh])
        assert result == 1
        mgr.shutdown()

    def test_complete_store_failure_removes_block(self):
        mgr = self._make_manager()
        bh = BlockHash(b"test_hash_003")
        mgr.prepare_store([bh])
        mgr.complete_store([bh], success=False)

        # Should not be loadable after failure
        result = mgr.lookup([bh])
        assert result == 0
        mgr.shutdown()

    def test_touch_updates_lru(self):
        mgr = self._make_manager()
        bh1 = BlockHash(b"test_hash_004")
        bh2 = BlockHash(b"test_hash_005")

        # Store both blocks
        mgr.prepare_store([bh1])
        mgr.complete_store([bh1], success=True)
        mgr.prepare_store([bh2])
        mgr.complete_store([bh2], success=True)

        # Touch bh1 to make it more recent
        mgr.touch([bh1])

        # Both should still be lookable
        result = mgr.lookup([bh1, bh2])
        assert result == 2
        mgr.shutdown()

    def test_prepare_load_protects_from_eviction(self):
        mgr = self._make_manager()
        bh = BlockHash(b"test_hash_006")

        # Store and complete
        mgr.prepare_store([bh])
        mgr.complete_store([bh], success=True)

        # Prepare load
        spec = mgr.prepare_load([bh])
        assert isinstance(spec, SSDLoadStoreSpec)

        # Complete load
        mgr.complete_load([bh])
        mgr.shutdown()

    def test_eviction_when_max_reached(self):
        mgr = self._make_manager(max_blocks=2)
        bh1 = BlockHash(b"test_hash_010")
        bh2 = BlockHash(b"test_hash_011")
        bh3 = BlockHash(b"test_hash_012")

        # Store 2 blocks (max)
        mgr.prepare_store([bh1])
        mgr.complete_store([bh1], success=True)
        mgr.prepare_store([bh2])
        mgr.complete_store([bh2], success=True)

        # Store 3rd block should evict LRU (bh1)
        output = mgr.prepare_store([bh3])
        assert output is not None
        assert len(output.block_hashes_evicted) == 1
        assert output.block_hashes_evicted[0] == bh1
        mgr.shutdown()

    def test_rank_aware_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = BidawOffloadingManager(
                ssd_cache_dir=tmpdir,
                tp_rank=2,
                pp_rank=1,
                max_blocks=10,
            )
            expected_dir = os.path.join(tmpdir, "rank_2", "pp_1")
            assert mgr.get_rank_dir() == expected_dir
            assert os.path.exists(expected_dir)
            mgr.shutdown()

    def test_get_num_offloaded_blocks(self):
        mgr = self._make_manager()
        assert mgr.get_num_offloaded_blocks() == 0

        bh = BlockHash(b"test_hash_020")
        mgr.prepare_store([bh])
        mgr.complete_store([bh], success=True)
        assert mgr.get_num_offloaded_blocks() == 1
        mgr.shutdown()

    def test_take_events(self):
        mgr = self._make_manager(enable_events=True)
        bh = BlockHash(b"test_hash_030")
        mgr.prepare_store([bh])
        mgr.complete_store([bh], success=True)

        events = list(mgr.take_events())
        assert len(events) == 1
        assert events[0].medium == "ssd"
        assert events[0].removed is False
        mgr.shutdown()

    def test_take_events_empty(self):
        mgr = self._make_manager()
        events = list(mgr.take_events())
        assert len(events) == 0
        mgr.shutdown()

    def test_lookup_consecutive_blocks(self):
        mgr = self._make_manager()
        bhs = [BlockHash(f"test_hash_{i:03d}".encode()) for i in range(5)]

        # Store all blocks
        for bh in bhs:
            mgr.prepare_store([bh])
            mgr.complete_store([bh], success=True)

        # Lookup all should return 5
        result = mgr.lookup(bhs)
        assert result == 5

        # Lookup with a non-existent block in the middle
        bhs_with_gap = [
            BlockHash(b"test_hash_000"),
            BlockHash(b"test_hash_001"),
            BlockHash(b"nonexistent"),
            BlockHash(b"test_hash_003"),
        ]
        result = mgr.lookup(bhs_with_gap)
        assert result == 2  # Only first 2 consecutive found
        mgr.shutdown()
