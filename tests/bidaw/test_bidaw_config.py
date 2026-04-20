# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for Step 1: BidawConfig dataclass and EngineArgs wiring."""

import dataclasses

import pytest


# ---------------------------------------------------------------------------
# BidawConfig tests -- zero heavy dependencies (pure dataclass)
# ---------------------------------------------------------------------------

class TestBidawConfigDefaults:
    """Verify BidawConfig field defaults match the plan."""

    def _make_config(self):
        from vllm.bidaw.config import BidawConfig
        return BidawConfig()

    def test_enable_bidaw_default(self):
        cfg = self._make_config()
        assert cfg.enable_bidaw is False

    def test_enable_io_aware_scheduling_default(self):
        cfg = self._make_config()
        assert cfg.enable_io_aware_scheduling is True

    def test_kv_size_unit_default(self):
        cfg = self._make_config()
        assert cfg.kv_size_unit == 1e8

    def test_skip_oversize_requests_default(self):
        cfg = self._make_config()
        assert cfg.skip_oversize_requests is True

    def test_enable_answer_length_eviction_default(self):
        cfg = self._make_config()
        assert cfg.enable_answer_length_eviction is True

    def test_num_promising_buckets_default(self):
        cfg = self._make_config()
        assert cfg.num_promising_buckets == 20

    def test_eviction_threshold_default(self):
        cfg = self._make_config()
        assert cfg.eviction_threshold == 0.05

    def test_ghost_cache_history_size_default(self):
        cfg = self._make_config()
        assert cfg.ghost_cache_history_size == 10000

    def test_enable_storage_efficient_tensor_default(self):
        cfg = self._make_config()
        assert cfg.enable_storage_efficient_tensor is True

    def test_is_mha_model_default(self):
        cfg = self._make_config()
        assert cfg.is_mha_model is True

    def test_transform_stream_priority_default(self):
        cfg = self._make_config()
        assert cfg.transform_stream_priority == -1

    def test_big_block_size_default(self):
        cfg = self._make_config()
        assert cfg.big_block_size == 256

    def test_small_block_size_default(self):
        cfg = self._make_config()
        assert cfg.small_block_size == 16

    def test_tensor_parallel_size_default(self):
        cfg = self._make_config()
        assert cfg.tensor_parallel_size == 1

    def test_pipeline_parallel_size_default(self):
        cfg = self._make_config()
        assert cfg.pipeline_parallel_size == 1

    def test_ssd_cache_dir_default(self):
        cfg = self._make_config()
        assert cfg.ssd_cache_dir == "/tmp/bidaw_kv_cache"

    def test_ssd_io_threads_default(self):
        cfg = self._make_config()
        assert cfg.ssd_io_threads == 4

    def test_enable_inclusive_caching_default(self):
        cfg = self._make_config()
        assert cfg.enable_inclusive_caching is True

    def test_tensor_cache_memory_gb_default(self):
        cfg = self._make_config()
        assert cfg.tensor_cache_memory_gb == 0.0

    def test_field_count(self):
        """Ensure BidawConfig has exactly 19 fields as per the plan."""
        cfg = self._make_config()
        field_names = [f.name for f in dataclasses.fields(cfg)]
        assert len(field_names) == 19


class TestBidawConfigCustomValues:
    """Verify BidawConfig accepts custom values correctly."""

    def test_custom_values(self):
        from vllm.bidaw.config import BidawConfig
        cfg = BidawConfig(
            enable_bidaw=True,
            enable_io_aware_scheduling=False,
            kv_size_unit=5e7,
            skip_oversize_requests=False,
            enable_answer_length_eviction=False,
            num_promising_buckets=50,
            eviction_threshold=0.1,
            ghost_cache_history_size=50000,
            enable_storage_efficient_tensor=False,
            is_mha_model=False,
            transform_stream_priority=0,
            big_block_size=512,
            small_block_size=32,
            tensor_parallel_size=2,
            pipeline_parallel_size=4,
            ssd_cache_dir="/mnt/ssd/bidaw",
            ssd_io_threads=8,
            enable_inclusive_caching=False,
            tensor_cache_memory_gb=4.0,
        )
        assert cfg.enable_bidaw is True
        assert cfg.enable_io_aware_scheduling is False
        assert cfg.kv_size_unit == 5e7
        assert cfg.skip_oversize_requests is False
        assert cfg.enable_answer_length_eviction is False
        assert cfg.num_promising_buckets == 50
        assert cfg.eviction_threshold == 0.1
        assert cfg.ghost_cache_history_size == 50000
        assert cfg.enable_storage_efficient_tensor is False
        assert cfg.is_mha_model is False
        assert cfg.transform_stream_priority == 0
        assert cfg.big_block_size == 512
        assert cfg.small_block_size == 32
        assert cfg.tensor_parallel_size == 2
        assert cfg.pipeline_parallel_size == 4
        assert cfg.ssd_cache_dir == "/mnt/ssd/bidaw"
        assert cfg.ssd_io_threads == 8
        assert cfg.enable_inclusive_caching is False
        assert cfg.tensor_cache_memory_gb == 4.0

    def test_partial_custom_values(self):
        """Only override a subset; rest should use defaults."""
        from vllm.bidaw.config import BidawConfig
        cfg = BidawConfig(enable_bidaw=True, ssd_cache_dir="/custom/path")
        assert cfg.enable_bidaw is True
        assert cfg.ssd_cache_dir == "/custom/path"
        # Defaults unchanged
        assert cfg.enable_io_aware_scheduling is True
        assert cfg.kv_size_unit == 1e8
        assert cfg.big_block_size == 256


class TestBidawConfigReExport:
    """Verify BidawConfig is properly exported from all expected locations."""

    def test_import_from_vllm_bidaw(self):
        from vllm.bidaw import BidawConfig
        assert BidawConfig is not None
        assert dataclasses.is_dataclass(BidawConfig)

    def test_import_from_vllm_bidaw_config(self):
        from vllm.bidaw.config import BidawConfig
        assert BidawConfig is not None

    def test_import_from_vllm_config(self):
        from vllm.config import BidawConfig
        assert BidawConfig is not None

    def test_import_from_vllm_config_bidaw(self):
        from vllm.config.bidaw import BidawConfig
        assert BidawConfig is not None

    def test_all_exports_are_identical(self):
        """All re-export paths should yield the same class object."""
        from vllm.bidaw import BidawConfig as BC_bidaw
        from vllm.bidaw.config import BidawConfig as BC_config
        from vllm.config import BidawConfig as BC_vllm_config
        from vllm.config.bidaw import BidawConfig as BC_vllm_bidaw

        assert BC_bidaw is BC_config
        assert BC_vllm_config is BC_config
        assert BC_vllm_bidaw is BC_config


class TestEngineArgsBidawFields:
    """Verify EngineArgs has all Bidaw-related fields."""

    def test_engine_args_has_bidaw_fields(self):
        from vllm.engine.arg_utils import EngineArgs
        ea = EngineArgs(model="fake-model")
        # All Bidaw fields should be accessible
        assert hasattr(ea, "enable_bidaw")
        assert hasattr(ea, "enable_io_aware_scheduling")
        assert hasattr(ea, "kv_size_unit")
        assert hasattr(ea, "skip_oversize_requests")
        assert hasattr(ea, "enable_answer_length_eviction")
        assert hasattr(ea, "num_promising_buckets")
        assert hasattr(ea, "eviction_threshold")
        assert hasattr(ea, "ghost_cache_history_size")
        assert hasattr(ea, "enable_storage_efficient_tensor")
        assert hasattr(ea, "is_mha_model")
        assert hasattr(ea, "transform_stream_priority")
        assert hasattr(ea, "big_block_size")
        assert hasattr(ea, "small_block_size")
        assert hasattr(ea, "ssd_cache_dir")
        assert hasattr(ea, "ssd_io_threads")
        assert hasattr(ea, "enable_inclusive_caching")
        assert hasattr(ea, "tensor_cache_memory_gb")

    def test_engine_args_bidaw_defaults_match_config(self):
        from vllm.bidaw.config import BidawConfig
        from vllm.engine.arg_utils import EngineArgs

        ea = EngineArgs(model="fake-model")
        bc = BidawConfig()

        assert ea.enable_bidaw == bc.enable_bidaw
        assert ea.enable_io_aware_scheduling == bc.enable_io_aware_scheduling
        assert ea.kv_size_unit == bc.kv_size_unit
        assert ea.skip_oversize_requests == bc.skip_oversize_requests
        assert ea.enable_answer_length_eviction == bc.enable_answer_length_eviction
        assert ea.num_promising_buckets == bc.num_promising_buckets
        assert ea.eviction_threshold == bc.eviction_threshold
        assert ea.ghost_cache_history_size == bc.ghost_cache_history_size
        assert ea.enable_storage_efficient_tensor == bc.enable_storage_efficient_tensor
        assert ea.is_mha_model == bc.is_mha_model
        assert ea.transform_stream_priority == bc.transform_stream_priority
        assert ea.big_block_size == bc.big_block_size
        assert ea.small_block_size == bc.small_block_size
        assert ea.ssd_cache_dir == bc.ssd_cache_dir
        assert ea.ssd_io_threads == bc.ssd_io_threads
        assert ea.enable_inclusive_caching == bc.enable_inclusive_caching
        assert ea.tensor_cache_memory_gb == bc.tensor_cache_memory_gb
