# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Step 2: ALE Eviction Manager.

Covers:
- WeightedReuseDistanceTracker: access recording, WRD computation
- AnswerLengthReusePredictor: observation recording, prediction, interpolation
- GhostCache: bucketing, hit rate computation, Belady simulation
- BidawEvictionManager: on_answer_generated, on_kv_accessed, select_eviction_candidate
"""

import pytest

from vllm.bidaw.ale.reuse_tracker import WeightedReuseDistanceTracker
from vllm.bidaw.ale.answer_length_predictor import AnswerLengthReusePredictor
from vllm.bidaw.ale.ghost_cache import GhostCache
from vllm.bidaw.config import BidawConfig


class TestWeightedReuseDistanceTracker:
    def test_first_access_returns_none(self):
        tracker = WeightedReuseDistanceTracker()
        result = tracker.on_kv_accessed("user_a", 1000)
        assert result is None

    def test_second_access_returns_wrd(self):
        tracker = WeightedReuseDistanceTracker()
        # user_a accesses
        tracker.on_kv_accessed("user_a", 1000)
        # user_b accesses in between
        tracker.on_kv_accessed("user_b", 2000)
        # user_a accesses again
        wrd = tracker.on_kv_accessed("user_a", 1500)
        assert wrd is not None
        assert wrd == 2000  # only user_b's size counted

    def test_same_user_consecutive_no_wrd(self):
        tracker = WeightedReuseDistanceTracker()
        tracker.on_kv_accessed("user_a", 1000)
        tracker.on_kv_accessed("user_a", 1000)
        # Should be 0 since no other users accessed between
        wrd = tracker.on_kv_accessed("user_a", 1000)
        assert wrd == 0

    def test_get_user_wrd_distribution(self):
        tracker = WeightedReuseDistanceTracker()
        tracker.on_kv_accessed("user_a", 1000)
        tracker.on_kv_accessed("user_b", 2000)
        tracker.on_kv_accessed("user_a", 1500)  # wrd=2000
        dist = tracker.get_user_wrd_distribution("user_a")
        assert len(dist) == 1
        assert dist[0] == 2000

    def test_get_all_users(self):
        tracker = WeightedReuseDistanceTracker()
        tracker.on_kv_accessed("user_a", 1000)
        tracker.on_kv_accessed("user_b", 2000)
        users = tracker.get_all_users()
        assert users == {"user_a", "user_b"}

    def test_wrd_percentile(self):
        tracker = WeightedReuseDistanceTracker()
        tracker.on_kv_accessed("user_a", 1000)
        # Create multiple WRD values
        for i in range(10):
            tracker.on_kv_accessed(f"user_other_{i}", 100)
            tracker.on_kv_accessed("user_a", 500)
        p50 = tracker.get_wrd_percentile("user_a", 50)
        assert p50 >= 0


class TestAnswerLengthReusePredictor:
    def test_record_and_predict(self):
        pred = AnswerLengthReusePredictor()
        pred.record_observation(answer_length=10, wrd=5000)
        pred.record_observation(answer_length=10, wrd=3000)
        pred.record_observation(answer_length=10, wrd=7000)
        # Min WRD for length 10 should be 3000
        lb = pred.predict_lower_bound(10)
        assert lb == 3000

    def test_predict_unknown_length_returns_zero(self):
        pred = AnswerLengthReusePredictor()
        assert pred.predict_lower_bound(999) == 0.0

    def test_interpolation(self):
        pred = AnswerLengthReusePredictor()
        pred.record_observation(answer_length=10, wrd=1000)
        pred.record_observation(answer_length=20, wrd=2000)
        # For length 15, should interpolate to min(1000, 2000) = 1000
        lb = pred.predict_lower_bound(15)
        assert lb == 1000

    def test_extrapolation_below_known(self):
        pred = AnswerLengthReusePredictor()
        pred.record_observation(answer_length=10, wrd=1000)
        # Length 5 is below known range, should return value at 10
        lb = pred.predict_lower_bound(5)
        assert lb == 1000

    def test_extrapolation_above_known(self):
        pred = AnswerLengthReusePredictor()
        pred.record_observation(answer_length=10, wrd=1000)
        # Length 100 is above known range, should return value at 10
        lb = pred.predict_lower_bound(100)
        assert lb == 1000

    def test_reset(self):
        pred = AnswerLengthReusePredictor()
        pred.record_observation(answer_length=10, wrd=1000)
        pred.reset()
        assert pred.predict_lower_bound(10) == 0.0

    def test_observation_count(self):
        pred = AnswerLengthReusePredictor()
        pred.record_observation(answer_length=10, wrd=1000)
        pred.record_observation(answer_length=10, wrd=2000)
        pred.record_observation(answer_length=20, wrd=3000)
        assert pred.get_observation_count(10) == 2
        assert pred.get_observation_count(20) == 1
        assert pred.get_observation_count(99) == 0

    def test_get_known_lengths(self):
        pred = AnswerLengthReusePredictor()
        pred.record_observation(answer_length=20, wrd=1000)
        pred.record_observation(answer_length=10, wrd=2000)
        assert pred.get_known_lengths() == [10, 20]


class TestGhostCache:
    def test_bucket_count(self):
        gc = GhostCache(
            perf_layer_size_bytes=10000,
            num_promising_buckets=20,
        )
        # small + promising + extreme
        assert gc.get_bucket_count() == 1 + 20

    def test_wrd_to_bucket_small(self):
        gc = GhostCache(perf_layer_size_bytes=10000)
        bucket = gc._wrd_to_bucket(5000.0)
        assert bucket == 0  # small bucket

    def test_wrd_to_bucket_extreme(self):
        gc = GhostCache(perf_layer_size_bytes=10000)
        bucket = gc._wrd_to_bucket(999999.0)
        assert bucket == gc.get_bucket_count() - 1  # extreme bucket (last index)

    def test_record_access_with_wrd(self):
        gc = GhostCache(perf_layer_size_bytes=10000)
        bucket = gc.record_access_with_wrd("user_a", 5000, wrd=1000.0)
        assert bucket == 0

    def test_belady_simulation_empty(self):
        gc = GhostCache(perf_layer_size_bytes=10000)
        result = gc.run_belady_simulation()
        assert result == {}

    def test_belady_simulation_with_data(self):
        gc = GhostCache(perf_layer_size_bytes=10000)
        # Simulate a small access trace
        for _ in range(5):
            gc.record_access("user_a", 2000)
            gc.record_access("user_b", 2000)
            gc.record_access("user_a", 2000)
        result = gc.run_belady_simulation()
        assert isinstance(result, dict)
        for bucket, rate in result.items():
            assert 0.0 <= rate <= 1.0

    def test_reset(self):
        gc = GhostCache(perf_layer_size_bytes=10000)
        gc.record_access("user_a", 2000)
        gc.reset()
        assert list(gc._trace) == []
        assert gc.get_all_hit_rates() == {}

    def test_hit_rate_defaults_zero(self):
        gc = GhostCache(perf_layer_size_bytes=10000)
        assert gc.get_hit_rate(0) == 0.0
        assert gc.get_hit_rate(999) == 0.0


class TestBidawEvictionManager:
    def _make_manager(self, perf_layer_size: int = 100000):
        config = BidawConfig(
            eviction_threshold=0.05,
            num_promising_buckets=5,
            ghost_cache_history_size=100,
        )
        from vllm.bidaw.ale.eviction_manager import BidawEvictionManager
        manager = BidawEvictionManager(config, perf_layer_size)
        return manager

    def test_on_answer_generated(self):
        mgr = self._make_manager()
        mgr.on_answer_generated("user_a", 50)
        assert mgr.get_user_answer_length("user_a") == 50

    def test_on_kv_accessed(self):
        mgr = self._make_manager()
        mgr.on_kv_accessed("user_a", 5000)
        # Should update kv_sizes
        assert "user_a" in mgr._user_kv_sizes

    def test_select_eviction_candidate_empty(self):
        mgr = self._make_manager()
        result = mgr.select_eviction_candidate()
        assert result is None

    def test_select_eviction_candidate_with_users(self):
        mgr = self._make_manager()
        # Create some users with different access patterns
        mgr.on_answer_generated("user_a", 10)
        mgr.on_answer_generated("user_b", 100)
        # Add KV accesses
        mgr.on_kv_accessed("user_a", 5000)
        mgr.on_kv_accessed("user_b", 3000)
        candidate = mgr.select_eviction_candidate()
        assert candidate is not None
        assert candidate in ("user_a", "user_b")

    def test_trigger_eviction_above_threshold(self):
        mgr = self._make_manager()
        # Free memory ratio above threshold -> no eviction
        result = mgr.trigger_eviction_if_needed(free_memory_ratio=0.5)
        assert result is None

    def test_trigger_eviction_below_threshold(self):
        mgr = self._make_manager()
        # Add users first
        mgr.on_answer_generated("user_a", 10)
        mgr.on_kv_accessed("user_a", 5000)
        # Free memory ratio below threshold -> should trigger
        result = mgr.trigger_eviction_if_needed(free_memory_ratio=0.01)
        assert result == "user_a"

    def test_remove_user(self):
        mgr = self._make_manager()
        mgr.on_answer_generated("user_a", 50)
        mgr._user_kv_sizes["user_a"] = 5000
        mgr.remove_user("user_a")
        assert "user_a" not in mgr._user_answer_lengths
        assert "user_a" not in mgr._user_kv_sizes

    def test_shutdown(self):
        mgr = self._make_manager()
        mgr.shutdown()

    def test_eviction_candidate_constrained_by_answer_length(self):
        """Test that answer length lower bound constrains hit potential."""
        mgr = self._make_manager()
        # user_a has short answer, user_b has long answer
        mgr.on_answer_generated("user_a", 5)
        mgr.on_answer_generated("user_b", 100)
        # Add accesses with WRDs
        for _ in range(5):
            mgr.on_kv_accessed("user_a", 1000)
            mgr.on_kv_accessed("user_b", 1000)
        # Both should be valid candidates
        candidate = mgr.select_eviction_candidate({"user_a", "user_b"})
        assert candidate is not None
