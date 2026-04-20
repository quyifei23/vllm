# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bidaw Eviction Manager.

Implements the Answer-Length-based Eviction (ALE) policy from paper §3.3:
- §3.3.1: Predict weighted reuse distance lower bound from answer length
- §3.3.2: Ghost cache estimates hit potential per WRD bucket
- §3.3.3: Combined eviction decision using overall hit potential

Trigger: when free host memory falls below eviction_threshold (5% default).
"""

import logging
from collections import defaultdict

from vllm.bidaw.ale.answer_length_predictor import AnswerLengthReusePredictor
from vllm.bidaw.ale.ghost_cache import GhostCache
from vllm.bidaw.ale.reuse_tracker import WeightedReuseDistanceTracker
from vllm.bidaw.config import BidawConfig

logger = logging.getLogger(__name__)


class BidawEvictionManager:
    """
    Manages KV cache eviction using the ALE policy.

    Core interface:
    1. on_answer_generated(user_id, answer_length) - called after generation
    2. on_kv_accessed(user_id, kv_size_bytes) - called when KV loaded to GPU
    3. select_eviction_candidate() -> str | None - returns user_id to evict
    4. trigger_eviction_if_needed(free_memory_ratio, candidate_users)

    Paper §3.3 overall hit potential formula:
    Overall_potential = prob_small × 1.0
                      + prob_extreme × 0.0
                      + Σ prob_promising(i) × hit_promising(i)

    The answer-length-predicted lower bound constrains the distribution:
    buckets below the lower bound get probability 0, then renormalize.
    """

    def __init__(self, config: BidawConfig, perf_layer_size_bytes: int) -> None:
        self._config = config
        self._perf_layer_size = perf_layer_size_bytes
        self._eviction_threshold = config.eviction_threshold

        # Core components
        self._reuse_tracker = WeightedReuseDistanceTracker(
            history_size=config.ghost_cache_history_size,
        )
        self._predictor = AnswerLengthReusePredictor()
        self._ghost_cache = GhostCache(
            perf_layer_size_bytes=perf_layer_size_bytes,
            num_promising_buckets=config.num_promising_buckets,
            history_size=config.ghost_cache_history_size,
        )

        # Per-user: last answer length
        self._user_answer_lengths: dict[str, int] = {}
        # Per-user: current KV size in the performance layer
        self._user_kv_sizes: dict[str, int] = defaultdict(int)

        # Start ghost cache background update
        self._ghost_cache.start_background_update(interval_seconds=60.0)

    def on_answer_generated(self, user_id: str, answer_length: int) -> None:
        """
        Called when a request finishes generation (paper Fig.9 step 3).

        Updates the user's answer length record for future eviction decisions.
        """
        self._user_answer_lengths[user_id] = answer_length

    def on_kv_accessed(self, user_id: str, kv_size_bytes: int) -> None:
        """
        Called when KV is loaded into GPU memory.

        Updates the weighted reuse distance tracker and ghost cache.
        """
        wrd = self._reuse_tracker.on_kv_accessed(user_id, kv_size_bytes)
        self._user_kv_sizes[user_id] += kv_size_bytes

        if wrd is not None:
            # Record in ghost cache with actual WRD
            self._ghost_cache.record_access_with_wrd(
                user_id, kv_size_bytes, wrd
            )
            # Update predictor if we have a known answer length
            if user_id in self._user_answer_lengths:
                self._predictor.record_observation(
                    self._user_answer_lengths[user_id], wrd
                )

    def select_eviction_candidate(self, candidate_users: set[str] | None = None) -> str | None:
        """
        Select the user KV with the lowest overall hit potential for eviction.

        Args:
            candidate_users: Optional set of user_ids to consider.
                If None, all users are candidates.

        Returns:
            user_id of the eviction candidate, or None if no candidates.
        """
        if not candidate_users:
            candidate_users = set(self._user_kv_sizes.keys())
        if not candidate_users:
            return None

        hit_rates = self._ghost_cache.get_all_hit_rates()
        lowest_potential = float("inf")
        eviction_target: str | None = None

        for user_id in candidate_users:
            potential = self._compute_overall_hit_potential(user_id, hit_rates)
            if potential < lowest_potential:
                lowest_potential = potential
                eviction_target = user_id

        return eviction_target

    def _compute_overall_hit_potential(
        self, user_id: str, hit_rates: dict[int, float]
    ) -> float:
        """
        Compute overall hit potential for a user (paper formula 2).

        Overall_potential = prob_small × 1.0
                          + prob_extreme × 0.0
                          + Σ prob_promising(i) × hit_promising(i)

        The answer-length-predicted lower bound constrains: set
        probabilities below the lower bound to 0, then renormalize.
        """
        # Get predicted lower bound from answer length
        answer_length = self._user_answer_lengths.get(user_id, 0)
        wrd_lower_bound = self._predictor.predict_lower_bound(answer_length)

        # Get user's WRD distribution
        wrd_distribution = self._reuse_tracker.get_user_wrd_distribution(user_id)
        if not wrd_distribution:
            # No history data: assume low potential (good eviction candidate)
            return 0.0

        # Partition WRD values into buckets
        total = len(wrd_distribution)
        bucket_probs: dict[int, float] = defaultdict(float)
        for wrd in wrd_distribution:
            bucket = self._ghost_cache._wrd_to_bucket(wrd)
            bucket_probs[bucket] += 1.0

        # Apply lower bound constraint: zero out buckets below threshold
        lower_bucket = self._ghost_cache._wrd_to_bucket(wrd_lower_bound)
        for bucket in list(bucket_probs.keys()):
            if bucket < lower_bucket:
                bucket_probs[bucket] = 0.0

        # Renormalize
        total_prob = sum(bucket_probs.values())
        if total_prob > 0:
            for bucket in bucket_probs:
                bucket_probs[bucket] /= total_prob

        # Compute overall potential
        potential = 0.0
        for bucket, prob in bucket_probs.items():
            rate = hit_rates.get(bucket, 0.0)
            potential += prob * rate

        return potential

    def trigger_eviction_if_needed(
        self,
        free_memory_ratio: float,
        candidate_users: set[str] | None = None,
    ) -> str | None:
        """
        Trigger eviction if free memory ratio is below threshold.

        Args:
            free_memory_ratio: Current free memory / total memory.
            candidate_users: Optional candidate set for eviction.

        Returns:
            user_id of evicted user, or None if eviction not triggered.
        """
        if free_memory_ratio >= self._eviction_threshold:
            return None

        candidate = self.select_eviction_candidate(candidate_users)
        if candidate is not None:
            logger.info(
                "Eviction triggered: user %s (free_mem_ratio=%.3f, threshold=%.3f)",
                candidate,
                free_memory_ratio,
                self._eviction_threshold,
            )
            return candidate
        return None

    def remove_user(self, user_id: str) -> None:
        """Remove a user from all tracking structures."""
        self._user_answer_lengths.pop(user_id, None)
        self._user_kv_sizes.pop(user_id, None)

    def get_user_answer_length(self, user_id: str) -> int:
        """Get the last recorded answer length for a user."""
        return self._user_answer_lengths.get(user_id, 0)

    def shutdown(self) -> None:
        """Shut down the eviction manager and background threads."""
        self._ghost_cache.stop_background_update()

    @property
    def reuse_tracker(self) -> WeightedReuseDistanceTracker:
        return self._reuse_tracker

    @property
    def predictor(self) -> AnswerLengthReusePredictor:
        return self._predictor

    @property
    def ghost_cache(self) -> GhostCache:
        return self._ghost_cache
