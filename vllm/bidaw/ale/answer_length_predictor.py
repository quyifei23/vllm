# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Predicts weighted reuse distance lower bound from answer length.

Paper §3.3.1: maintains online statistics mapping answer_length ->
weighted_reuse_distance lower bound for each answer length value.
"""

import bisect
from collections import defaultdict


class AnswerLengthReusePredictor:
    """
    Maintains answer_length → weighted_reuse_distance lower bound statistics.

    For each answer_length value l, records the minimum observed weighted
    reuse distance. For new requests, uses the answer_length from the
    previous round to estimate a lower bound.

    Uses interpolation for unseen answer lengths (nearest known length).
    """

    def __init__(self) -> None:
        # answer_length -> list of observed weighted reuse distances
        self._observations: dict[int, list[float]] = defaultdict(list)
        # answer_length -> minimum observed WRD (lower bound)
        self._min_wrd: dict[int, float] = {}

    def record_observation(self, answer_length: int, wrd: float) -> None:
        """
        Record an observation: a request with `answer_length` had
        weighted reuse distance `wrd`.
        """
        if answer_length <= 0:
            return
        self._observations[answer_length].append(wrd)
        current_min = self._min_wrd.get(answer_length, float("inf"))
        if wrd < current_min:
            self._min_wrd[answer_length] = wrd

    def predict_lower_bound(self, answer_length: int) -> float:
        """
        Predict the weighted reuse distance lower bound for a request
        with the given answer_length.

        Returns:
            Estimated lower bound. 0.0 if no data available.
        """
        if answer_length in self._min_wrd:
            return self._min_wrd[answer_length]

        # Interpolate from nearest known lengths
        sorted_lengths = sorted(self._min_wrd.keys())
        if not sorted_lengths:
            return 0.0

        # Find insertion point
        idx = bisect.bisect_left(sorted_lengths, answer_length)

        if idx == 0:
            return self._min_wrd[sorted_lengths[0]]
        if idx == len(sorted_lengths):
            return self._min_wrd[sorted_lengths[-1]]

        # Linear interpolation between adjacent known lengths
        lower_len = sorted_lengths[idx - 1]
        upper_len = sorted_lengths[idx]
        lower_val = self._min_wrd[lower_len]
        upper_val = self._min_wrd[upper_len]

        # Use the minimum of the two as conservative lower bound
        return min(lower_val, upper_val)

    def get_observation_count(self, answer_length: int) -> int:
        """Return how many observations exist for a given answer length."""
        return len(self._observations.get(answer_length, []))

    def get_known_lengths(self) -> list[int]:
        """Return all known answer lengths, sorted."""
        return sorted(self._min_wrd.keys())

    def reset(self) -> None:
        """Reset all observations."""
        self._observations.clear()
        self._min_wrd.clear()
