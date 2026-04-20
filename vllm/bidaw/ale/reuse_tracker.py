# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tracks weighted reuse distance for each user's KV cache accesses.

Weighted reuse distance definition (paper §3.3.1):
"the aggregate size of other unique KVs accessed between
 successive accesses to the target KV"
"""

from collections import defaultdict, deque
from typing import NamedTuple


class ReuseDistanceEntry(NamedTuple):
    user_id: str
    reuse_distance: float
    kv_size_bytes: int


class WeightedReuseDistanceTracker:
    """
    Tracks per-user weighted reuse distance history distribution.

    Maintains a global KV access sequence and computes the weighted
    reuse distance for each access (aggregate size of other unique
    KVs accessed since last access to the same user's KV).
    """

    def __init__(self, history_size: int = 10000) -> None:
        self._history_size = history_size
        # Global access log: circular buffer of (user_id, kv_size_bytes)
        self._access_log: deque[tuple[str, int]] = deque(maxlen=history_size)
        # Per-user: weighted reuse distance history
        self._user_wrd_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        # Last seen position of each user in the access log for distance calc
        self._last_access_pos: dict[str, int] = {}
        # Global monotonic counter for access positions
        self._global_pos: int = 0

    def on_kv_accessed(self, user_id: str, kv_size_bytes: int) -> float | None:
        """
        Record a KV access for `user_id` with size `kv_size_bytes`.

        Returns the computed weighted reuse distance if this is a re-access
        to the same user, otherwise None.
        """
        current_pos = self._global_pos
        self._access_log.append((user_id, kv_size_bytes))

        wrd = None
        if user_id in self._last_access_pos:
            # Compute weighted reuse distance: aggregate size of OTHER
            # unique KVs accessed between last access and now
            wrd = self._compute_wrd(
                user_id, self._last_access_pos[user_id], current_pos
            )
            self._user_wrd_history[user_id].append(wrd)

        self._last_access_pos[user_id] = current_pos
        self._global_pos += 1
        return wrd

    def _compute_wrd(self, user_id: str, start_pos: int, end_pos: int) -> float:
        """
        Compute weighted reuse distance between start_pos and end_pos.

        Sum the kv_size_bytes of all unique users (excluding user_id)
        that were accessed in (start_pos, end_pos).
        """
        # Convert global positions to deque indices
        history_start = self._global_pos - len(self._access_log)
        seen_users: set[str] = set()
        total_size = 0
        for pos in range(start_pos + 1, end_pos):
            if pos < history_start:
                # Position has been evicted from the circular buffer
                continue
            log_idx = pos - history_start
            access_user, access_size = self._access_log[log_idx]
            if access_user != user_id and access_user not in seen_users:
                seen_users.add(access_user)
                total_size += access_size
        return float(total_size)

    def get_user_wrd_distribution(self, user_id: str) -> list[float]:
        """Return the weighted reuse distance history for a user."""
        return list(self._user_wrd_history[user_id])

    def get_all_users(self) -> set[str]:
        """Return all users that have been tracked."""
        return set(self._last_access_pos.keys())

    def get_wrd_percentile(self, user_id: str, percentile: float) -> float:
        """Return the given percentile of weighted reuse distance for a user."""
        history = list(self._user_wrd_history[user_id])
        if not history:
            return 0.0
        history.sort()
        idx = int(len(history) * percentile / 100.0)
        idx = min(idx, len(history) - 1)
        return history[idx]
