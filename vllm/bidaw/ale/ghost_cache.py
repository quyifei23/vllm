# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Ghost cache running Belady optimal eviction policy in background.

Used to estimate hit_promising(i) for each weighted reuse distance bucket
under the optimal strategy on the historical I/O trace.
"""

import threading
import time
from collections import defaultdict, deque
from typing import NamedTuple


class AccessRecord(NamedTuple):
    user_id: str
    kv_size_bytes: int
    timestamp: float


class GhostCache:
    """
    Runs Belady optimal eviction on the historical access trace.

    Belady's algorithm: evict the block whose next access is farthest
    in the future. Since we run on the past portion of the trace,
    we can look ahead within that trace to find the true next access.

    Partitions weighted reuse distance into:
      - small bucket: WRD < performance_layer_capacity, hit_rate = 1.0
      - m promising buckets: WRD >= capacity but may hit
      - extreme bucket: hit_rate = 0.0
    """

    def __init__(
        self,
        perf_layer_size_bytes: int,
        num_promising_buckets: int = 20,
        history_size: int = 10000,
    ) -> None:
        self._perf_layer_size = perf_layer_size_bytes
        self._num_promising_buckets = num_promising_buckets
        self._history_size = history_size

        # Access trace buffer
        self._trace: deque[AccessRecord] = deque(maxlen=history_size)

        # Bucket boundaries: [0, perf_layer_size, ..., extreme]
        self._bucket_boundaries = self._compute_bucket_boundaries()

        # Statistics: per-bucket hit counts and access counts
        self._bucket_hits: dict[int, int] = defaultdict(int)
        self._bucket_accesses: dict[int, int] = defaultdict(int)

        # Background thread control
        self._running = False
        self._update_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Computed hit rates per bucket (updated by background thread)
        self._hit_rates: dict[int, float] = {}

    def _compute_bucket_boundaries(self) -> list[float]:
        """Compute the boundaries for each WRD bucket."""
        # Bucket 0: [0, perf_layer_size), Bucket 1+: evenly spaced above
        boundaries = [float(self._perf_layer_size)]
        max_wrd = self._perf_layer_size * 10
        step = (max_wrd - self._perf_layer_size) / max(1, self._num_promising_buckets)
        for i in range(1, self._num_promising_buckets + 1):
            boundaries.append(self._perf_layer_size + step * i)
        return boundaries

    def record_access(self, user_id: str, kv_size_bytes: int) -> int:
        """
        Record a KV access. Returns the bucket index for this access.

        The caller should compute the WRD and pass it separately;
        this is a simplified version that uses kv_size as proxy.
        """
        record = AccessRecord(
            user_id=user_id,
            kv_size_bytes=kv_size_bytes,
            timestamp=time.monotonic(),
        )
        self._trace.append(record)

        # Determine bucket: use kv_size as a proxy for WRD bucketing
        bucket = self._wrd_to_bucket(float(kv_size_bytes))
        return bucket

    def record_access_with_wrd(self, user_id: str, kv_size_bytes: int, wrd: float) -> int:
        """
        Record a KV access with the actual weighted reuse distance.
        Returns the bucket index.
        """
        record = AccessRecord(
            user_id=user_id,
            kv_size_bytes=kv_size_bytes,
            timestamp=time.monotonic(),
        )
        self._trace.append(record)

        bucket = self._wrd_to_bucket(wrd)
        return bucket

    def _wrd_to_bucket(self, wrd: float) -> int:
        """Map a weighted reuse distance to a bucket index."""
        for i, boundary in enumerate(self._bucket_boundaries):
            if wrd < boundary:
                return i
        return len(self._bucket_boundaries) - 1  # extreme bucket

    def get_hit_rate(self, bucket: int) -> float:
        """Get the computed hit rate for a bucket."""
        return self._hit_rates.get(bucket, 0.0)

    def get_all_hit_rates(self) -> dict[int, float]:
        """Get all computed hit rates."""
        return dict(self._hit_rates)

    def get_bucket_count(self) -> int:
        """Return the total number of buckets."""
        return len(self._bucket_boundaries)

    def get_num_promising_buckets(self) -> int:
        """Return the number of promising buckets."""
        return self._num_promising_buckets

    def get_perf_layer_size(self) -> int:
        """Return the performance layer size in bytes."""
        return self._perf_layer_size

    def run_belady_simulation(self) -> dict[int, float]:
        """
        Run Belady optimal eviction on the current trace.

        Returns a dict mapping bucket_index -> hit_rate.

        This is the core ghost cache simulation: for each access in the
        trace, simulate whether the data would have been in the cache
        under Belady's optimal policy, grouped by WRD bucket.
        """
        with self._lock:
            trace = list(self._trace)

        if not trace:
            return {}

        # Build next-access index: for each position, find next access
        # to the same user_id
        next_access: dict[int, int | None] = {}
        future_pos: dict[str, deque[int]] = defaultdict(deque)
        for i, rec in enumerate(trace):
            future_pos[rec.user_id].append(i)

        for i, rec in enumerate(trace):
            q = future_pos[rec.user_id]
            if q and q[0] == i:
                q.popleft()
            next_access[i] = q[0] if q else None

        # Simulate Belady
        cache: dict[str, int] = {}  # user_id -> next_access_pos
        cache_size = 0
        sim_bucket_hits: dict[int, int] = defaultdict(int)
        sim_bucket_accesses: dict[int, int] = defaultdict(int)

        for i, rec in enumerate(trace):
            # Use kv_size as proxy for WRD bucketing in the simulation
            bucket = self._wrd_to_bucket(float(rec.kv_size_bytes))
            sim_bucket_accesses[bucket] += 1

            if rec.user_id in cache:
                # Hit
                sim_bucket_hits[bucket] += 1
                # Update next access position
                cache[rec.user_id] = next_access.get(i, i + len(trace))
            else:
                # Miss: check if we need to evict
                while cache_size + rec.kv_size_bytes > self._perf_layer_size and cache:
                    # Belady: evict the user whose next access is farthest
                    victim = max(cache, key=lambda uid: cache[uid])
                    # Find victim's kv size from trace
                    victim_size = self._get_user_last_kv_size(cache, victim, trace)
                    cache_size -= victim_size
                    del cache[victim]

                cache[rec.user_id] = next_access.get(i, i + len(trace))
                cache_size += rec.kv_size_bytes

        # Compute hit rates
        hit_rates = {}
        for bucket in sim_bucket_accesses:
            if sim_bucket_accesses[bucket] > 0:
                hit_rates[bucket] = (
                    sim_bucket_hits[bucket] / sim_bucket_accesses[bucket]
                )
            else:
                hit_rates[bucket] = 0.0

        # Update internal hit rates
        with self._lock:
            self._hit_rates = hit_rates
            self._bucket_hits = dict(sim_bucket_hits)
            self._bucket_accesses = dict(sim_bucket_accesses)

        return hit_rates

    def _get_user_last_kv_size(
        self, cache: dict[str, int], user_id: str, trace: list[AccessRecord]
    ) -> int:
        """Get the last known kv_size for a user from the trace."""
        for i in range(len(trace) - 1, -1, -1):
            if trace[i].user_id == user_id:
                return trace[i].kv_size_bytes
        return 0

    def start_background_update(self, interval_seconds: float = 60.0) -> None:
        """
        Start a background thread that periodically re-runs the
        Belady simulation.
        """
        if self._running:
            return
        self._running = True

        def _update_loop() -> None:
            while self._running:
                self.run_belady_simulation()
                time.sleep(interval_seconds)

        self._update_thread = threading.Thread(target=_update_loop, daemon=True)
        self._update_thread.start()

    def stop_background_update(self) -> None:
        """Stop the background update thread."""
        self._running = False
        if self._update_thread is not None:
            self._update_thread.join(timeout=5.0)
            self._update_thread = None

    def reset(self) -> None:
        """Reset all state."""
        self._trace.clear()
        self._bucket_hits.clear()
        self._bucket_accesses.clear()
        self._hit_rates.clear()
