# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bidaw I/O-Aware Scheduler.

Paper §3.2: dual-queue scheduling with disk-HRRN scoring.

Core mechanisms:
- ready_queue: requests whose KV data is already loaded (ready to run)
- preparing_queue: requests waiting for SSD→GPU KV data loading
- disk-HRRN: scheduling priority = 1 + wait_time / (kv_size / kv_size_unit)

The scheduler promotes requests from preparing_queue to ready_queue
when their SSD load completes, and schedules from ready_queue using
disk-HRRN to balance fairness and I/O efficiency.
"""

import heapq
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class RequestQueue:
    """
    A priority queue for Bidaw scheduling using a min-heap.

    Requests are ordered by disk-HRRN score (higher = more urgent).
    We negate scores to use Python's min-heap as a max-heap.
    """

    def __init__(self) -> None:
        # Heap of (-score, counter, request) tuples; counter avoids request comparison
        self._heap: list[tuple[float, int, "Request"]] = []
        self._counter: int = 0
        # Track valid entries for lazy removal
        self._valid_ids: set[str] = set()

    def add_request(self, request: "Request", score: float = 0.0) -> None:
        """Add a request with its HRRN score."""
        self._counter += 1
        heapq.heappush(self._heap, (-score, self._counter, request))
        self._valid_ids.add(request.request_id)

    def remove_request(self, request_id: str) -> "Request | None":
        """Lazily remove and return a request by ID."""
        if request_id not in self._valid_ids:
            return None
        self._valid_ids.discard(request_id)
        for _, _, req in self._heap:
            if req.request_id == request_id:
                return req
        return None

    def peek_request(self) -> "Request | None":
        """Return the highest-priority request without removing it."""
        self._clean_heap()
        if not self._heap:
            return None
        return self._heap[0][2]

    def pop_request(self) -> "Request | None":
        """Remove and return the highest-priority request."""
        self._clean_heap()
        if not self._heap:
            return None
        _, _, req = heapq.heappop(self._heap)
        self._valid_ids.discard(req.request_id)
        return req

    def _clean_heap(self) -> None:
        """Remove invalid entries from the top of the heap."""
        while self._heap and self._heap[0][2].request_id not in self._valid_ids:
            heapq.heappop(self._heap)

    def get_request(self, request_id: str) -> "Request | None":
        """Get a request by ID without removing it."""
        for _, _, req in self._heap:
            if req.request_id == request_id and req.request_id in self._valid_ids:
                return req
        return None

    def __len__(self) -> int:
        return len(self._valid_ids)

    def __bool__(self) -> bool:
        return len(self._valid_ids) > 0

    def __iter__(self):
        for _, _, req in self._heap:
            if req.request_id in self._valid_ids:
                yield req

    def update_score(self, request_id: str, score: float) -> None:
        """Update the HRRN score for a request (lazy removal + re-add)."""
        self._valid_ids.discard(request_id)
        for _, _, req in self._heap:
            if req.request_id == request_id:
                self._counter += 1
                heapq.heappush(self._heap, (-score, self._counter, req))
                self._valid_ids.add(request_id)
                break


class DiskHRRNScorer:
    """
    Computes disk-HRRN scores for requests.

    Formula (paper §3.2):
        ratio = 1 + wait_time / (kv_size / kv_size_unit)

    where:
        - wait_time: seconds since request arrival
        - kv_size: estimated KV cache size for the request
        - kv_size_unit: normalization constant from config
    """

    def __init__(
        self, kv_size_unit: float = 1e8, is_mha_model: bool = True
    ) -> None:
        self._kv_size_unit = kv_size_unit
        self._is_mha_model = is_mha_model

    def compute_score(self, request: "Request", kv_size: float | None = None) -> float:
        """
        Compute the disk-HRRN score for a request.

        Args:
            request: The request to score.
            kv_size: Pre-computed KV size in bytes. If None, estimated from tokens.

        Returns:
            HRRN score (higher = more urgent to schedule).
        """
        wait_time = time.monotonic() - request.arrival_time

        if kv_size is None:
            # Estimate KV size from token count
            kv_size = self._estimate_kv_size(request)

        if kv_size <= 0:
            return float("inf")  # Very urgent if no data to load

        ratio = 1.0 + wait_time / (kv_size / self._kv_size_unit)
        return ratio

    def _estimate_kv_size(self, request: "Request") -> float:
        """
        Estimate the KV cache size for a request.

        For MHA models with tensor 6 caching: ~50 bytes/token
        (hidden_size * dtype_bytes, half of full KV).
        For GQA models: ~100 bytes/token (full KV).
        """
        num_tokens = request.num_prompt_tokens
        # MHA with tensor 6: half the size of full KV cache
        bytes_per_token = 50.0 if self._is_mha_model else 100.0
        return num_tokens * bytes_per_token


@dataclass
class SSDLoadStatus:
    """Tracks the SSD load status for a request."""

    request_id: str
    # Time when the load was initiated
    load_start_time: float = 0.0
    # Estimated KV size in bytes
    kv_size_bytes: float = 0.0
    # Whether the load is in progress
    is_loading: bool = False
    # Whether the load is complete
    is_complete: bool = False


class BidawScheduler:
    """
    I/O-aware scheduler with dual queues and disk-HRRN scoring.

    This is a standalone scheduler that wraps the v1 Scheduler's behavior
    to add Bidaw's I/O awareness:

    1. Dual queues:
       - ready_queue: requests with all KV data in GPU memory
       - preparing_queue: requests waiting for SSD→GPU KV data loading

    2. disk-HRRN scoring:
       - Priority = 1 + wait_time / (kv_size / kv_size_unit)
       - Balances fairness (wait time) with I/O efficiency (KV size)

    3. Promotion logic:
       - When SSD load completes, request moves from preparing → ready
       - Promotion order respects HRRN score
    """

    def __init__(
        self,
        kv_size_unit: float = 1e8,
        skip_oversize_requests: bool = True,
        max_kv_size_bytes: float = 0,
        is_mha_model: bool = True,
    ) -> None:
        self._ready_queue = RequestQueue()
        self._preparing_queue = RequestQueue()
        self._scorer = DiskHRRNScorer(
            kv_size_unit=kv_size_unit, is_mha_model=is_mha_model
        )
        self._skip_oversize = skip_oversize_requests
        self._max_kv_size_bytes = max_kv_size_bytes

        # Track SSD load status per request
        self._load_status: dict[str, SSDLoadStatus] = {}

        # HRRN scores for logging
        self._disk_hrrn_scores: dict[str, float] = {}

    def add_request(self, request: "Request", needs_ssd_load: bool = False) -> None:
        """
        Add a request to the appropriate queue.

        Args:
            request: The request to add.
            needs_ssd_load: If True, request goes to preparing_queue.
                           If False, request goes to ready_queue.
        """
        score = self._scorer.compute_score(request)
        self._disk_hrrn_scores[request.request_id] = score

        if needs_ssd_load:
            self._preparing_queue.add_request(request, score)
            self._load_status[request.request_id] = SSDLoadStatus(
                request_id=request.request_id,
                kv_size_bytes=self._estimate_request_kv_size(request),
            )
        else:
            self._ready_queue.add_request(request, score)

    def promote_from_preparing_queue(self, request_id: str) -> None:
        """
        Promote a request from preparing_queue to ready_queue.

        Called when SSD load completes for a request.

        Args:
            request_id: The ID of the request to promote.
        """
        request = self._preparing_queue.remove_request(request_id)
        if request is not None:
            # Update score since wait time has changed
            score = self._scorer.compute_score(request)
            self._ready_queue.add_request(request, score)
            self._disk_hrrn_scores[request_id] = score

            if request_id in self._load_status:
                self._load_status[request_id].is_complete = True
                self._load_status[request_id].is_loading = False

    def get_next_request(self) -> "Request | None":
        """
        Get the next request to schedule from ready_queue.

        Returns the highest-priority request (by disk-HRRN score).

        Returns:
            The next request to schedule, or None if no requests ready.
        """
        return self._ready_queue.pop_request()

    def start_ssd_load(self, request_id: str) -> None:
        """
        Mark that an SSD load has started for a request.

        Args:
            request_id: The ID of the request loading from SSD.
        """
        if request_id in self._load_status:
            self._load_status[request_id].is_loading = True
            self._load_status[request_id].load_start_time = time.monotonic()

    def _estimate_request_kv_size(self, request: "Request") -> float:
        """Estimate the KV cache size for a request in bytes (uses scorer's estimate)."""
        return self._scorer._estimate_kv_size(request)

    def _estimate_full_kv_size(self, request: "Request") -> float:
        """Estimate full KV size regardless of tensor 6 caching (for oversize check)."""
        num_tokens = request.num_prompt_tokens
        return num_tokens * 100.0  # full KV: ~100 bytes/token

    def should_skip_request(self, request: "Request") -> bool:
        """
        Check if a request should be skipped due to oversize.

        Args:
            request: The request to check.

        Returns:
            True if the request exceeds the max KV size.
        """
        if not self._skip_oversize or self._max_kv_size_bytes <= 0:
            return False

        kv_size = self._estimate_full_kv_size(request)
        return kv_size > self._max_kv_size_bytes

    def get_ready_queue(self) -> RequestQueue:
        """Return the ready queue."""
        return self._ready_queue

    def get_preparing_queue(self) -> RequestQueue:
        """Return the preparing queue."""
        return self._preparing_queue

    def get_disk_hrrn_score(self, request_id: str) -> float:
        """Get the disk-HRRN score for a request."""
        return self._disk_hrrn_scores.get(request_id, 0.0)

    def get_load_status(self, request_id: str) -> SSDLoadStatus | None:
        """Get the SSD load status for a request."""
        return self._load_status.get(request_id)

    def remove_request(self, request_id: str) -> None:
        """Remove a request from all queues and tracking."""
        self._ready_queue.remove_request(request_id)
        self._preparing_queue.remove_request(request_id)
        self._load_status.pop(request_id, None)
        self._disk_hrrn_scores.pop(request_id, None)

    def update_scores(self) -> None:
        """Recompute HRRN scores for all requests."""
        # Snapshot requests first to avoid heap modification during iteration
        for req in list(self._ready_queue):
            score = self._scorer.compute_score(req)
            self._ready_queue.update_score(req.request_id, score)
            self._disk_hrrn_scores[req.request_id] = score

        for req in list(self._preparing_queue):
            score = self._scorer.compute_score(req)
            self._preparing_queue.update_score(req.request_id, score)
            self._disk_hrrn_scores[req.request_id] = score
