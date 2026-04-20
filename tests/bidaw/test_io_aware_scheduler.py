# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Step 5: I/O-Aware Scheduler.

Covers:
- RequestQueue: add/remove/peek/pop operations
- DiskHRRNScorer: score computation with different wait times and KV sizes
- BidawScheduler: dual queue management, promotion, score updates
- SSDLoadStatus: tracking
"""

import time
from unittest.mock import MagicMock

from vllm.bidaw.scheduler import (
    BidawScheduler,
    DiskHRRNScorer,
    RequestQueue,
    SSDLoadStatus,
)


def make_mock_request(request_id: str, num_prompt_tokens: int = 100, arrival_time: float | None = None) -> MagicMock:
    """Create a mock Request for testing."""
    req = MagicMock()
    req.request_id = request_id
    req.num_prompt_tokens = num_prompt_tokens
    req.num_tokens = num_prompt_tokens
    req.arrival_time = arrival_time if arrival_time is not None else time.monotonic() - 1.0
    req.status = MagicMock()
    return req


class TestRequestQueue:
    def test_add_and_len(self):
        q = RequestQueue()
        req = make_mock_request("req_1")
        q.add_request(req, score=1.0)
        assert len(q) == 1

    def test_pop_request(self):
        q = RequestQueue()
        req1 = make_mock_request("req_1")
        req2 = make_mock_request("req_2")
        q.add_request(req1, score=1.0)
        q.add_request(req2, score=2.0)

        # Higher score should be popped first
        popped = q.pop_request()
        assert popped.request_id == "req_2"
        assert len(q) == 1

    def test_peek_request(self):
        q = RequestQueue()
        req = make_mock_request("req_1")
        q.add_request(req, score=5.0)
        peeked = q.peek_request()
        assert peeked.request_id == "req_1"
        assert len(q) == 1  # peek doesn't remove

    def test_remove_request(self):
        q = RequestQueue()
        req = make_mock_request("req_1")
        q.add_request(req, score=1.0)
        removed = q.remove_request("req_1")
        assert removed is not None
        assert removed.request_id == "req_1"
        assert len(q) == 0

    def test_remove_nonexistent(self):
        q = RequestQueue()
        assert q.remove_request("nonexistent") is None

    def test_empty_queue(self):
        q = RequestQueue()
        assert len(q) == 0
        assert bool(q) is False
        assert q.peek_request() is None
        assert q.pop_request() is None

    def test_update_score(self):
        q = RequestQueue()
        req = make_mock_request("req_1")
        q.add_request(req, score=1.0)
        q.update_score("req_1", 10.0)
        assert q.get_request("req_1") is not None

    def test_iter(self):
        q = RequestQueue()
        req1 = make_mock_request("req_1")
        req2 = make_mock_request("req_2")
        q.add_request(req1, score=1.0)
        q.add_request(req2, score=2.0)
        reqs = list(q)
        assert len(reqs) == 2


class TestDiskHRRNScorer:
    def test_score_increases_with_wait_time(self):
        scorer = DiskHRRNScorer(kv_size_unit=1e8)
        req1 = make_mock_request("req_1", arrival_time=time.monotonic() - 1.0)
        req2 = make_mock_request("req_2", arrival_time=time.monotonic() - 10.0)

        score1 = scorer.compute_score(req1, kv_size=10000.0)
        score2 = scorer.compute_score(req2, kv_size=10000.0)
        assert score2 > score1

    def test_score_decreases_with_kv_size(self):
        scorer = DiskHRRNScorer(kv_size_unit=1e8)
        req1 = make_mock_request("req_1")
        req2 = make_mock_request("req_2")

        score1 = scorer.compute_score(req1, kv_size=1000.0)
        score2 = scorer.compute_score(req2, kv_size=100000.0)
        assert score1 > score2

    def test_score_with_zero_kv_size(self):
        scorer = DiskHRRNScorer(kv_size_unit=1e8)
        req = make_mock_request("req_1")
        score = scorer.compute_score(req, kv_size=0)
        assert score == float("inf")

    def test_estimate_kv_size(self):
        scorer = DiskHRRNScorer()
        req = make_mock_request("req_1", num_prompt_tokens=1000)
        kv_size = scorer._estimate_kv_size(req)
        # Should be roughly 1000 * 100 = 100000
        assert kv_size > 0
        assert kv_size == 1000 * 100.0


class TestSSDLoadStatus:
    def test_default_values(self):
        status = SSDLoadStatus(request_id="req_1")
        assert status.request_id == "req_1"
        assert status.load_start_time == 0.0
        assert status.kv_size_bytes == 0.0
        assert status.is_loading is False
        assert status.is_complete is False


class TestBidawScheduler:
    def _make_scheduler(self, **kwargs):
        return BidawScheduler(
            kv_size_unit=1e8,
            skip_oversize_requests=True,
            max_kv_size_bytes=1e9,
            **kwargs,
        )

    def test_add_request_to_ready_queue(self):
        sched = self._make_scheduler()
        req = make_mock_request("req_1")
        sched.add_request(req, needs_ssd_load=False)

        assert len(sched.get_ready_queue()) == 1
        assert len(sched.get_preparing_queue()) == 0

    def test_add_request_to_preparing_queue(self):
        sched = self._make_scheduler()
        req = make_mock_request("req_1")
        sched.add_request(req, needs_ssd_load=True)

        assert len(sched.get_ready_queue()) == 0
        assert len(sched.get_preparing_queue()) == 1

    def test_promote_from_preparing_to_ready(self):
        sched = self._make_scheduler()
        req = make_mock_request("req_1")
        sched.add_request(req, needs_ssd_load=True)

        sched.promote_from_preparing_queue("req_1")

        assert len(sched.get_ready_queue()) == 1
        assert len(sched.get_preparing_queue()) == 0

    def test_promote_nonexistent_request(self):
        sched = self._make_scheduler()
        sched.promote_from_preparing_queue("nonexistent")
        # Should not raise, queues remain empty
        assert len(sched.get_ready_queue()) == 0

    def test_get_next_request(self):
        sched = self._make_scheduler()
        req = make_mock_request("req_1")
        sched.add_request(req, needs_ssd_load=False)

        next_req = sched.get_next_request()
        assert next_req is not None
        assert next_req.request_id == "req_1"
        assert len(sched.get_ready_queue()) == 0  # popped

    def test_get_next_request_empty(self):
        sched = self._make_scheduler()
        assert sched.get_next_request() is None

    def test_start_ssd_load(self):
        sched = self._make_scheduler()
        req = make_mock_request("req_1")
        sched.add_request(req, needs_ssd_load=True)

        sched.start_ssd_load("req_1")
        status = sched.get_load_status("req_1")
        assert status is not None
        assert status.is_loading is True
        assert status.load_start_time > 0

    def test_should_skip_oversize_request(self):
        sched = BidawScheduler(
            skip_oversize_requests=True,
            max_kv_size_bytes=1e6,  # 1MB
        )
        # Large request (100000 tokens * 100 bytes = 10MB > 1MB max)
        req = make_mock_request("req_1", num_prompt_tokens=100000)
        assert sched.should_skip_request(req) is True

    def test_should_not_skip_small_request(self):
        sched = self._make_scheduler()
        req = make_mock_request("req_1", num_prompt_tokens=10)
        assert sched.should_skip_request(req) is False

    def test_should_skip_disabled(self):
        sched = BidawScheduler(skip_oversize_requests=False, max_kv_size_bytes=1)
        req = make_mock_request("req_1", num_prompt_tokens=100000)
        assert sched.should_skip_request(req) is False

    def test_remove_request(self):
        sched = self._make_scheduler()
        req = make_mock_request("req_1")
        sched.add_request(req, needs_ssd_load=False)

        sched.remove_request("req_1")
        assert len(sched.get_ready_queue()) == 0
        assert sched.get_disk_hrrn_score("req_1") == 0.0

    def test_update_scores(self):
        sched = self._make_scheduler()
        req1 = make_mock_request("req_1")
        req2 = make_mock_request("req_2")
        sched.add_request(req1, needs_ssd_load=False)
        sched.add_request(req2, needs_ssd_load=False)

        old_score1 = sched.get_disk_hrrn_score("req_1")
        # Wait a tiny bit to let time pass
        time.sleep(0.01)
        sched.update_scores()
        new_score1 = sched.get_disk_hrrn_score("req_1")
        # Score should have increased (wait time increased)
        assert new_score1 >= old_score1

    def test_load_status_tracks_complete(self):
        sched = self._make_scheduler()
        req = make_mock_request("req_1")
        sched.add_request(req, needs_ssd_load=True)

        status = sched.get_load_status("req_1")
        assert status is not None
        assert status.is_complete is False

        sched.promote_from_preparing_queue("req_1")
        status = sched.get_load_status("req_1")
        assert status is not None
        assert status.is_complete is True
        assert status.is_loading is False

    def test_get_disk_hrrn_score_nonexistent(self):
        sched = self._make_scheduler()
        assert sched.get_disk_hrrn_score("nonexistent") == 0.0
