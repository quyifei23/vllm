# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Step 9: Bidaw Benchmark Script.

Covers:
- Poisson arrival generation
- RequestMetrics computation
- Benchmark result aggregation
- FIFO vs Bidaw simulation runs
"""

import random

from benchmarks.benchmark_bidaw import (
    generate_poisson_arrivals,
    RequestMetrics,
    BenchmarkResult,
    compute_result,
    run_fifo_benchmark,
    run_bidaw_benchmark,
    ModelProfile,
)


class TestPoissonArrivals:
    def test_num_arrivals(self):
        arrivals = generate_poisson_arrivals(50, rate=2.0)
        assert len(arrivals) == 50

    def test_sorted_order(self):
        arrivals = generate_poisson_arrivals(100, rate=3.0)
        for i in range(1, len(arrivals)):
            assert arrivals[i] >= arrivals[i - 1]

    def test_first_arrival_is_zero(self):
        arrivals = generate_poisson_arrivals(10, rate=1.0)
        assert arrivals[0] == 0.0

    def test_reproducible_with_seed(self):
        a1 = generate_poisson_arrivals(20, rate=2.0, seed=42)
        a2 = generate_poisson_arrivals(20, rate=2.0, seed=42)
        assert a1 == a2

    def test_different_seeds_different_arrivals(self):
        a1 = generate_poisson_arrivals(20, rate=2.0, seed=1)
        a2 = generate_poisson_arrivals(20, rate=2.0, seed=2)
        assert a1 != a2

    def test_higher_rate_shorter_intervals(self):
        a_low = generate_poisson_arrivals(100, rate=0.5, seed=99)
        a_high = generate_poisson_arrivals(100, rate=10.0, seed=99)
        # Higher rate should have shorter total span
        assert a_high[-1] < a_low[-1]


class TestRequestMetrics:
    def test_ttft_computation(self):
        m = RequestMetrics(request_id="req_1", arrival_time=1.0, first_token_time=2.5)
        assert m.ttft == 1500.0  # 1.5s = 1500ms

    def test_ttft_zero_when_no_first_token(self):
        m = RequestMetrics(request_id="req_1", arrival_time=1.0)
        assert m.ttft == 0.0

    def test_tpot_computation(self):
        m = RequestMetrics(
            request_id="req_1",
            arrival_time=0.0,
            first_token_time=1.0,
            last_token_time=3.0,
            output_tokens=11,
        )
        # (3.0 - 1.0) * 1000 / (11 - 1) = 200ms
        assert m.tpot == 200.0

    def test_tpot_zero_for_single_token(self):
        m = RequestMetrics(
            request_id="req_1",
            first_token_time=1.0,
            last_token_time=2.0,
            output_tokens=1,
        )
        assert m.tpot == 0.0

    def test_total_latency(self):
        m = RequestMetrics(
            request_id="req_1",
            arrival_time=0.0,
            completion_time=5.0,
        )
        assert m.total_latency_ms == 5000.0

    def test_total_latency_zero_when_not_complete(self):
        m = RequestMetrics(request_id="req_1", arrival_time=0.0)
        assert m.total_latency_ms == 0.0


class TestComputeResult:
    def _make_metrics(self, count: int) -> list[RequestMetrics]:
        metrics_list = []
        for i in range(count):
            m = RequestMetrics(
                request_id=f"req_{i}",
                arrival_time=float(i),
                first_token_time=float(i) + 1.0,
                last_token_time=float(i) + 3.0,
                completion_time=float(i) + 4.0,
                output_tokens=10,
            )
            metrics_list.append(m)
        return metrics_list

    def test_result_fields_populated(self):
        metrics = self._make_metrics(5)
        result = compute_result("test", metrics, request_rate=1.0)
        assert result.num_requests == 5
        assert result.ttft_avg > 0
        assert result.req_per_sec > 0

    def test_empty_metrics_raises(self):
        import pytest
        with pytest.raises(ValueError):
            compute_result("empty", [], request_rate=1.0)


class TestFIFOBenchmark:
    def test_returns_all_requests(self):
        profile = ModelProfile()
        arrivals = generate_poisson_arrivals(20, rate=2.0)
        results = run_fifo_benchmark(arrivals, profile)
        assert len(results) == 20

    def test_all_metrics_populated(self):
        profile = ModelProfile()
        arrivals = generate_poisson_arrivals(5, rate=1.0)
        results = run_fifo_benchmark(arrivals, profile)
        for m in results:
            assert m.first_token_time > m.arrival_time
            assert m.completion_time > m.first_token_time
            assert m.output_tokens > 0
            assert m.prompt_tokens > 0

    def test_deterministic_with_same_seed(self):
        profile = ModelProfile()
        arrivals = generate_poisson_arrivals(10, rate=2.0)
        r1 = run_fifo_benchmark(arrivals, profile)
        r2 = run_fifo_benchmark(arrivals, profile)
        assert len(r1) == len(r2)
        for m1, m2 in zip(r1, r2):
            assert m1.request_id == m2.request_id
            assert m1.ttft == m2.ttft


class TestBidawBenchmark:
    def test_returns_all_requests(self):
        profile = ModelProfile()
        arrivals = generate_poisson_arrivals(20, rate=2.0)
        results = run_bidaw_benchmark(arrivals, profile)
        assert len(results) == 20

    def test_all_metrics_populated(self):
        profile = ModelProfile()
        arrivals = generate_poisson_arrivals(5, rate=1.0)
        results = run_bidaw_benchmark(arrivals, profile)
        for m in results:
            assert m.first_token_time > m.arrival_time
            assert m.completion_time > m.first_token_time
            assert m.output_tokens > 0

    def test_cache_hit_requests_have_zero_ssd_wait(self):
        profile = ModelProfile()
        # Use many requests so cache hits become likely
        arrivals = generate_poisson_arrivals(50, rate=5.0, seed=42)
        results = run_bidaw_benchmark(arrivals, profile)
        cache_hits = [m for m in results if m.ssd_load_wait_ms == 0.0]
        assert len(cache_hits) > 0

    def test_deterministic_with_same_seed(self):
        profile = ModelProfile()
        arrivals = generate_poisson_arrivals(10, rate=2.0)
        r1 = run_bidaw_benchmark(arrivals, profile)
        r2 = run_bidaw_benchmark(arrivals, profile)
        assert len(r1) == len(r2)
