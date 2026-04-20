# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bidaw Benchmark — Poisson arrival simulation with TTFT/TBT/throughput metrics.

Benchmarks the Bidaw I/O-aware scheduler against a FIFO baseline using
simulated request arrivals. Measures:
- TTFT (Time To First Token): latency from request arrival to first output token
- TBT (Time Between Tokens): average inter-token latency
- Throughput: requests/sec and tokens/sec
- SSD I/O wait time distribution

Usage:
    python benchmarks/benchmark_bidaw.py --num-requests 100 --req-rate 2.0
    python benchmarks/benchmark_bidaw.py --num-requests 200 --req-rate 5.0 --enable-bidaw
    python benchmarks/benchmark_bidaw.py --compare --num-requests 100
"""

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Metric Recording ──────────────────────────────────────────────────

@dataclass
class RequestMetrics:
    """Per-request timing metrics."""
    request_id: str
    arrival_time: float = 0.0
    first_token_time: float = 0.0
    last_token_time: float = 0.0
    completion_time: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    ssd_load_wait_ms: float = 0.0
    token_timestamps: list[float] = field(default_factory=list)

    @property
    def ttft(self) -> float:
        """Time to first token in milliseconds."""
        if self.first_token_time > 0:
            return (self.first_token_time - self.arrival_time) * 1000
        return 0.0

    @property
    def tpot(self) -> float:
        """Time per output token (TBT) in milliseconds."""
        if self.output_tokens > 1 and self.last_token_time > self.first_token_time:
            return (self.last_token_time - self.first_token_time) * 1000 / (self.output_tokens - 1)
        return 0.0

    @property
    def total_latency_ms(self) -> float:
        return (self.completion_time - self.arrival_time) * 1000 if self.completion_time > 0 else 0.0


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    name: str
    num_requests: int
    request_rate: float
    total_time_s: float
    # Throughput
    req_per_sec: float
    tok_per_sec: float
    # Latency percentiles (ms)
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float
    ttft_avg: float
    # TBT percentiles (ms)
    tbt_p50: float
    tbt_p90: float
    tbt_avg: float
    # Total latency percentiles (ms)
    total_p50: float
    total_p90: float
    total_avg: float
    # SSD metrics
    avg_ssd_wait_ms: float

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  Benchmark: {self.name}",
            f"{'='*60}",
            f"  Requests:        {self.num_requests}  @  {self.request_rate:.1f} req/s",
            f"  Total time:      {self.total_time_s:.2f}s",
            f"",
            f"  Throughput:",
            f"    Requests/sec:  {self.req_per_sec:.2f}",
            f"    Tokens/sec:    {self.tok_per_sec:.2f}",
            f"",
            f"  TTFT (ms):",
            f"    p50: {self.ttft_p50:.1f}   p90: {self.ttft_p90:.1f}   p99: {self.ttft_p99:.1f}   avg: {self.ttft_avg:.1f}",
            f"",
            f"  TBT (ms):",
            f"    p50: {self.tbt_p50:.1f}   p90: {self.tbt_p90:.1f}   avg: {self.tbt_avg:.1f}",
            f"",
            f"  Total Latency (ms):",
            f"    p50: {self.total_p50:.1f}   p90: {self.total_p90:.1f}   avg: {self.total_avg:.1f}",
            f"",
            f"  Avg SSD Wait:    {self.avg_ssd_wait_ms:.1f}ms",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# ── Poisson Arrival Generator ─────────────────────────────────────────

def generate_poisson_arrivals(
    num_requests: int,
    rate: float,
    seed: int = 42,
) -> list[float]:
    """
    Generate arrival timestamps for `num_requests` using a Poisson process.

    Inter-arrival times are exponentially distributed with mean 1/rate.

    Args:
        num_requests: Number of requests to generate.
        rate: Average requests per second (lambda).
        seed: Random seed for reproducibility.

    Returns:
        Sorted list of absolute arrival timestamps (seconds).
    """
    rng = random.Random(seed)
    arrivals = [0.0]
    for _ in range(num_requests - 1):
        interval = rng.expovariate(rate)
        arrivals.append(arrivals[-1] + interval)
    return arrivals


# ── Simulated Model ───────────────────────────────────────────────────

@dataclass
class ModelProfile:
    """Simulated model performance characteristics."""
    prompt_tokens_mean: int = 512
    prompt_tokens_stddev: int = 256
    output_tokens_mean: int = 128
    output_tokens_stddev: int = 64
    # Simulated processing speeds (tokens/sec)
    prompt_proc_speed: float = 5000.0
    output_proc_speed: float = 50.0
    # Simulated SSD load latency (ms per 1K tokens of prompt)
    ssd_load_ms_per_k: float = 50.0
    is_mha_model: bool = True


def simulate_request_tokens(
    metrics: RequestMetrics,
    profile: ModelProfile,
    rng: random.Random,
) -> None:
    """Simulate prompt + output token counts for a request."""
    prompt = max(1, int(rng.gauss(profile.prompt_tokens_mean, profile.prompt_tokens_stddev)))
    output = max(1, int(rng.gauss(profile.output_tokens_mean, profile.output_tokens_stddev)))
    metrics.prompt_tokens = prompt
    metrics.output_tokens = output


def simulate_ssd_load_wait(metrics: RequestMetrics, profile: ModelProfile) -> float:
    """Simulate SSD load wait time based on prompt size."""
    wait_ms = profile.ssd_load_ms_per_k * (metrics.prompt_tokens / 1000.0)
    metrics.ssd_load_wait_ms = wait_ms
    return wait_ms / 1000.0  # seconds


# ── FIFO Baseline Scheduler ──────────────────────────────────────────

def run_fifo_benchmark(
    arrivals: list[float],
    profile: ModelProfile,
    name: str = "FIFO Baseline",
) -> list[RequestMetrics]:
    """
    Run a FIFO (First-In-First-Out) baseline benchmark.

    No I/O awareness: all requests wait for SSD load sequentially.
    """
    rng = random.Random(42)
    metrics_list: list[RequestMetrics] = []
    current_time = 0.0

    for i, arrival in enumerate(arrivals):
        m = RequestMetrics(request_id=f"req_{i}")
        m.arrival_time = arrival
        simulate_request_tokens(m, profile, rng)

        # Request starts processing when it arrives (or when GPU is free)
        start_time = max(arrival, current_time)
        current_time = start_time

        # SSD load (blocking)
        ssd_wait = simulate_ssd_load_wait(m, profile)
        current_time += ssd_wait

        # Prompt processing
        prompt_time = m.prompt_tokens / profile.prompt_proc_speed
        current_time += prompt_time

        # First token ready
        m.first_token_time = current_time

        # Output tokens generated one-by-one
        for tok_idx in range(m.output_tokens):
            m.token_timestamps.append(current_time)
            current_time += 1.0 / profile.output_proc_speed

        m.last_token_time = current_time
        m.completion_time = current_time
        metrics_list.append(m)

    return metrics_list


# ── Bidaw I/O-Aware Scheduler ─────────────────────────────────────────

def run_bidaw_benchmark(
    arrivals: list[float],
    profile: ModelProfile,
    name: str = "Bidaw I/O-Aware",
) -> list[RequestMetrics]:
    """
    Run a Bidaw I/O-aware benchmark simulation.

    Key differences from FIFO:
    1. Dual queues: ready (data in memory) and preparing (loading from SSD)
    2. HRRN scheduling: small requests with long wait times get priority
    3. Parallel SSD loading: multiple requests can load simultaneously
    4. Tensor 6 caching (MHA): ~50% KV size reduction for cached requests

    Simplified simulation model:
    - SSD loads happen in parallel (up to io_parallelism)
    - GPU executes one request at a time
    - HRRN score determines scheduling order from ready queue
    - Tensor 6 cache hit rate increases with reuse (returning users)
    """
    io_parallelism = 4
    kv_size_unit = 1e8

    rng = random.Random(42)
    metrics_list: list[RequestMetrics] = []
    current_time = 0.0

    # Track which requests are in which state
    ready_queue: list[RequestMetrics] = []
    preparing_queue: list[tuple[RequestMetrics, float]] = []  # (metrics, load_complete_time)
    active_io_slots = 0

    # Track user IDs for tensor cache simulation
    user_cache: dict[str, int] = {}  # user_id -> access count
    cache_hit_rate_base = 0.15  # base cache hit rate per access

    request_idx = 0
    completed = 0
    total_requests = len(arrivals)

    while completed < total_requests:
        # Check for new arrivals
        while request_idx < total_requests and arrivals[request_idx] <= current_time:
            m = RequestMetrics(request_id=f"req_{request_idx}")
            m.arrival_time = arrivals[request_idx]
            simulate_request_tokens(m, profile, rng)
            user_id = f"user_{request_idx % 10}"  # 10 users cycling

            # Cache hit probability increases with reuse
            access_count = user_cache.get(user_id, 0)
            cache_hit_prob = min(0.7, cache_hit_rate_base + access_count * 0.08)
            user_cache[user_id] = access_count + 1
            cache_hit = rng.random() < cache_hit_prob

            if cache_hit:
                # Tensor 6 cached: skip SSD load, reduced prompt time
                m.ssd_load_wait_ms = 0.0
                ready_queue.append(m)
            else:
                # Need SSD load
                ssd_wait = simulate_ssd_load_wait(m, profile)
                complete_time = current_time + ssd_wait
                preparing_queue.append((m, complete_time))

                # Start I/O if slot available
                if active_io_slots < io_parallelism:
                    active_io_slots += 1
                else:
                    # I/O slot busy, delay load start
                    complete_time += ssd_wait * 0.5
                    preparing_queue[-1] = (m, complete_time)

            request_idx += 1

        # Check for completed SSD loads
        still_preparing = []
        for m, complete_time in preparing_queue:
            if complete_time <= current_time:
                ready_queue.append(m)
                active_io_slots = max(0, active_io_slots - 1)
            else:
                still_preparing.append((m, complete_time))
        preparing_queue = still_preparing

        # Schedule from ready queue using HRRN
        if ready_queue:
            # Score each request: 1 + wait_time / (kv_size / kv_size_unit)
            scored = []
            for m in ready_queue:
                wait_time = current_time - m.arrival_time
                kv_size = m.prompt_tokens * 50 if profile.is_mha_model else m.prompt_tokens * 100
                if kv_size <= 0:
                    score = float("inf")
                else:
                    score = 1.0 + wait_time / (kv_size / kv_size_unit)
                scored.append((score, m))
            scored.sort(key=lambda x: -x[0])  # highest score first

            # Execute highest-priority request
            _, m = scored[0]
            ready_queue.remove(m)

            # Cache hit: reduced prompt processing
            cache_hit_factor = 0.5 if m.ssd_load_wait_ms == 0 else 1.0
            prompt_time = m.prompt_tokens / profile.prompt_proc_speed * cache_hit_factor

            # First token
            current_time += prompt_time
            m.first_token_time = current_time

            # Output tokens
            for _ in range(m.output_tokens):
                m.token_timestamps.append(current_time)
                current_time += 1.0 / profile.output_proc_speed

            m.last_token_time = current_time
            m.completion_time = current_time
            metrics_list.append(m)
            completed += 1
        elif preparing_queue:
            # No ready requests, advance time to next SSD load completion
            next_complete = min(ct for _, ct in preparing_queue)
            current_time = max(current_time, next_complete)
        else:
            # Nothing ready, advance to next arrival
            if request_idx < total_requests:
                current_time = max(current_time, arrivals[request_idx])
            else:
                break

    # Sort by request_id for consistent ordering
    metrics_list.sort(key=lambda m: int(m.request_id.split("_")[1]))
    return metrics_list


# ── Result Aggregation ────────────────────────────────────────────────

def compute_result(
    name: str,
    metrics_list: list[RequestMetrics],
    request_rate: float,
) -> BenchmarkResult:
    """Aggregate per-request metrics into a BenchmarkResult."""
    if not metrics_list:
        raise ValueError("No metrics to aggregate")

    total_time = max(m.completion_time for m in metrics_list) - min(m.arrival_time for m in metrics_list)
    total_tokens = sum(m.output_tokens for m in metrics_list)

    ttfts = [m.ttft for m in metrics_list if m.ttft > 0]
    tbts = [m.tpot for m in metrics_list if m.tpot > 0]
    totals = [m.total_latency_ms for m in metrics_list if m.total_latency_ms > 0]
    ssd_waits = [m.ssd_load_wait_ms for m in metrics_list]

    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        return float(statistics.quantiles(data, n=100)[min(p - 1, 98)])

    return BenchmarkResult(
        name=name,
        num_requests=len(metrics_list),
        request_rate=request_rate,
        total_time_s=total_time,
        req_per_sec=len(metrics_list) / total_time if total_time > 0 else 0,
        tok_per_sec=total_tokens / total_time if total_time > 0 else 0,
        ttft_p50=percentile(ttfts, 50),
        ttft_p90=percentile(ttfts, 90),
        ttft_p99=percentile(ttfts, 99),
        ttft_avg=statistics.mean(ttfts) if ttfts else 0,
        tbt_p50=percentile(tbts, 50),
        tbt_p90=percentile(tbts, 90),
        tbt_avg=statistics.mean(tbts) if tbts else 0,
        total_p50=percentile(totals, 50),
        total_p90=percentile(totals, 90),
        total_avg=statistics.mean(totals) if totals else 0,
        avg_ssd_wait_ms=statistics.mean(ssd_waits) if ssd_waits else 0,
    )


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bidaw Benchmark: Poisson arrival simulation with TTFT/TBT/throughput metrics",
    )
    parser.add_argument("--num-requests", type=int, default=100,
                        help="Number of requests to simulate (default: 100)")
    parser.add_argument("--req-rate", type=float, default=2.0,
                        help="Request arrival rate in requests/sec (default: 2.0)")
    parser.add_argument("--enable-bidaw", action="store_true",
                        help="Run Bidaw I/O-aware scheduler (default: FIFO baseline)")
    parser.add_argument("--compare", action="store_true",
                        help="Run both FIFO and Bidaw and compare")
    parser.add_argument("--prompt-mean", type=int, default=512,
                        help="Mean prompt tokens (default: 512)")
    parser.add_argument("--output-mean", type=int, default=128,
                        help="Mean output tokens (default: 128)")
    parser.add_argument("--ssd-latency", type=float, default=50.0,
                        help="SSD load ms per 1K prompt tokens (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--json-output", type=str, default="",
                        help="Path to write JSON results")
    args = parser.parse_args()

    profile = ModelProfile(
        prompt_tokens_mean=args.prompt_mean,
        output_tokens_mean=args.output_mean,
        ssd_load_ms_per_k=args.ssd_latency,
    )

    arrivals = generate_poisson_arrivals(args.num_requests, args.req_rate, args.seed)

    results: list[BenchmarkResult] = []

    if args.compare or not args.enable_bidaw:
        print("\n[1/2] Running FIFO Baseline...")
        fifo_metrics = run_fifo_benchmark(arrivals, profile)
        fifo_result = compute_result("FIFO Baseline", fifo_metrics, args.req_rate)
        results.append(fifo_result)
        print(fifo_result.summary())

    if args.compare or args.enable_bidaw:
        label = "[2/2]" if args.compare else "[1/1]"
        print(f"\n{label} Running Bidaw I/O-Aware...")
        bidaw_metrics = run_bidaw_benchmark(arrivals, profile)
        bidaw_result = compute_result("Bidaw I/O-Aware", bidaw_metrics, args.req_rate)
        results.append(bidaw_result)
        print(bidaw_result.summary())

    # Comparison summary
    if len(results) == 2:
        fifo_r, bidaw_r = results
        print(f"\n{'='*60}")
        print("  Comparison: Bidaw vs FIFO")
        print(f"{'='*60}")
        print(f"  TTFT p50:    {bidaw_r.ttft_p50:.1f}ms vs {fifo_r.ttft_p50:.1f}ms  "
              f"({'+' if bidaw_r.ttft_p50 > fifo_r.ttft_p50 else ''}{((bidaw_r.ttft_p50 - fifo_r.ttft_p50) / fifo_r.ttft_p50 * 100):+.1f}%)")
        print(f"  TTFT p99:    {bidaw_r.ttft_p99:.1f}ms vs {fifo_r.ttft_p99:.1f}ms  "
              f"({'+' if bidaw_r.ttft_p99 > fifo_r.ttft_p99 else ''}{((bidaw_r.ttft_p99 - fifo_r.ttft_p99) / fifo_r.ttft_p99 * 100):+.1f}%)")
        print(f"  Throughput:  {bidaw_r.req_per_sec:.2f} vs {fifo_r.req_per_sec:.2f} req/s  "
              f"({'+' if bidaw_r.req_per_sec > fifo_r.req_per_sec else ''}{((bidaw_r.req_per_sec - fifo_r.req_per_sec) / fifo_r.req_per_sec * 100):+.1f}%)")
        print(f"  Total p50:   {bidaw_r.total_p50:.1f}ms vs {fifo_r.total_p50:.1f}ms  "
              f"({'+' if bidaw_r.total_p50 > fifo_r.total_p50 else ''}{((bidaw_r.total_p50 - fifo_r.total_p50) / fifo_r.total_p50 * 100):+.1f}%)")
        print(f"{'='*60}")

    if args.json_output:
        json_data = [asdict(r) for r in results]
        with open(args.json_output, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults written to {args.json_output}")


if __name__ == "__main__":
    main()
