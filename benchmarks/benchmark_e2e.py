# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bidaw E2E Benchmark — Real vLLM inference with ShareGPT workload.

Compares Bidaw (with I/O-aware scheduling, ALE eviction, tensor caching)
against the vLLM baseline on real model inference.

Usage:
    # Process ShareGPT first:
    python benchmarks/process_sharegpt.py --max-samples 200

    # Run benchmark (baseline only):
    python benchmarks/benchmark_e2e.py --baseline

    # Run benchmark (Bidaw only):
    python benchmarks/benchmark_e2e.py --enable-bidaw

    # Run both and compare:
    python benchmarks/benchmark_e2e.py --compare

    # Custom model/SSD settings:
    python benchmarks/benchmark_e2e.py --compare \
        --max-requests 100 \
        --ssd-cache-dir /tmp/bidaw_e2e_test
"""

import argparse
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# ── Metric Recording ──────────────────────────────────────────────────

@dataclass
class RequestMetric:
    """Per-request timing metrics from real inference."""
    request_id: str
    user_id: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float = 0.0       # Time to first token
    tbt_ms: float = 0.0        # Time between tokens
    total_ms: float = 0.0      # Total request latency
    token_timestamps: list[float] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""
    name: str
    model: str
    num_requests: int
    total_time_s: float
    # Throughput
    req_per_sec: float
    tok_per_sec: float
    # TTFT (ms)
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float
    ttft_avg: float
    # TBT (ms)
    tbt_p50: float
    tbt_p90: float
    tbt_avg: float
    # Total latency (ms)
    total_p50: float
    total_p90: float
    total_avg: float

    def summary(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  Benchmark: {self.name}",
            f"  Model: {self.model}",
            f"{'='*60}",
            f"  Requests:        {self.num_requests}",
            f"  Total time:      {self.total_time_s:.2f}s",
            f"",
            f"  Throughput:",
            f"    Requests/sec:  {self.req_per_sec:.2f}",
            f"    Tokens/sec:    {self.tok_per_sec:.2f}",
            f"",
            f"  TTFT (ms):",
            f"    p50: {self.ttft_p50:.1f}   p90: {self.tbt_p90:.1f}   p99: {self.ttft_p99:.1f}   avg: {self.ttft_avg:.1f}",
            f"",
            f"  TBT (ms):",
            f"    p50: {self.tbt_p50:.1f}   p90: {self.tbt_p90:.1f}   avg: {self.tbt_avg:.1f}",
            f"",
            f"  Total Latency (ms):",
            f"    p50: {self.total_p50:.1f}   p90: {self.total_p90:.1f}   avg: {self.total_avg:.1f}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    return float(statistics.quantiles(data, n=100)[min(p - 1, 98)])


def compute_result(
    name: str,
    model: str,
    metrics: list[RequestMetric],
    total_time: float,
) -> BenchmarkResult:
    ttfts = [m.ttft_ms for m in metrics if m.ttft_ms > 0]
    tbts = [m.tbt_ms for m in metrics if m.tbt_ms > 0]
    totals = [m.total_ms for m in metrics if m.total_ms > 0]
    total_tokens = sum(m.output_tokens for m in metrics)

    return BenchmarkResult(
        name=name,
        model=model,
        num_requests=len(metrics),
        total_time_s=total_time,
        req_per_sec=len(metrics) / total_time if total_time > 0 else 0,
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
    )


# ── Load Workload ─────────────────────────────────────────────────────

def load_workload(path: str, max_requests: int = 0) -> list[dict]:
    """Load processed ShareGPT workload."""
    with open(path, "r") as f:
        data = json.load(f)
    if max_requests > 0:
        data = data[:max_requests]
    return data


# ── Baseline Benchmark ────────────────────────────────────────────────

def run_baseline(model_path: str, workload: list[dict]) -> list[RequestMetric]:
    """Run vLLM baseline (no Bidaw)."""
    from vllm import LLM, SamplingParams

    print(f"  Creating vLLM engine (baseline)...")
    llm = LLM(
        model=model_path,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,
        disable_log_stats=False,
    )
    print(f"  Engine ready")

    prompts = [r["prompt"] for r in workload]
    max_tokens = max(r["output_tokens"] for r in workload)
    max_tokens = min(max_tokens, 512)  # Cap for faster testing

    print(f"  Generating with max_tokens={max_tokens}...")
    start = time.monotonic()
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=max_tokens, temperature=0.0),
        use_tqdm=True,
    )
    total_time = time.monotonic() - start

    metrics = []
    for i, (req, out) in enumerate(zip(workload, outputs)):
        m = RequestMetric(
            request_id=f"req_{i}",
            user_id=req["user_id"],
            prompt_tokens=len(out.prompt_token_ids),
            output_tokens=len(out.outputs[0].token_ids),
            total_ms=0.0,
            ttft_ms=0.0,
        )
        if out.metrics is not None:
            m.ttft_ms = out.metrics.first_token_latency * 1000 if out.metrics.first_token_latency else 0.0
            # last_token_ts and first_token_ts are monotonic (same clock domain)
            if out.metrics.last_token_ts and out.metrics.first_token_ts:
                m.total_ms = out.metrics.first_token_latency * 1000 + (out.metrics.last_token_ts - out.metrics.first_token_ts) * 1000
            # TBT = time between tokens (generation time / num gen tokens)
            if out.metrics.last_token_ts and out.metrics.first_token_ts and out.metrics.num_generation_tokens > 1:
                m.tbt_ms = ((out.metrics.last_token_ts - out.metrics.first_token_ts) * 1000) / (out.metrics.num_generation_tokens - 1)
        metrics.append(m)

    return metrics, total_time


# ── Bidaw Benchmark ───────────────────────────────────────────────────

def run_bidaw(model_path: str, workload: list[dict], ssd_cache_dir: str) -> list[RequestMetric]:
    """Run vLLM with Bidaw enabled."""
    from vllm import LLM, SamplingParams

    print(f"  Creating vLLM engine (Bidaw)...")
    llm = LLM(
        model=model_path,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,
        enable_bidaw=True,
        ssd_cache_dir=ssd_cache_dir,
        disable_log_stats=False,
    )
    print(f"  Engine ready")

    prompts = [r["prompt"] for r in workload]
    max_tokens = max(r["output_tokens"] for r in workload)
    max_tokens = min(max_tokens, 512)

    print(f"  Generating with max_tokens={max_tokens}...")
    start = time.monotonic()
    outputs = llm.generate(
        prompts,
        SamplingParams(max_tokens=max_tokens, temperature=0.0),
        use_tqdm=True,
    )
    total_time = time.monotonic() - start

    metrics = []
    for i, (req, out) in enumerate(zip(workload, outputs)):
        m = RequestMetric(
            request_id=f"req_{i}",
            user_id=req["user_id"],
            prompt_tokens=len(out.prompt_token_ids),
            output_tokens=len(out.outputs[0].token_ids),
            total_ms=0.0,
            ttft_ms=0.0,
        )
        if out.metrics is not None:
            m.ttft_ms = out.metrics.first_token_latency * 1000 if out.metrics.first_token_latency else 0.0
            # last_token_ts and first_token_ts are monotonic (same clock domain)
            if out.metrics.last_token_ts and out.metrics.first_token_ts:
                m.total_ms = out.metrics.first_token_latency * 1000 + (out.metrics.last_token_ts - out.metrics.first_token_ts) * 1000
            # TBT = time between tokens (generation time / num gen tokens)
            if out.metrics.last_token_ts and out.metrics.first_token_ts and out.metrics.num_generation_tokens > 1:
                m.tbt_ms = ((out.metrics.last_token_ts - out.metrics.first_token_ts) * 1000) / (out.metrics.num_generation_tokens - 1)
        metrics.append(m)

    return metrics, total_time


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bidaw E2E Benchmark")
    parser.add_argument("--model", type=str, default="/root/autodl-fs/model/OPT-6.7B")
    parser.add_argument("--workload", type=str, default="data/sharegpt_workload.json")
    parser.add_argument("--max-requests", type=int, default=50)
    parser.add_argument("--ssd-cache-dir", type=str, default="/tmp/bidaw_e2e_test")
    parser.add_argument("--baseline", action="store_true", help="Run baseline only")
    parser.add_argument("--enable-bidaw", action="store_true", help="Run Bidaw only")
    parser.add_argument("--compare", action="store_true", help="Run both and compare")
    parser.add_argument("--output-json", type=str, default="data/e2e_results.json")
    args = parser.parse_args()

    # Default to compare if no mode specified
    if not args.baseline and not args.enable_bidaw:
        args.compare = True

    # Load workload
    print(f"Loading workload from {args.workload}...")
    workload = load_workload(args.workload, 0)
    print(f"  {len(workload)} raw requests")

    # Filter requests fitting in model's 2048 context window
    MODEL_MAX = 2048
    cap_output = min(max(r["output_tokens"] for r in workload), 512)
    workload = [r for r in workload if r["prompt_tokens"] + cap_output <= MODEL_MAX]
    if args.max_requests > 0:
        workload = workload[:args.max_requests]
    print(f"  {len(workload)} requests fitting in {MODEL_MAX} context (cap output={cap_output})")

    results = []

    if args.compare or args.baseline:
        label = "[1/2]" if args.compare else "[1/1]"
        print(f"\n{label} Running Baseline (vLLM without Bidaw)...")
        baseline_metrics, baseline_time = run_baseline(args.model, workload)
        result = compute_result("vLLM Baseline", args.model, baseline_metrics, baseline_time)
        results.append(result)
        print(result.summary())

    if args.compare or args.enable_bidaw:
        label = "[2/2]" if args.compare else "[1/1]"
        print(f"\n{label} Running Bidaw (vLLM with Bidaw)...")
        bidaw_metrics, bidaw_time = run_bidaw(args.model, workload, args.ssd_cache_dir)
        result = compute_result("Bidaw I/O-Aware", args.model, bidaw_metrics, bidaw_time)
        results.append(result)
        print(result.summary())

    # Comparison
    if len(results) == 2:
        baseline_r, bidaw_r = results
        print(f"\n{'='*60}")
        print("  Comparison: Bidaw vs Baseline")
        print(f"{'='*60}")
        print(f"  TTFT p50:    {bidaw_r.ttft_p50:.1f}ms vs {baseline_r.ttft_p50:.1f}ms  "
              f"({((bidaw_r.ttft_p50 - baseline_r.ttft_p50) / baseline_r.ttft_p50 * 100):+.1f}%)")
        print(f"  Throughput:  {bidaw_r.req_per_sec:.2f} vs {baseline_r.req_per_sec:.2f} req/s  "
              f"({((bidaw_r.req_per_sec - baseline_r.req_per_sec) / baseline_r.req_per_sec * 100):+.1f}%)")
        print(f"  Total p50:   {bidaw_r.total_p50:.1f}ms vs {baseline_r.total_p50:.1f}ms  "
              f"({((bidaw_r.total_p50 - baseline_r.total_p50) / baseline_r.total_p50 * 100):+.1f}%)")
        print(f"{'='*60}")

    # Save results
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        json_data = [asdict(r) for r in results]
        with open(args.output_json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
