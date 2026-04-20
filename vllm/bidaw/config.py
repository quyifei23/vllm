# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Bidaw: I/O-Aware KV Cache Management for Interactive LLM Serving

Based on the paper "Bidaw: I/O-Aware KV Cache Management for Interactive LLM Serving"
"""

from dataclasses import dataclass

from vllm.utils.hashing import safe_hash


@dataclass
class BidawConfig:
    """
    Bidaw global configuration, shared across all components.

    Parameters sourced from the paper's experimental setup (§5):
    - A800 80GB GPU
    - 200GB host memory (performance layer)
    - 1.5 GB/s SSD (4x SATA SSD RAID-5) as capacity layer
    - PCIe Gen 4 connecting GPU and host
    """

    # ── Master switch ──────────────────────────────────────────────
    enable_bidaw: bool = False
    """Master switch to enable all Bidaw features."""

    # ── Mechanism 1: I/O-Aware Scheduling (§3.2) ───────────────────
    enable_io_aware_scheduling: bool = True
    """Enable dual-queue (ready/preparing) scheduling with disk-HRRN."""

    kv_size_unit: float = 1e8
    """Unit (bytes) for kv_size in disk-HRRN: ratio = 1 + wait_time / (kv_size / unit)."""

    skip_oversize_requests: bool = True
    """Skip requests that don't fit in current GPU memory (per FlashGen approach)."""

    # ── Mechanism 2: Answer-Length-Based Eviction (§3.3) ───────────
    enable_answer_length_eviction: bool = True
    """Enable ALE eviction policy using previous-round answer length."""

    num_promising_buckets: int = 20
    """Number of weighted reuse distance buckets in ghost cache."""

    eviction_threshold: float = 0.05
    """Trigger eviction when free host memory falls below this ratio."""

    ghost_cache_history_size: int = 10000
    """Historical trace length for ghost cache statistics."""

    # ── Mechanism 3: Storage-Efficient Tensor Cache (§4) ───────────
    enable_storage_efficient_tensor: bool = True
    """Cache storage-efficient tensors (normalized activations) instead of KV."""

    is_mha_model: bool = True
    """Whether the model uses MHA (True) or GQA (False). MHA uses tensor6 caching."""

    transform_stream_priority: int = -1
    """CUDA stream priority for tensor6→KV reconstruction (negative = low priority)."""

    # ── Mixed-Granularity GPU Memory Allocation (§4) ───────────────
    big_block_size: int = 256
    """Token count per big block, for history and query tokens."""

    small_block_size: int = 16
    """Token count per small block, for response tokens (unpredictable length)."""

    # ── Multi-GPU Parallelism ──────────────────────────────────────
    tensor_parallel_size: int = 1
    """Tensor parallelism size (will be synced with parallel_config)."""

    pipeline_parallel_size: int = 1
    """Pipeline parallelism size (will be synced with parallel_config)."""

    # ── SSD Storage ────────────────────────────────────────────────
    ssd_cache_dir: str = "/tmp/bidaw_kv_cache"
    """Directory for SSD-based KV cache (capacity layer)."""

    ssd_io_threads: int = 4
    """Number of threads for async SSD I/O operations."""

    # ── Inclusive Caching (§4) ─────────────────────────────────────
    enable_inclusive_caching: bool = True
    """Keep copies in performance layer when evicting to capacity layer."""

    # ── Tensor Cache Memory ────────────────────────────────────────
    tensor_cache_memory_gb: float = 0.0
    """Max memory (GB) for storage-efficient tensor cache. 0 = auto."""

    def compute_hash(self) -> str:
        """Compute a hash string representing this config's state."""
        factors = [
            self.enable_bidaw,
            self.enable_io_aware_scheduling,
            self.kv_size_unit,
            self.skip_oversize_requests,
            self.enable_answer_length_eviction,
            self.num_promising_buckets,
            self.eviction_threshold,
            self.ghost_cache_history_size,
            self.enable_storage_efficient_tensor,
            self.is_mha_model,
            self.transform_stream_priority,
            self.big_block_size,
            self.small_block_size,
            self.tensor_parallel_size,
            self.pipeline_parallel_size,
            self.ssd_cache_dir,
            self.ssd_io_threads,
            self.enable_inclusive_caching,
            self.tensor_cache_memory_gb,
        ]
        return safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()[:10]
