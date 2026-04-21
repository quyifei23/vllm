# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bidaw 配置类

Bidaw 论文实现的顶层配置，包含三个核心机制及辅助组件的参数：
1. I/O 感知请求调度（双队列 + disk-HRRN）
2. 基于前轮答案长度的驱逐策略（ALE + ghost cache）
3. 存储高效张量缓存（normalized activation 缓存）
4. 混合粒度 GPU 内存分配

注意：
- 直接使用 pydantic dataclass（不通过 vllm.config.utils），避免循环导入。
- is_mha_model 改为从 ModelConfig 自动检测（num_kv_heads == num_attention_heads）。
- kv_size_unit 改为从 ModelConfig 自动推导单 token KV 大小。
- transform_stream_priority 为内部实现细节，不暴露为配置。
- enable_inclusive_caching 始终为 True，简化代码复杂度。
"""

import hashlib
import json

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass


@dataclass(config=ConfigDict(extra="forbid"))
class BidawConfig:
    """Bidaw 论文实现的配置类。

    所有 Bidaw 功能通过 enable_bidaw 总开关控制，关闭时
    完全退化为原始 vLLM 行为。
    """

    # ── 总开关 ──────────────────────────────────────────────────
    enable_bidaw: bool = False
    """是否启用 Bidaw。关闭时所有 Bidaw 组件不生效，完全退化为 vLLM 原始行为。"""

    # ── 机制一：I/O 感知调度 ────────────────────────────────────
    enable_io_aware_scheduling: bool = True
    """是否启用 I/O 感知调度器（双队列 + disk-HRRN 策略）。"""

    skip_oversize_requests: bool = True
    """ready queue 中若第一个请求不适合当前 GPU 内存，
    是否跳过并尝试后续请求（论文 §3.2，对应 FlashGen 的做法）。
    """

    # ── 机制二：基于前轮答案长度的驱逐策略 ────────────────────────
    enable_answer_length_eviction: bool = True
    """是否启用基于前轮答案长度的 KV 驱逐策略（ALE）。"""

    num_promising_buckets: int = 20
    """Ghost cache 中 weighted reuse distance 的分桶数量 m。
    论文称 "multiple fine-grained buckets"，具体数量未给出，建议 20。
    """

    eviction_threshold: float = 0.05
    """触发驱逐的阈值：性能层空闲空间低于总容量的此比例时触发。
    论文 §5.6: "free host memory falls below 5%"。
    """

    ghost_cache_history_size: int = 10000
    """User 级别 KV 访问 trace 的记录长度。
    Ghost cache 在后台对历史访问 trace 运行 Belady 最优策略，
    统计各 weighted reuse distance 桶的命中率，为驱逐决策提供参考。
    """

    # ── 机制三：存储高效张量缓存 ─────────────────────────────────
    enable_storage_efficient_tensor: bool = True
    """是否启用存储高效张量缓存（缓存 normalized activation 而非 KV tensor）。"""

    # ── SSD 存储路径 ─────────────────────────────────────────────
    ssd_cache_dir: str = "/tmp/bidaw_kv_cache"
    """SSD 容量层的缓存路径。"""

    def compute_hash(self) -> str:
        """提供唯一标识此配置的 hash 值。

        所有 Bidaw 参数都会影响编译/执行图，因此全部参与 hash 计算。
        """
        params = {
            "enable_bidaw": self.enable_bidaw,
            "enable_io_aware_scheduling": self.enable_io_aware_scheduling,
            "skip_oversize_requests": self.skip_oversize_requests,
            "enable_answer_length_eviction": self.enable_answer_length_eviction,
            "num_promising_buckets": self.num_promising_buckets,
            "eviction_threshold": self.eviction_threshold,
            "ghost_cache_history_size": self.ghost_cache_history_size,
            "enable_storage_efficient_tensor": self.enable_storage_efficient_tensor,
            "ssd_cache_dir": self.ssd_cache_dir,
        }
        data = json.dumps(params, sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()
