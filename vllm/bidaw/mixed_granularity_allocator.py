# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""混合粒度 GPU 内存块分配器（论文 §4）

此组件已舍弃，不再实现。

舍弃原因：
- 论文中混合粒度的收益主要来自减少分配器调用次数（8-12% GPU 利用率提升）
- 实际运行中 big_block 不断拆分后很难凑齐 16 个连续 small_block 合并
- 与 vLLM 现有 block_size 策略功能重合，引入的复杂度远大于收益
- vLLM 的 BlockPool 假设统一 block 大小，强行实现会侵入多个核心文件

后续如需 GPU 内存块优化，建议通过 vLLM 原生 BlockPool 配置（如 block_size）实现。
"""

raise NotImplementedError(
    "vllm.bidaw.mixed_granularity_allocator 已舍弃，不再实现。"
    "如需 GPU 内存块优化，请使用 vLLM 原生 BlockPool 机制。"
)
