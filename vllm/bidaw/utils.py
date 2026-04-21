# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bidaw 公共工具函数"""

from dataclasses import dataclass


@dataclass
class KVStorageStatus:
    """记录每个请求的 KV 存储状态。

    Attributes:
        storage_layer: KV 所在存储层级
            - "gpu": KV 在 GPU 显存中（HBM）
            - "host": KV 在 CPU 内存中（DRAM）
            - "ssd": KV 在 SSD 中（Disk）
            - "none": 无 KV 缓存（第一轮对话）
        kv_size_bytes: 该请求的 KV 总字节数
    """
    storage_layer: str = "none"
    kv_size_bytes: int = 0


def compute_kv_size_bytes(seq_len: int, model_config) -> int:
    """计算请求的 KV 总字节数。

    公式：= 2 × num_kv_heads × head_dim × dtype_size × num_layers × seq_len

    其中：
    - 2: K 和 V 两个张量
    - num_kv_heads: KV attention heads 总数（跨所有 GPU）
    - head_dim: 每个 attention head 的维度
    - dtype_size: 数据类型字节数（FP16=2, FP8=1）
    - num_layers: Transformer 总层数（跨所有 pipeline stage）
    - seq_len: 序列长度（tokens）

    注意：此函数计算的是模型**总的** KV 大小，不考虑 TP/PP 分片。
    disk-HRRN 调度公式中的 kv_size 使用此总值。

    Args:
        seq_len: 请求的序列长度（tokens）
        model_config: vLLM ModelConfig 对象

    Returns:
        KV 总字节数（int）。
    """
    if seq_len <= 0:
        return 0

    num_kv_heads = model_config.get_total_num_kv_heads()
    head_dim = model_config.get_head_size()
    dtype_size = model_config.dtype.itemsize
    num_layers = model_config.get_total_num_hidden_layers()

    return 2 * num_kv_heads * head_dim * dtype_size * num_layers * seq_len
