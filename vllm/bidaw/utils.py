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


def compute_kv_size_bytes(seq_len, model_config):
    """计算请求的 KV 总字节数。

    公式：= 2 × num_kv_heads × head_dim × dtype_size × num_layers × seq_len

    其中：
    - 2: K 和 V 两个张量
    - num_kv_heads: 从 model_config 获取
    - head_dim: 每个 attention head 的维度
    - dtype_size: 数据类型字节数（FP16=2, FP32=4, BF16=2）
    - num_layers: Transformer 层数
    - seq_len: 序列长度

    Args:
        seq_len: 请求的序列长度（tokens）
        model_config: vLLM ModelConfig 对象

    Returns:
        KV 总字节数（int）。
    """
    pass
