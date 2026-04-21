# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bidaw 工具函数单元测试"""

from unittest.mock import MagicMock

import pytest

from vllm.bidaw.utils import KVStorageStatus, compute_kv_size_bytes


def _make_mock_model_config(
    num_kv_heads: int = 40,
    head_dim: int = 64,
    dtype_itemsize: int = 2,
    num_layers: int = 40,
) -> MagicMock:
    """创建模拟的 ModelConfig 对象。"""
    mc = MagicMock()
    mc.get_total_num_kv_heads.return_value = num_kv_heads
    mc.get_head_size.return_value = head_dim
    mc.dtype.itemsize = dtype_itemsize
    mc.get_total_num_hidden_layers.return_value = num_layers
    return mc


class TestComputeKVSizeBytes:
    """compute_kv_size_bytes 测试。"""

    def test_basic_calculation(self):
        """基本计算：OPT-13B 级别模型，seq_len=100。

        预期: 2 * 40 * 64 * 2 * 40 * 100 = 40,960,000 bytes
        """
        mc = _make_mock_model_config()
        result = compute_kv_size_bytes(100, mc)
        assert result == 2 * 40 * 64 * 2 * 40 * 100
        assert result == 40_960_000

    def test_zero_seq_len(self):
        """空序列应返回 0。"""
        mc = _make_mock_model_config()
        assert compute_kv_size_bytes(0, mc) == 0
        assert compute_kv_size_bytes(-1, mc) == 0

    def test_fp8_dtype(self):
        """FP8 (dtype_size=1) 的结果应为 FP16 的一半。"""
        mc_fp16 = _make_mock_model_config(dtype_itemsize=2)
        mc_fp8 = _make_mock_model_config(dtype_itemsize=1)
        size_fp16 = compute_kv_size_bytes(100, mc_fp16)
        size_fp8 = compute_kv_size_bytes(100, mc_fp8)
        assert size_fp8 == size_fp16 // 2

    def test_gqa_model(self):
        """GQA 模型（num_kv_heads < num_attention_heads）计算正确。"""
        # LLaMA 3 级别：32 个 attention heads，8 个 KV heads
        mc = _make_mock_model_config(num_kv_heads=8, num_layers=32)
        result = compute_kv_size_bytes(1000, mc)
        assert result == 2 * 8 * 64 * 2 * 32 * 1000

    def test_different_head_dim(self):
        """不同 head_dim 的计算正确。"""
        mc = _make_mock_model_config(head_dim=128)
        result = compute_kv_size_bytes(100, mc)
        assert result == 2 * 40 * 128 * 2 * 40 * 100

    def test_large_seq_len(self):
        """长序列计算不溢出。"""
        mc = _make_mock_model_config()
        result = compute_kv_size_bytes(32768, mc)
        expected = 2 * 40 * 64 * 2 * 40 * 32768
        assert result == expected


class TestKVStorageStatus:
    """KVStorageStatus 测试。"""

    def test_default_values(self):
        """默认值应为 storage_layer='none', kv_size_bytes=0。"""
        status = KVStorageStatus()
        assert status.storage_layer == "none"
        assert status.kv_size_bytes == 0

    def test_custom_values(self):
        """自定义值应正确设置。"""
        status = KVStorageStatus(storage_layer="ssd", kv_size_bytes=500_000)
        assert status.storage_layer == "ssd"
        assert status.kv_size_bytes == 500_000

    def test_all_storage_layers(self):
        """所有合法的 storage_layer 值都应可设置。"""
        for layer in ("gpu", "host", "ssd", "none"):
            status = KVStorageStatus(storage_layer=layer)
            assert status.storage_layer == layer
