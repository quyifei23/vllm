# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Step 6: Storage-Efficient Tensor Cache.

Covers:
- BidawTensorCacher: store/load round-trip, reconstruction, memory bounds
- _find_attention_layernorms: model scanning for RMSNorm layers
- Circular buffer overflow: oldest entries evicted
"""

import pytest
import torch
import torch.nn as nn

from vllm.bidaw.tensor_cache import (
    BidawTensorCacher,
    _extract_layer_id,
    _find_attention_layernorms,
    _find_qkv_weights,
)


class TestBidawTensorCacher:
    def _make_cacher(self, **kwargs):
        return BidawTensorCacher(
            hidden_size=4096,
            num_layers=32,
            dtype=torch.float16,
            max_tokens=1000,
            memory_gb=0.0,
            **kwargs,
        )

    def test_init(self):
        cacher = self._make_cacher()
        assert cacher._hidden_size == 4096
        assert cacher._num_layers == 32
        assert cacher._max_tokens == 1000

    def test_store_and_load_tensor(self):
        cacher = self._make_cacher()

        # Create a fake normalized activation tensor
        norm_act = torch.randn(10, 4096, dtype=torch.float16)
        cacher.store_tensor(layer_id=0, token_start=0, norm_act=norm_act)

        # Load it back
        loaded = cacher.load_tensor(layer_id=0, token_start=0, num_tokens=10)
        assert loaded is not None
        assert loaded.shape == (10, 4096)
        assert loaded.dtype == torch.float16
        # Should be close to original (round-trip through CPU)
        assert torch.allclose(loaded, norm_act, atol=1e-5)

    def test_load_nonexistent_returns_none(self):
        cacher = self._make_cacher()
        result = cacher.load_tensor(layer_id=0, token_start=0, num_tokens=10)
        assert result is None

    def test_load_partial_overlap(self):
        cacher = self._make_cacher()

        # Store tokens 5-15
        norm_act = torch.randn(10, 4096, dtype=torch.float16)
        cacher.store_tensor(layer_id=0, token_start=5, norm_act=norm_act)

        # Request tokens 10-20 (partial overlap)
        loaded = cacher.load_tensor(layer_id=0, token_start=10, num_tokens=10)
        assert loaded is not None
        assert loaded.shape == (5, 4096)  # only 5 tokens overlap

    def test_memory_usage(self):
        cacher = self._make_cacher()

        norm_act = torch.randn(10, 4096, dtype=torch.float16)
        cacher.store_tensor(layer_id=0, token_start=0, norm_act=norm_act)

        usage = cacher.get_memory_usage()
        assert usage > 0
        # 10 * 4096 * 2 bytes (float16)
        expected = 10 * 4096 * 2
        assert usage == expected

    def test_clear(self):
        cacher = self._make_cacher()

        norm_act = torch.randn(10, 4096, dtype=torch.float16)
        cacher.store_tensor(layer_id=0, token_start=0, norm_act=norm_act)
        assert cacher.get_memory_usage() > 0

        cacher.clear()
        assert cacher.get_memory_usage() == 0

    def test_circular_buffer_overflow(self):
        cacher = BidawTensorCacher(
            hidden_size=100,
            num_layers=1,
            dtype=torch.float16,
            max_tokens=10,  # small capacity
        )

        # Fill beyond capacity
        for i in range(5):
            norm_act = torch.randn(5, 100, dtype=torch.float16)
            cacher.store_tensor(layer_id=0, token_start=i * 5, norm_act=norm_act)

        # Should have evicted oldest entries, keeping only 10 tokens worth
        # Last two stores (token_start=15 and 20) should be in buffer
        loaded = cacher.load_tensor(layer_id=0, token_start=0, num_tokens=5)
        assert loaded is None  # oldest evicted

        loaded = cacher.load_tensor(layer_id=0, token_start=20, num_tokens=5)
        assert loaded is not None

    def test_multi_layer_isolation(self):
        cacher = self._make_cacher()

        # Store different tensors in different layers
        for layer_id in range(3):
            norm_act = torch.randn(10, 4096, dtype=torch.float16) * (layer_id + 1)
            cacher.store_tensor(layer_id=layer_id, token_start=0, norm_act=norm_act)

        # Each layer should have its own data
        for layer_id in range(3):
            loaded = cacher.load_tensor(layer_id=layer_id, token_start=0, num_tokens=10)
            assert loaded is not None
            assert loaded.shape == (10, 4096)

    def test_register_layer_weights(self):
        cacher = self._make_cacher()
        w_k = torch.randn(4096, 4096, dtype=torch.float16)
        w_v = torch.randn(4096, 4096, dtype=torch.float16)

        cacher.register_layer_weights(layer_id=0, w_k=w_k, w_v=w_v)
        assert cacher._layer_weights[0] == (w_k, w_v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestReconstructKV:
    def test_reconstruct_kv_basic(self):
        cacher = BidawTensorCacher(
            hidden_size=64,
            num_layers=1,
            dtype=torch.float16,
            max_tokens=100,
        )

        w_k = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        w_v = torch.randn(64, 64, dtype=torch.float16, device="cuda")
        cacher.register_layer_weights(layer_id=0, w_k=w_k, w_v=w_v)

        norm_act = torch.randn(10, 64, dtype=torch.float16, device="cpu").pin_memory()
        result = cacher.reconstruct_kv(layer_id=0, norm_act_cpu=norm_act)

        assert result is not None
        K, V = result
        assert K.shape == (10, 64)
        assert V.shape == (10, 64)
        assert K.device.type == "cuda"
        assert V.device.type == "cuda"

    def test_reconstruct_kv_no_weights(self):
        cacher = BidawTensorCacher(
            hidden_size=64,
            num_layers=1,
            dtype=torch.float16,
            max_tokens=100,
        )

        norm_act = torch.randn(10, 64, dtype=torch.float16, device="cpu").pin_memory()
        result = cacher.reconstruct_kv(layer_id=0, norm_act_cpu=norm_act)
        assert result is None

    def test_reconstruct_kv_numerical_correctness(self):
        cacher = BidawTensorCacher(
            hidden_size=64,
            num_layers=1,
            dtype=torch.float32,  # use float32 for precision
            max_tokens=100,
        )

        w_k = torch.randn(64, 64, dtype=torch.float32, device="cuda")
        w_v = torch.randn(64, 64, dtype=torch.float32, device="cuda")
        cacher.register_layer_weights(layer_id=0, w_k=w_k, w_v=w_v)

        norm_act_cpu = torch.randn(10, 64, dtype=torch.float32, device="cpu").pin_memory()
        result = cacher.reconstruct_kv(layer_id=0, norm_act_cpu=norm_act_cpu)
        assert result is not None
        K, V = result

        # Compute expected values
        norm_act_gpu = norm_act_cpu.cuda()
        expected_K = torch.matmul(norm_act_gpu, w_k.T)
        expected_V = torch.matmul(norm_act_gpu, w_v.T)

        assert torch.allclose(K, expected_K, atol=1e-3)
        assert torch.allclose(V, expected_V, atol=1e-3)


class TestFindAttentionLayernorms:
    def test_find_layernorms_in_mock_model(self):
        from vllm.model_executor.layers.layernorm import RMSNorm

        # Mock RMSNorm that passes isinstance() but avoids CustomOp init
        class MockRMSNorm(RMSNorm):
            def __init__(self, dim: int, eps: float = 1e-5):
                nn.Module.__init__(self)

        class MockDecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = MockRMSNorm(128)
                self.post_attention_layernorm = MockRMSNorm(128)

        class MockModel(nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.layers = nn.ModuleList(
                    [MockDecoderLayer() for _ in range(num_layers)]
                )

        model = MockModel(num_layers=4)
        layernorms = _find_attention_layernorms(model)

        assert len(layernorms) == 4
        for layer_id in range(4):
            assert layer_id in layernorms
            rmsnorm, _ = layernorms[layer_id]
            assert isinstance(rmsnorm, RMSNorm)

    def test_no_layernorms_in_empty_model(self):
        class EmptyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

        model = EmptyModel()
        layernorms = _find_attention_layernorms(model)
        assert len(layernorms) == 0


class TestExtractLayerId:
    def test_standard_name(self):
        assert _extract_layer_id("model.layers.3.input_layernorm") == 3
        assert _extract_layer_id("model.layers.0.k_proj") == 0
        assert _extract_layer_id("transformer.layers.31.v_proj") == 31

    def test_no_layers_in_name(self):
        assert _extract_layer_id("model.head") is None
        assert _extract_layer_id("linear") is None

    def test_non_numeric_layer(self):
        assert _extract_layer_id("model.layers.foo.input_layernorm") is None


class TestFindQKVWeights:
    def test_extract_layer_id_for_weight_lookup(self):
        """Verify _extract_layer_id works for weight module names."""
        assert _extract_layer_id("model.layers.3.k_proj") == 3
        assert _extract_layer_id("model.layers.5.v_proj") == 5
        assert _extract_layer_id("model.layers.7.qkv_proj") == 7
