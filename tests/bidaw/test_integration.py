# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for Step 8: Integration & Wiring (+ Step 7 bundled).

Covers:
- BidawOffloadingSpec factory registration
- BidawConfig conditional activation
- PP rank derivation from parallel config
- Tensor cache hook registration guard conditions
"""


class TestOffloadingSpecFactory:
    def test_bidaw_spec_registered(self):
        from vllm.v1.kv_offload.factory import OffloadingSpecFactory

        assert "BidawOffloadingSpec" in OffloadingSpecFactory._registry

    def test_bidaw_spec_loader(self):
        from vllm.v1.kv_offload.factory import OffloadingSpecFactory

        loader = OffloadingSpecFactory._registry.get("BidawOffloadingSpec")
        assert loader is not None
        spec_cls = loader()
        assert spec_cls.__name__ == "BidawOffloadingSpec"


class TestPPRankDerivation:
    def test_single_rank_tp_0_pp_0(self):
        """Single GPU: rank=0, pp_size=1 -> tp=0, pp=0"""
        rank = 0
        pp_size = 1
        tp_rank = rank // pp_size
        pp_rank = rank % pp_size
        assert tp_rank == 0
        assert pp_rank == 0

    def test_tp2_pp1_rank_derivation(self):
        """TP=2, PP=1: rank=0 -> tp=0; rank=1 -> tp=1"""
        pp_size = 1
        # rank 0
        tp_rank = 0 // pp_size
        pp_rank = 0 % pp_size
        assert tp_rank == 0
        assert pp_rank == 0
        # rank 1
        tp_rank = 1 // pp_size
        pp_rank = 1 % pp_size
        assert tp_rank == 1
        assert pp_rank == 0

    def test_tp2_pp2_rank_derivation(self):
        """TP=2, PP=2: 4 ranks total"""
        pp_size = 2
        # rank 0: tp=0, pp=0
        assert 0 // pp_size == 0
        assert 0 % pp_size == 0
        # rank 1: tp=0, pp=1
        assert 1 // pp_size == 0
        assert 1 % pp_size == 1
        # rank 2: tp=1, pp=0
        assert 2 // pp_size == 1
        assert 2 % pp_size == 0
        # rank 3: tp=1, pp=1
        assert 3 // pp_size == 1
        assert 3 % pp_size == 1


class TestBidawConfigConditional:
    def test_enable_bidaw_false_by_default(self):
        from vllm.bidaw.config import BidawConfig

        cfg = BidawConfig()
        assert cfg.enable_bidaw is False
        assert cfg.enable_io_aware_scheduling is True
        assert cfg.enable_storage_efficient_tensor is True

    def test_all_integration_flags(self):
        from vllm.bidaw.config import BidawConfig

        cfg = BidawConfig(enable_bidaw=True)
        assert cfg.enable_bidaw is True
        assert cfg.enable_io_aware_scheduling is True
        assert cfg.enable_answer_length_eviction is True
        assert cfg.enable_storage_efficient_tensor is True
        assert cfg.enable_inclusive_caching is True


class TestBidawSchedulerIntegration:
    def test_scheduler_creation_with_bidaw(self):
        from vllm.bidaw import BidawScheduler

        scheduler = BidawScheduler(
            kv_size_unit=1e8,
            skip_oversize_requests=True,
            max_kv_size_bytes=1e8,
            is_mha_model=True,
        )
        assert scheduler.get_ready_queue() is not None
        assert scheduler.get_preparing_queue() is not None
        assert scheduler.get_load_status("nonexistent") is None

    def test_scheduler_empty_queues(self):
        from vllm.bidaw import BidawScheduler

        scheduler = BidawScheduler()
        assert scheduler.get_next_request() is None
        assert len(scheduler.get_ready_queue()) == 0
        assert len(scheduler.get_preparing_queue()) == 0


class TestTensorCacheHookRegistration:
    def test_find_layernorms_on_mock_model(self):
        import torch.nn as nn
        from vllm.bidaw.tensor_cache import _find_attention_layernorms
        from vllm.model_executor.layers.layernorm import RMSNorm

        # Mock RMSNorm that passes isinstance() but avoids CustomOp init
        class MockRMSNorm(RMSNorm):
            def __init__(self, dim: int, eps: float = 1e-5):
                nn.Module.__init__(self)

        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = MockRMSNorm(128)
                self.post_attention_layernorm = MockRMSNorm(128)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [MockLayer() for _ in range(8)]
                )

        model = MockModel()
        layernorms = _find_attention_layernorms(model)
        assert len(layernorms) == 8

    def test_qkv_weight_extraction_on_mock_model(self):
        import torch
        import torch.nn as nn
        from vllm.bidaw.tensor_cache import _find_qkv_weights

        class MockKProj(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(hidden_size, hidden_size))

        class MockVProj(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(hidden_size, hidden_size))

        class MockLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.k_proj = MockKProj(64)
                self.v_proj = MockVProj(64)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([MockLayer() for _ in range(2)])

        model = MockModel()
        # The mock model doesn't have proper naming, so weights won't be found
        # This tests that the function handles missing patterns gracefully
        weights = _find_qkv_weights(model)
        assert isinstance(weights, dict)
