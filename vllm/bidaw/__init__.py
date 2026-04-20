# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.bidaw.ale import (
    AnswerLengthReusePredictor,
    BidawEvictionManager,
    GhostCache,
    WeightedReuseDistanceTracker,
)
from vllm.bidaw.block_allocator import BidawBlockAllocator
from vllm.bidaw.config import BidawConfig
from vllm.bidaw.scheduler import BidawScheduler, DiskHRRNScorer, RequestQueue, SSDLoadStatus
from vllm.bidaw.tensor_cache import BidawTensorCacher, _find_attention_layernorms

__all__ = [
    "AnswerLengthReusePredictor",
    "BidawBlockAllocator",
    "BidawConfig",
    "BidawEvictionManager",
    "BidawOffloadingManager",
    "BidawOffloadingSpec",
    "BidawScheduler",
    "BidawTensorCacher",
    "DiskHRRNScorer",
    "GhostCache",
    "RequestQueue",
    "SSDGPUOffloadingHandler",
    "SSDLoadStatus",
    "SSDLoadStoreSpec",
    "WeightedReuseDistanceTracker",
    "_find_attention_layernorms",
]

# Lazy imports for SSD modules (avoid circular import with vllm.config)
def __getattr__(name: str):
    if name in (
        "BidawOffloadingManager",
        "BidawOffloadingSpec",
        "SSDGPUOffloadingHandler",
        "SSDLoadStoreSpec",
    ):
        from vllm.bidaw.ssd import (
            BidawOffloadingManager,
            BidawOffloadingSpec,
            SSDGPUOffloadingHandler,
            SSDLoadStoreSpec,
        )
        return {
            "BidawOffloadingManager": BidawOffloadingManager,
            "BidawOffloadingSpec": BidawOffloadingSpec,
            "SSDGPUOffloadingHandler": SSDGPUOffloadingHandler,
            "SSDLoadStoreSpec": SSDLoadStoreSpec,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
