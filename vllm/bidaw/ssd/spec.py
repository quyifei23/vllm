# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
BidawOffloadingSpec - SSD offloading spec for Bidaw.

Follows the OffloadingSpec pattern (like CPUOffloadingSpec) to register
the Bidaw SSD manager and handlers with the v1 offloading factory.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.abstract import LoadStoreSpec, OffloadingManager
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCaches, OffloadingSpec
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

from vllm.bidaw.ssd.manager import BidawOffloadingManager
from vllm.bidaw.ssd.mediums import SSDLoadStoreSpec
from vllm.bidaw.ssd.worker import SSDGPUOffloadingHandler

if TYPE_CHECKING:
    pass


class BidawOffloadingSpec(OffloadingSpec):
    """
    OffloadingSpec for Bidaw's SSD-backed KV cache.

    Configuration via kv_connector_extra_config:
        ssd_cache_dir: Directory for SSD storage (default: /tmp/bidaw_kv_cache)
        ssd_io_threads: Number of I/O threads (default: 4)
        max_ssd_blocks: Maximum number of blocks on SSD (default: 100000)
        tp_rank: Tensor parallelism rank (default: 0)
        pp_rank: Pipeline parallelism rank (default: 0)
    """

    def __init__(
        self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
    ) -> None:
        super().__init__(vllm_config, kv_cache_config)

        self._ssd_cache_dir = self.extra_config.get("ssd_cache_dir", "/tmp/bidaw_kv_cache")
        self._ssd_io_threads = int(self.extra_config.get("ssd_io_threads", 4))
        self._max_ssd_blocks = int(self.extra_config.get("max_ssd_blocks", 100000))

        parallel_config = vllm_config.parallel_config
        self._tp_rank = parallel_config.rank // parallel_config.pipeline_parallel_size
        self._pp_rank = parallel_config.rank % parallel_config.pipeline_parallel_size

        # scheduler-side
        self._manager: BidawOffloadingManager | None = None
        # worker-side
        self._handler: SSDGPUOffloadingHandler | None = None

    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )
            self._manager = BidawOffloadingManager(
                ssd_cache_dir=self._ssd_cache_dir,
                ssd_io_threads=self._ssd_io_threads,
                max_blocks=self._max_ssd_blocks,
                tp_rank=self._tp_rank,
                pp_rank=self._pp_rank,
                enable_events=enable_events,
            )
        return self._manager

    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handler:
            if not current_platform.is_cuda_alike():
                raise Exception(
                    "Bidaw SSD Offloading is currently only supported on CUDA-alike GPUs"
                )

            self._handler = SSDGPUOffloadingHandler(
                kv_caches=kv_caches,
                ssd_cache_dir=self._ssd_cache_dir,
                io_threads=self._ssd_io_threads,
            )

        assert self._handler is not None
        yield GPULoadStoreSpec, SSDLoadStoreSpec, self._handler.gpu_to_ssd_handler
        yield SSDLoadStoreSpec, GPULoadStoreSpec, self._handler.ssd_to_gpu_handler
