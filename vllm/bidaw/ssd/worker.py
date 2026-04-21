# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SSD GPU Offloading Handler for Bidaw.

Handles async KV data transfers between GPU memory and SSD storage
using a thread pool for I/O operations and CUDA streams for GPU transfers.
"""

import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCaches
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
)

from vllm.bidaw.ssd.mediums import SSDLoadStoreSpec

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


@dataclass
class SSDTransfer:
    """Tracks an in-flight SSD transfer."""

    job_id: int
    num_bytes: int
    transfer_type: tuple[str, str]
    start_time: float
    done_event: threading.Event
    success: bool = False


class SSDDirectionHandler(OffloadingHandler):
    """
    Handles transfers in a single direction (GPU->SSD or SSD->GPU).

    Uses a thread pool for I/O operations.
    GPU->CPU copies are done synchronously within the transfer thread,
    then files are written inline to avoid pool exhaustion.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        ssd_cache_dir: str,
        io_threads: int,
        gpu_to_ssd: bool,
    ) -> None:
        self._kv_caches = kv_caches
        self._ssd_cache_dir = ssd_cache_dir
        self._gpu_to_ssd = gpu_to_ssd
        self._io_pool = ThreadPoolExecutor(
            max_workers=io_threads,
            thread_name_prefix=f"ssd_{'gpu_to_ssd' if gpu_to_ssd else 'ssd_to_gpu'}",
        )

        # Pending transfers queue
        self._pending: deque[SSDTransfer] = deque()
        self._finished: list[TransferResult] = []
        self._lock = threading.Lock()
        self._all_transfers: dict[int, SSDTransfer] = {}

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Initiate an async transfer.

        Args:
            job_id: Unique ID for completion notification.
            spec: (src_spec, dst_spec) transfer specification.

        Returns:
            True if transfer was submitted successfully.
        """
        src_spec, dst_spec = spec

        transfer = SSDTransfer(
            job_id=job_id,
            num_bytes=0,
            transfer_type=(src_spec.medium(), dst_spec.medium()),
            start_time=time.monotonic(),
            done_event=threading.Event(),
        )

        with self._lock:
            self._pending.append(transfer)
            self._all_transfers[job_id] = transfer

        # Submit I/O work
        if self._gpu_to_ssd:
            self._io_pool.submit(self._gpu_to_ssd_transfer, transfer, src_spec, dst_spec)
        else:
            self._io_pool.submit(self._ssd_to_gpu_transfer, transfer, src_spec, dst_spec)

        return True

    def _gpu_to_ssd_transfer(
        self,
        transfer: SSDTransfer,
        src_spec: LoadStoreSpec,
        dst_spec: SSDLoadStoreSpec,
    ) -> None:
        """Write KV data from GPU memory to SSD files."""
        try:
            tensor_infos = []
            for td in self._kv_caches.tensors:
                tensor_infos.append(td.tensor.view(torch.int8))

            block_ids = src_spec.block_ids
            if len(block_ids) == 0:
                transfer.success = True
                transfer.done_event.set()
                return

            file_paths = list(dst_spec.block_paths.values())
            assert len(file_paths) == len(block_ids), (
                f"block_paths count ({len(file_paths)}) != block_ids count ({len(block_ids)})"
            )

            num_blocks = len(block_ids)
            per_block_size = sum(t.shape[1] for t in tensor_infos)

            tensor_offsets = []
            offset = 0
            for t in tensor_infos:
                tensor_offsets.append(offset)
                offset += t.shape[1]

            # Batch GPU->CPU copy for all blocks into one contiguous pinned tensor
            src = torch.empty(
                (num_blocks, per_block_size),
                dtype=torch.int8,
                device="cpu",
                pin_memory=True,
            )
            for i, block_id in enumerate(block_ids):
                for j, tensor in enumerate(tensor_infos):
                    s = tensor_offsets[j]
                    e = s + tensor.shape[1]
                    src[i, s:e].copy_(tensor[int(block_id)])

            # Write files sequentially (avoid re-submitting to pool - deadlock risk)
            for i in range(num_blocks):
                dir_path = os.path.dirname(file_paths[i])
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                src[i].numpy().tofile(file_paths[i])
                transfer.num_bytes += os.path.getsize(file_paths[i])

            transfer.success = True
        except Exception as e:
            logger.error("GPU->SSD transfer failed for job %d: %s", transfer.job_id, e)
            transfer.success = False
        finally:
            transfer.done_event.set()
            with self._lock:
                self._finished.append(
                    TransferResult(
                        job_id=transfer.job_id,
                        success=transfer.success,
                        transfer_size=transfer.num_bytes,
                        transfer_time=time.monotonic() - transfer.start_time,
                        transfer_type=transfer.transfer_type,
                    )
                )

    def _ssd_to_gpu_transfer(
        self,
        transfer: SSDTransfer,
        src_spec: SSDLoadStoreSpec,
        dst_spec: LoadStoreSpec,
    ) -> None:
        """Read KV data from SSD files to GPU memory."""
        try:
            tensor_infos = []
            for td in self._kv_caches.tensors:
                tensor_infos.append(td.tensor.view(torch.int8))

            file_paths = list(src_spec.block_paths.values())
            block_ids = dst_spec.block_ids
            if len(block_ids) == 0:
                transfer.success = True
                transfer.done_event.set()
                return

            assert len(file_paths) == len(block_ids), (
                f"block_paths count ({len(file_paths)}) != block_ids count ({len(block_ids)})"
            )

            tensor_offsets = []
            offset = 0
            for t in tensor_infos:
                tensor_offsets.append(offset)
                offset += t.shape[1]

            per_block_size = offset

            # Pre-allocate pinned buffer for reads
            buf = torch.empty(
                per_block_size,
                dtype=torch.int8,
                device="cpu",
                pin_memory=True,
            )

            for i, block_id in enumerate(block_ids):
                file_path = file_paths[i]
                if not os.path.exists(file_path):
                    logger.warning("SSD block file not found: %s", file_path)
                    transfer.success = False
                    transfer.done_event.set()
                    return

                data = torch.frombuffer(memoryview(open(file_path, "rb").read()), dtype=torch.int8)
                file_size = len(data)
                transfer.num_bytes += file_size

                # Split and copy to GPU
                for j, tensor in enumerate(tensor_infos):
                    s = tensor_offsets[j]
                    e = s + tensor.shape[1]
                    tensor[int(block_id)].copy_(data[s:e].cuda())

            transfer.success = True
        except Exception as e:
            logger.error("SSD->GPU transfer failed for job %d: %s", transfer.job_id, e)
            transfer.success = False
        finally:
            transfer.done_event.set()
            with self._lock:
                self._finished.append(
                    TransferResult(
                        job_id=transfer.job_id,
                        success=transfer.success,
                        transfer_size=transfer.num_bytes,
                        transfer_time=time.monotonic() - transfer.start_time,
                        transfer_type=transfer.transfer_type,
                    )
                )

    def get_finished(self) -> list[TransferResult]:
        """Return newly finished transfers."""
        with self._lock:
            results = list(self._finished)
            self._finished.clear()
        return results

    def wait(self, job_ids: set[int]) -> None:
        """Block until specified jobs finish."""
        for job_id in job_ids:
            transfer = self._all_transfers.get(job_id)
            if transfer:
                transfer.done_event.wait()


class SSDGPUOffloadingHandler:
    """
    Bidirectional SSD offloading handler.

    Provides handlers for:
    - GPU -> SSD (store KV cache to disk)
    - SSD -> GPU (load KV cache from disk)
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        ssd_cache_dir: str,
        io_threads: int = 4,
    ) -> None:
        self._gpu_to_ssd = SSDDirectionHandler(
            kv_caches=kv_caches,
            ssd_cache_dir=ssd_cache_dir,
            io_threads=io_threads,
            gpu_to_ssd=True,
        )
        self._ssd_to_gpu = SSDDirectionHandler(
            kv_caches=kv_caches,
            ssd_cache_dir=ssd_cache_dir,
            io_threads=io_threads,
            gpu_to_ssd=False,
        )

    @property
    def gpu_to_ssd_handler(self) -> OffloadingHandler:
        return self._gpu_to_ssd

    @property
    def ssd_to_gpu_handler(self) -> OffloadingHandler:
        return self._ssd_to_gpu
