# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SSD LoadStoreSpec for Bidaw three-tier storage.

Defines the medium type and metadata for SSD-backed KV cache blocks.
"""

from dataclasses import dataclass, field

from vllm.v1.kv_offload.abstract import LoadStoreSpec


@dataclass
class SSDLoadStoreSpec(LoadStoreSpec):
    """
    Metadata for loading/storing KV cache blocks on SSD.

    Contains file paths and offset information for each block
    that needs to be transferred between GPU and SSD.
    """

    # Base directory for SSD cache files
    cache_dir: str = "/tmp/bidaw_kv_cache"
    # Mapping from block_hash to file path on SSD
    block_paths: dict[str, str] = field(default_factory=dict)
    # Number of I/O threads to use
    io_threads: int = 4

    @staticmethod
    def medium() -> str:
        return "ssd"

    def get_file_path(self, block_hash: str) -> str:
        """Get the SSD file path for a given block hash."""
        return self.block_paths.get(block_hash, "")

    def add_block_path(self, block_hash: str, file_path: str) -> None:
        """Register a block hash to its SSD file path."""
        self.block_paths[block_hash] = file_path
