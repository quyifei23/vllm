# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""混合粒度 GPU 内存块分配（论文 §4）

职责：
- big_block（256 tokens）：用于已知长度的历史对话和 query
- small_block（16 tokens）：用于不可预测长度的 response
- 1 个 big_block 可拆分为 16 个 small_block
- 16 个空闲 small_block 可合并为 1 个 big_block（避免碎片）

与 vLLM BlockSpaceManager 的集成：
- 方案 A：继承 BlockSpaceManager，重写 allocate/free 方法（推荐）
"""


class MixedGranularityBlock:
    """表示一个物理内存块，可以是 big 或 small 粒度。

    每个块记录其类型（big/small）、分配状态、所属 request_id
    以及在 GPU 显存中的物理偏移。
    """
    pass


class MixedGranularityAllocator:
    """管理 GPU 上 big_block 和 small_block 的分配与回收。

    维护两个空闲池：free_big_blocks 和 free_small_blocks。
    当 big_block 不足时，可通过拆分 big_block 获得 small_block；
    当 small_block 碎片过多时，尝试合并为 big_block。
    """

    def allocate_big_block(self, request_id, block_type):
        """分配一个 big_block 给指定请求。

        Args:
            request_id: 请求 ID
            block_type: 块类型标识（history/query）

        Returns:
            MixedGranularityBlock 对象。
        """
        pass

    def allocate_small_block(self, request_id):
        """分配一个 small_block 给指定请求（用于 response 阶段）。

        Args:
            request_id: 请求 ID

        Returns:
            MixedGranularityBlock 对象。
        """
        pass

    def split_big_to_small(self, big_block_id):
        """将 1 个空闲 big_block 拆分为 16 个 small_block。

        Args:
            big_block_id: 要拆分的 big_block ID

        Returns:
            16 个 small_block ID 列表。
        """
        pass

    def merge_small_to_big(self, small_block_ids):
        """将 16 个连续的空闲 small_block 合并为 1 个 big_block。

        Args:
            small_block_ids: 16 个连续 small_block ID 列表

        Returns:
            合并后的 big_block ID。
        """
        pass

    def free_block(self, block_id):
        """释放指定 block，回收到相应的空闲池。

        若回收的是 small_block，检查是否有 16 个连续空闲 small_block
        可合并为 big_block。

        Args:
            block_id: 要释放的 block ID。
        """
        pass

    def get_fragmentation_ratio(self):
        """计算当前内存碎片化程度。

        Returns:
            碎片化比率（0.0 表示无碎片，1.0 表示完全碎片化）。
        """
        pass


class BidawBlockSpaceManager:
    """替换 vLLM 原有 BlockSpaceManager，支持混合粒度分配。

    继承 vLLM 的 BlockSpaceManager，重写 allocate/free 方法，
    将 token 分配请求路由到 MixedGranularityAllocator。
    """

    def allocate_for_sequence(self, seq, token_type):
        """为序列分配内存块。

        Args:
            seq: 序列对象
            token_type: token 类型（"history"/"query"/"response"），
                       决定使用 big_block 还是 small_block。
        """
        pass

    def can_allocate(self, num_tokens):
        """检查是否有足够空间分配指定数量的 token。

        Args:
            num_tokens: 需要分配的 token 数量

        Returns:
            bool: 是否可以分配。
        """
        pass
