# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""I/O 感知请求调度器（论文 §3.2）

职责：
- 双队列机制：ready_queue（KV 在 DRAM）+ preparing_queue（KV 在 SSD）
- ready_queue 按 FCFS 调度（按原始到达时间插入）
- preparing_queue 按 disk-HRRN 决定加载顺序
- preparing_queue 中的请求加载完成后，提升到 ready_queue

与 v1 Scheduler 的关系：
- 继承 v1 的 Scheduler 类
- 在 schedule() 的 Phase 2（调度 waiting 请求）中注入双队列逻辑
- preparing_queue 语义与 v1 的 WAITING_FOR_REMOTE_KVS 状态重合

多卡适配：
- TP：调度决策在 rank 0 上做，disk-HRRN 中的 kv_size 使用总 KV 大小
- PP：调度器在第一个 pipeline stage 上运行，kv_size 需聚合所有 stage
"""

from vllm.v1.core.sched.scheduler import Scheduler


class BidawScheduler(Scheduler):
    """继承 v1 Scheduler，重写调度逻辑以支持双队列 + disk-HRRN。

    ready_queue: FCFS 队列，按原始到达时间排序。
    preparing_queue: 优先级队列，按 disk-HRRN Response Ratio 动态计算优先级。
    """

    def _promote_ready_requests(self):
        """检查 preparing_queue 中哪些请求的 KV 已从 SSD 加载到 DRAM，
        将加载完成的请求按原始 arrival_time 插入到 ready_queue 的正确位置。
        """
        pass

    def _compute_disk_hrrn_priority(self, request):
        """计算 disk-HRRN Response Ratio：
        ratio = 1 + waiting_time / kv_size_bytes

        - waiting_time: 请求在 preparing_queue 中等待的秒数
        - kv_size_bytes: 请求的 KV 总字节数（从 ModelConfig 推导：
          token_kv_size × sequence_length）
        - 第一轮对话（kv_size=0）：直接返回最高优先级

        每次 schedule() 调用重新计算，waiting_time 随时间增长。
        """
        pass

    def _select_waiting_request(self):
        """决定从哪个队列取下一个请求：
        1. 优先检查 ready_queue，按 FCFS 取第一个请求
        2. 若 ready_queue 为空，检查 preparing_queue 中是否有可加载的请求
           （按 disk-HRRN ratio 选择最高优先级的）
        3. 若两者皆空，返回 None

        当 enable_io_aware_scheduling=False 时，退化为父类的原始调度逻辑。
        """
        pass


class BidawKVStorageTracker:
    """追踪所有用户 KV 的存储位置（GPU/DRAM/SSD）。

    维护 user_id → KVStorageStatus 映射，记录每个用户的 KV 当前
    位于哪一层存储（HBM、DRAM 或 SSD），以及 KV 总大小。

    在三级存储流程中，每次 swap_blocks 或 SSD 文件 I/O 完成后
    需调用 update_storage_status 更新状态。
    """

    def update_storage_status(self, user_id, storage_layer, kv_size_bytes):
        """更新指定 user_id 的 KV 存储状态。

        Args:
            user_id: 用户唯一标识
            storage_layer: 存储层级，"gpu" | "host" | "ssd" | "none"
            kv_size_bytes: 该用户 KV 的总字节数
        """
        pass

    def get_storage_status(self, user_id):
        """获取指定 user_id 的 KV 存储状态。

        Returns:
            KVStorageStatus 对象，包含 storage_layer 和 kv_size_bytes。
        """
        pass
