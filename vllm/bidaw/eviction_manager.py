# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""基于前轮答案长度的 KV 驱逐策略（论文 §3.3）

职责：
- WeightedReuseDistanceTracker：追踪每个用户 KV 的 weighted reuse distance
- AnswerLengthReusePredictor：维护 answer_length → WRD 下界的统计关系
- GhostCache：在后台运行 Belady 最优策略，统计各 WRD 桶的命中率
- BidawEvictionManager：综合计算 overall hit potential，选择驱逐候选

触发时机：
- DRAM 空闲空间低于 eviction_threshold（5%）时触发

多卡适配：
- Ghost Cache 是逻辑上的，不涉及实际数据传输，TP/PP 不影响实现
"""


class WeightedReuseDistanceTracker:
    """追踪每个用户 KV 的 weighted reuse distance 历史分布。

    WRD 定义：两次访问同一用户 KV 之间，其他用户 KV 的总访问大小（字节）。
    记录每个 user_id 的 WRD 分布（直方图），用于预测该用户未来是否可能被再次访问。

    注意：ghost cache 中记录的是 User 级别的 KV Cache 访问 trace，
    同一 User 的多轮对话映射到一个 user_id。
    """

    def on_kv_accessed(self, user_id, kv_size_bytes):
        """记录一次 KV 访问事件，更新该用户的 WRD 分布。

        Args:
            user_id: 用户唯一标识
            kv_size_bytes: 本次访问的 KV 总字节数
        """
        pass

    def get_distribution(self, user_id):
        """获取指定用户 KV 的 WRD 历史分布（直方图）。

        Returns:
            WRD 分布数组，长度为 num_promising_buckets。
        """
        pass


class AnswerLengthReusePredictor:
    """维护 answer_length → WRD 下界的统计关系。

    对于每个用户，统计其历史生成的答案长度（tokens）与对应的 WRD 下界，
    以便在当前轮次生成答案后，预测下一轮次该用户 KV 的 reuse 可能性。
    """

    def on_answer_generated(self, user_id, answer_length, wrd):
        """记录一次答案生成事件，更新 answer_length → WRD 的统计。

        Args:
            user_id: 用户唯一标识
            answer_length: 本轮生成的答案长度（tokens）
            wrd: 对应的 weighted reuse distance
        """
        pass

    def predict_wrd_lower_bound(self, user_id, answer_length):
        """根据该用户历史 answer_length → WRD 统计，预测给定答案长度
        对应的 WRD 下界。

        Args:
            user_id: 用户唯一标识
            answer_length: 预测的答案长度（tokens）

        Returns:
            预测的 WRD 下界值。
        """
        pass


class GhostCache:
    """后台运行 Belady 最优策略，统计各 WRD 桶的命中率。

    在后台线程中运行，定期对历史访问 trace 运行 Belady 模拟，
    统计每个 WRD bucket 的命中率，为驱逐决策提供参考数据。

    Ghost Cache 回答的问题是："如果我只能猜哪些 KV 会被再访问，怎么猜最准？"
    """

    def run_simulation(self):
        """对历史 trace 运行 Belady 模拟，更新各 WRD 桶的命中率。
        在后台线程中调用，不阻塞主流程。
        """
        pass

    def get_hit_rates(self):
        """获取各 WRD 桶的命中率。

        Returns:
            各桶命中率数组，长度为 num_promising_buckets。
        """
        pass


class BidawEvictionManager:
    """对外暴露驱逐决策接口。

    综合 WeightedReuseDistanceTracker、AnswerLengthReusePredictor 和
    GhostCache 的结果，计算每个用户的 overall hit potential，
    选择 potential 最低的用户作为驱逐候选。

    触发条件：DRAM 空闲空间低于 eviction_threshold（5%）。
    """

    def select_eviction_candidate(self):
        """选择 overall hit potential 最低的用户作为驱逐候选。

        Returns:
            被选中驱逐的 user_id。
        """
        pass

    def trigger_eviction_if_needed(self, free_memory_ratio):
        """检查 DRAM 空闲空间，低于阈值时触发驱逐。

        Args:
            free_memory_ratio: 当前 DRAM 空闲空间占比（0.0 ~ 1.0）
        """
        pass
