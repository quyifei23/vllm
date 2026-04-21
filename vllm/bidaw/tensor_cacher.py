# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""存储高效张量缓存（论文 §4）

职责：
- 在 Attention 层的 forward() 中 hook 住 normalized activation（tensor 6）
- 缓存 normalized activation 而非 KV tensor（节省空间）
- 加载时在低优先级 CUDA stream 上执行 W_K、W_V 矩阵乘法重建 KV

MHA vs GQA 判断：
- MHA（OPT、LLaMA、Qwen 等）：使用 tensor 6 缓存
- GQA（num_kv_heads < num_attention_heads）：KV 本身很小，直接缓存 KV
- 自动从 ModelConfig 检测：num_kv_heads == num_attention_heads 即为 MHA
"""


class TensorCacher:
    """截获并缓存 normalized activation（tensor 6）。

    MHA 模型中，KV 可以通过 normalized activation 乘以 W_K、W_V 重建，
    缓存 tensor 6 比缓存 KV 本身节省空间。

    GQA 模型中 KV 本身已很小，直接缓存 KV（由调用方判断，本组件
    只负责截获和缓存）。

    加载时在低优先级 CUDA stream 上执行 W_K、W_V 矩阵乘法重建 KV，
    不阻塞主推理流。
    """

    def maybe_cache_tensor6(self, layer_idx, request_ids, tensor6):
        """在 Attention forward() 中截获 tensor 6，根据当前状态
        决定是否缓存。

        Args:
            layer_idx: Attention 层索引
            request_ids: 当前 batch 中的 request_id 列表
            tensor6: normalized activation tensor
        """
        pass

    def load_tensor6(self, layer_idx, request_id):
        """从缓存中加载指定层的 tensor 6。

        Args:
            layer_idx: Attention 层索引
            request_id: 请求 ID

        Returns:
            normalized activation tensor，用于在低优先级 stream 上重建 KV。
        """
        pass

    def rebuild_kv(self, tensor6, k_proj, v_proj):
        """使用 W_K、W_V 投影矩阵从 normalized activation 重建 KV。

        在低优先级 CUDA stream 上执行，不阻塞主推理流。

        Args:
            tensor6: normalized activation tensor
            k_proj: K 投影矩阵
            v_proj: V 投影矩阵

        Returns:
            重建的 K 和 V 张量。
        """
        pass


def is_mha_model(model_config):
    """自动检测模型是否为 MHA 架构。

    通过比较 num_kv_heads 和 num_attention_heads 判断：
    - num_kv_heads == num_attention_heads → MHA
    - num_kv_heads < num_attention_heads → GQA

    Args:
        model_config: vLLM ModelConfig 对象

    Returns:
        bool: True 表示 MHA，False 表示 GQA。
    """
    pass
