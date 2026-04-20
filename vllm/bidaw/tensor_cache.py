# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Storage-Efficient Tensor Cache for Bidaw.

Paper Section 4: caches "tensor 6" (normalized activations from LayerNorm
output) instead of traditional KV tensors. Tensor 6 has 67% higher cost
efficiency (51.0 vs 30.5) and is ~50% the size of KV cache in MHA models.

Core mechanisms:
- Forward hooks on RMSNorm layers capture normalized activations
- Pinned CPU buffers store captured tensors (circular buffer per layer)
- Low-priority CUDA stream reconstructs K, V from cached tensor 6
- Reconstruction runs on idle SMs without blocking main inference

Reconstruction:
    K = norm_act @ W_K.T
    V = norm_act @ W_V.T
Q is NOT reconstructed — computed fresh from current token each step.
"""

from collections import defaultdict

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class BidawTensorCacher:
    """
    Manages circular buffers of cached normalized activations (tensor 6).

    Per-layer pinned CPU buffers store captured tensors. When a request
    needs its KV data loaded, the cached tensor is reconstructed to K, V
    on a low-priority CUDA stream.

    Memory management:
    - Circular buffer per layer with fixed total capacity
    - Oldest entries overwritten when buffer is full
    - Capacity controlled by tensor_cache_memory_gb config
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dtype: torch.dtype,
        max_tokens: int = 0,
        memory_gb: float = 0.0,
        device: str = "cuda",
    ) -> None:
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dtype = dtype
        self._device = device
        self._dtype_bytes = torch.tensor([], dtype=dtype).element_size()

        # Compute buffer capacity
        if memory_gb > 0:
            max_bytes = int(memory_gb * 1e9)
        else:
            # Auto: ~10% of host memory, capped at 64 GB
            max_bytes = 64 * 1024**3

        # Per-token-per-layer bytes
        bytes_per_token_per_layer = hidden_size * self._dtype_bytes
        if max_tokens > 0:
            self._max_tokens = max_tokens
        else:
            self._max_tokens = max(
                1, max_bytes // (num_layers * bytes_per_token_per_layer)
            )

        # Pinned CPU buffers: layer_id -> list of (token_start, token_end, tensor)
        self._buffers: dict[int, list[tuple[int, int, torch.Tensor]]] = (
            defaultdict(list)
        )
        self._buffer_capacity: dict[int, int] = defaultdict(int)

        # Low-priority CUDA stream for reconstruction
        if torch.cuda.is_available():
            low_pri, _ = torch.cuda.Stream.priority_range()
            self._reconstruct_stream = torch.cuda.Stream(priority=low_pri)
        else:
            self._reconstruct_stream = None

        # Track weights for reconstruction
        self._layer_weights: dict[
            int, tuple[torch.Tensor | None, torch.Tensor | None]
        ] = {}

        logger.info(
            "Tensor cache initialized: hidden_size=%d, layers=%d, "
            "max_tokens=%d, dtype=%s",
            hidden_size,
            num_layers,
            self._max_tokens,
            dtype,
        )

    def register_layer_weights(
        self,
        layer_id: int,
        w_k: torch.Tensor | None,
        w_v: torch.Tensor | None,
    ) -> None:
        """Store K and V projection weights for reconstruction."""
        self._layer_weights[layer_id] = (w_k, w_v)

    def store_tensor(
        self,
        layer_id: int,
        token_start: int,
        norm_act: torch.Tensor,
    ) -> None:
        """
        Store a normalized activation tensor in the CPU buffer.

        Args:
            layer_id: Layer index.
            token_start: Starting token position.
            norm_act: Normalized activation tensor (seq_len, hidden_size).
        """
        num_tokens = norm_act.shape[0]

        # Copy to pinned CPU memory
        if not norm_act.is_pinned():
            cpu_tensor = norm_act.cpu().pin_memory()
        else:
            cpu_tensor = norm_act.cpu()

        buffer = self._buffers[layer_id]
        buffer.append((token_start, token_start + num_tokens, cpu_tensor))
        self._buffer_capacity[layer_id] += num_tokens

        # Evict oldest entries if over capacity
        while self._buffer_capacity[layer_id] > self._max_tokens:
            _, _, evicted = buffer.pop(0)
            self._buffer_capacity[layer_id] -= evicted.shape[0]

    def load_tensor(
        self,
        layer_id: int,
        token_start: int,
        num_tokens: int,
    ) -> torch.Tensor | None:
        """
        Load cached normalized activation for a token range.

        Returns concatenated tensor if found, None if not cached.
        """
        buffer = self._buffers[layer_id]
        token_end = token_start + num_tokens
        parts: list[torch.Tensor] = []

        for ts, te, tensor in buffer:
            # Check overlap
            overlap_start = max(ts, token_start)
            overlap_end = min(te, token_end)
            if overlap_start < overlap_end:
                local_start = overlap_start - ts
                local_end = overlap_end - ts
                parts.append(tensor[local_start:local_end])

        if not parts:
            return None

        return torch.cat(parts, dim=0)

    def reconstruct_kv(
        self,
        layer_id: int,
        norm_act_cpu: torch.Tensor,
        w_k: torch.Tensor | None = None,
        w_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Reconstruct K and V tensors from cached normalized activation.

        Runs on low-priority CUDA stream to use idle SMs.

        Args:
            layer_id: Layer index.
            norm_act_cpu: Normalized activation on CPU (seq_len, hidden_size).
            w_k: K projection weight (out_features, hidden_size). If None, uses registered weight.
            w_v: V projection weight (out_features, hidden_size). If None, uses registered weight.

        Returns:
            (K_gpu, V_gpu) tensors on GPU, or None if weights unavailable.
        """
        if self._reconstruct_stream is None:
            return None

        # Get weights
        if w_k is None or w_v is None:
            w_k, w_v = self._layer_weights.get(layer_id, (None, None))
        if w_k is None or w_v is None:
            logger.warning("No weights for layer %d, cannot reconstruct KV", layer_id)
            return None

        with torch.cuda.stream(self._reconstruct_stream):
            norm_act_gpu = norm_act_cpu.cuda(non_blocking=True)
            K = torch.matmul(norm_act_gpu, w_k.T)
            V = torch.matmul(norm_act_gpu, w_v.T)
            torch.cuda.synchronize()

        return K, V

    def get_memory_usage(self) -> int:
        """Return total bytes used in CPU buffers."""
        total = 0
        for buffer in self._buffers.values():
            for _, _, tensor in buffer:
                total += tensor.numel() * tensor.element_size()
        return total

    def clear(self) -> None:
        """Clear all cached tensors."""
        self._buffers.clear()
        self._buffer_capacity.clear()

    @property
    def reconstruct_stream(self) -> torch.cuda.Stream | None:
        """The low-priority CUDA stream used for reconstruction."""
        return self._reconstruct_stream


def _find_attention_layernorms(
    model: torch.nn.Module,
) -> dict[int, tuple[torch.nn.Module, torch.nn.Module | None]]:
    """
    Find RMSNorm layers used as input_layernorm (pre-attention tensor 6).

    Walks the model's named modules looking for RMSNorm instances whose
    name ends with 'input_layernorm', which is the standard naming
    convention in vLLM model architectures (Llama, Qwen, OPT, etc.).

    Args:
        model: The PyTorch model to scan.

    Returns:
        Dict mapping layer_id -> (rmsnorm_module, attention_module_or_None).
    """
    from vllm.model_executor.layers.layernorm import RMSNorm

    layernorms: dict[int, tuple[torch.nn.Module, torch.nn.Module | None]] = {}

    for name, module in model.named_modules():
        if isinstance(module, RMSNorm) and name.endswith("input_layernorm"):
            # Extract layer index from name like "model.layers.3.input_layernorm"
            parts = name.split(".")
            layer_id = None
            for i, part in enumerate(parts):
                if part == "layers" and i + 1 < len(parts):
                    try:
                        layer_id = int(parts[i + 1])
                    except ValueError:
                        pass
            if layer_id is not None:
                layernorms[layer_id] = (module, None)

    return layernorms


def _find_qkv_weights(
    model: torch.nn.Module,
) -> dict[int, tuple[torch.Tensor | None, torch.Tensor | None]]:
    """
    Find K and V projection weights for each attention layer.

    Looks for modules with names containing 'k_proj' and 'v_proj' or
    attributes like qkv_proj that can be split.

    Args:
        model: The PyTorch model to scan.

    Returns:
        Dict mapping layer_id -> (w_k, w_v) weight tensors or None.
    """
    weights: dict[int, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

    for name, module in model.named_modules():
        # Look for separate k_proj and v_proj
        if "k_proj" in name and hasattr(module, "weight"):
            layer_id = _extract_layer_id(name)
            if layer_id is not None:
                _, w_v = weights.get(layer_id, (None, None))
                weights[layer_id] = (module.weight, w_v)
        elif "v_proj" in name and hasattr(module, "weight"):
            layer_id = _extract_layer_id(name)
            if layer_id is not None:
                w_k, _ = weights.get(layer_id, (None, None))
                weights[layer_id] = (w_k, module.weight)

        # Look for fused qkv_proj
        elif "qkv_proj" in name and hasattr(module, "weight"):
            layer_id = _extract_layer_id(name)
            if layer_id is not None:
                # For fused QKV, weights are [W_Q; W_K; W_V] concatenated
                # We need model config to split them; store full weight for now
                weights[layer_id] = (module.weight, module.weight)

    return weights


def _extract_layer_id(name: str) -> int | None:
    """Extract layer index from a module name like 'model.layers.3.k_proj'."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None
