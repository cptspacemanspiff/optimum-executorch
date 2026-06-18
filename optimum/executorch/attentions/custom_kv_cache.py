# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Tuple, Union

import torch


# If transformers is not installed, raise an ImportError
try:
    # NOTE: As of transformers v5 (PR #43168), the standalone HybridCache class was removed.
    # StaticCache now inspects the model config's `layer_types` and builds the appropriate
    # per-layer cache (full-attention vs. sliding-window/chunked) itself, so it is the
    # replacement for both the old StaticCache and HybridCache.
    from transformers.cache_utils import StaticCache
except ImportError:
    raise ImportError("transformers is not installed. Please install it to use Static/HybridCache.")

try:
    from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
        CustomKVCache,
        CustomRingKVCache,
    )
except ImportError:
    raise ImportError("ExecutorTorch is not installed. Please install it to use Custom Cache.")


def _resolve_cache_position(cache, layer_idx, key_states, cache_kwargs):
    """Resolve the per-token cache positions for a custom KV cache update.

    In transformers < v5, the model passed `cache_position` through `cache_kwargs`. Since the v5
    cache refactor, the model calls `update(key_states, value_states, layer_idx)` with no
    `cache_kwargs`; instead each cache layer derives its positions from its own `cumulative_length`,
    which the ExecuTorch export wrapper (`TorchExportableModuleWith{Static,Hybrid}Cache`) seeds with
    the start position before every forward. We reproduce the same `arange(q_len) + start` tensor here
    so the underlying ExecuTorch CustomKVCache/CustomRingKVCache still receives an `input_pos`.
    """
    if cache_kwargs is not None and cache_kwargs.get("cache_position") is not None:
        return cache_kwargs["cache_position"]
    # transformers v5 path: positions come from the HF layer's cumulative_length.
    layer = cache.layers[layer_idx]
    return torch.arange(key_states.shape[-2], device=key_states.device) + layer.cumulative_length


class ETCustomStaticCache(StaticCache):
    """
    Custom KV Cache implementation for ExecutorTorch that inherits from Hugging Face's StaticCache
    but uses custom operations for cache updates similar to ExecutorTorch's CustomStaticCache.
    """

    def __init__(
        self,
        config,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        num_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
        self.early_initialization(
            batch_size=max_batch_size, num_heads=num_heads, head_dim=head_dim, dtype=dtype, device=device
        )
        # NOTE: head_dim was split into k_head_dim/v_head_dim on the cache layers in transformers v5.

        # Validate device - handle both string and torch.device types
        if device is not None:
            device_type = (
                device if isinstance(device, str) else (device.type if isinstance(device, torch.device) else None)
            )
            # Extract just the device type (e.g., "cuda:0" -> "cuda")
            if isinstance(device_type, str):
                device_type = device_type.split(":")[0]
            assert device_type in [
                "cpu",
                "cuda",
                "mps",
            ], f"Device must be None or one of 'cpu', 'cuda', 'mps' (with optional index like 'cuda:0'), got {device}"

        # Create a list of CustomKVCache instances derived from each layer of the original Transformers cache, one per layer.
        self.kv_cache = torch.nn.ModuleList()
        for layer in self.layers:
            layer_cache = CustomKVCache(
                max_batch_size=layer.max_batch_size,
                max_context_length=layer.max_cache_len,
                n_heads=layer.num_heads,
                head_dim=layer.k_head_dim,
                dtype=dtype,
            )
            layer_cache.k_cache = layer_cache.k_cache.to(device)
            layer_cache.v_cache = layer_cache.v_cache.to(device)
            self.kv_cache.append(layer_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`
        using ExecutorTorch's CustomKVCache.

        Args:
            key_states (`torch.Tensor`):
                The new key states to cache. Shape: [batch_size, n_heads, seq_len, head_dim]
            value_states (`torch.Tensor`):
                The new value states to cache. Shape: [batch_size, n_heads, seq_len, head_dim]
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache update.

        Returns:
            A tuple containing the updated key and value states.
        """
        # In transformers v5 the model no longer threads `cache_position` via `cache_kwargs`;
        # derive it from the layer's cumulative_length (seeded by the export wrapper) instead.
        cache_position = _resolve_cache_position(self, layer_idx, key_states, cache_kwargs)
        torch._assert(cache_position is not None, "cache_position must be provided")

        # Get the CustomKVCache instance for this layer
        layer_cache = self.kv_cache[layer_idx]

        # Use the CustomKVCache's update method
        # CustomKVCache expects input_pos, k_val, v_val and handles the transpose internally
        k_out, v_out = layer_cache.update(
            input_pos=cache_position,
            k_val=key_states,
            v_val=value_states,
        )

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Return the current cache position (where this forward's first query token sits).

        Transformers v5 derives the query's ``cache_position``/``position_ids`` from this value
        whenever the caller does not pass ``cache_position`` explicitly -- which the ExecuTorch
        export wrapper (``TorchExportableModuleWithStaticCache.forward``) does not. Instead the
        wrapper seeds each ``StaticLayer.cumulative_length`` from the incoming ``cache_position[0]``
        before every forward, so we must report that (the base ``StaticLayer`` contract), and NOT a
        count of populated KV slots.

        A populated-slot count (``k_cache[..].any().sum()``) is persistent buffer state that only
        ever grows and is never reset between top-level forwards. Returning it overrides the seeded
        position, so the custom_sdpa causal/read window (``start_pos = position_ids[0][0]``) attends
        to stale slots on any rewound or repeated forward -- e.g. a second ``generate()`` call reads
        the previous call's K/V and produces garbage. The write position is unaffected (it follows
        ``cumulative_length`` directly), so writes and reads disagree, which is the actual defect.
        See ``transformers.cache_utils.StaticCache.get_seq_length``.
        """
        if layer_idx is None:
            layer_idx = 0
        return self.layers[layer_idx].get_seq_length()

    @classmethod
    def from_legacy_cache(
        cls,
        config,
        legacy_cache,
        max_cache_len=None,
        device=None,
        dtype=None,
    ):
        """
        Create an ETCustomStaticCache from a legacy cache implementation.

        Args:
            config: The model configuration
            legacy_cache: The legacy cache implementation
            max_cache_len: The maximum cache length
            device: The device for the new cache
            dtype: The data type for the new cache

        Returns:
            A new ETCustomStaticCache instance
        """
        assert hasattr(legacy_cache, "k_cache") and hasattr(legacy_cache, "v_cache")
        # Extract dimensions from the legacy cache
        assert len(legacy_cache.k_cache.shape) == 4
        if legacy_cache.k_cache.shape[1] == legacy_cache.n_heads:
            # Shape is [batch_size, n_heads, seq_len, head_dim]
            max_batch_size = legacy_cache.k_cache.shape[0]
        else:
            # Shape is [batch_size, seq_len, n_heads, head_dim]
            max_batch_size = legacy_cache.k_cache.shape[0]

        # Use the legacy cache's device and dtype if not specified
        if device is None and hasattr(legacy_cache, "device"):
            device = legacy_cache.device
        elif device is None and hasattr(legacy_cache.k_cache, "device"):
            device = legacy_cache.k_cache.device

        if dtype is None and hasattr(legacy_cache, "dtype"):
            dtype = legacy_cache.dtype
        elif dtype is None and hasattr(legacy_cache.k_cache, "dtype"):
            dtype = legacy_cache.k_cache.dtype

        # assert device is None or device == "cpu"
        assert dtype is None or dtype == torch.float32

        # Use the legacy cache's max_seq_len if max_cache_len is not specified
        if max_cache_len is None and hasattr(legacy_cache, "max_seq_len"):
            max_cache_len = legacy_cache.max_seq_len
        elif max_cache_len is None and hasattr(legacy_cache, "max_cache_len"):
            max_cache_len = legacy_cache.max_cache_len

        return cls(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )


# In transformers v5, HybridCache was folded into StaticCache (which now builds a mix of
# full-attention and sliding-window layers based on the model config), so we inherit from it.
class ETCustomHybridCache(StaticCache):
    """
    Custom Hybrid KV Cache implementation for ExecutorTorch that inherits from Hugging Face's StaticCache
    but uses ExecutorTorch's CustomKVCache for global layers and CustomRingKVCache for sliding window layers.
    """

    def __init__(
        self,
        config,
        max_batch_size: int,
        max_cache_len: Optional[int] = None,
        device: Union[torch.device, str, None] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            config=config,
            max_batch_size=max_batch_size,
            max_cache_len=max_cache_len,
            device=device,
            dtype=dtype,
        )
        num_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.early_initialization(
            batch_size=max_batch_size, num_heads=num_heads, head_dim=head_dim, dtype=dtype, device=device
        )

        # Validate device - handle both string and torch.device types
        if device is not None:
            device_type = (
                device if isinstance(device, str) else (device.type if isinstance(device, torch.device) else None)
            )
            # Extract just the device type (e.g., "cuda:0" -> "cuda")
            if isinstance(device_type, str):
                device_type = device_type.split(":")[0]
            assert device_type in [
                "cpu",
                "cuda",
                "mps",
            ], f"Device must be None or one of 'cpu', 'cuda', 'mps' (with optional index like 'cuda:0'), got {device}"

        self.cache_position = None
        # Create a list of cache instances, one per layer.
        # Use CustomKVCache for global layers and CustomRingKVCache for sliding window layers.
        self.kv_cache = torch.nn.ModuleList()
        for layer in self.layers:
            if layer.is_sliding:
                # This is a sliding window layer
                layer_cache = CustomRingKVCache(
                    max_batch_size=layer.max_batch_size,
                    max_context_length=layer.max_cache_len,
                    n_heads=layer.num_heads,
                    head_dim=layer.k_head_dim,
                    dtype=dtype,
                )
            else:
                layer_cache = CustomKVCache(
                    max_batch_size=layer.max_batch_size,
                    max_context_length=layer.max_cache_len,
                    n_heads=layer.num_heads,
                    head_dim=layer.k_head_dim,
                    dtype=dtype,
                )
                layer_cache.k_cache = layer_cache.k_cache.to(device)
                layer_cache.v_cache = layer_cache.v_cache.to(device)
            self.kv_cache.append(layer_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`
        using ExecutorTorch's CustomKVCache or CustomRingKVCache depending on the layer type.

        Args:
            key_states (`torch.Tensor`):
                The new key states to cache. Shape: [batch_size, n_heads, seq_len, head_dim]
            value_states (`torch.Tensor`):
                The new value states to cache. Shape: [batch_size, n_heads, seq_len, head_dim]
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache update.

        Returns:
            A tuple containing the updated key and value states.
        """
        # In transformers v5 the model no longer threads `cache_position` via `cache_kwargs`;
        # derive it from the layer's cumulative_length (seeded by the export wrapper) instead.
        cache_position = _resolve_cache_position(self, layer_idx, key_states, cache_kwargs)
        assert isinstance(cache_position, torch.Tensor)
        self.cache_position = cache_position

        # Get the cache instance for this layer (either CustomKVCache or CustomRingKVCache)
        layer_cache = self.kv_cache[layer_idx]

        # Use the cache's update method
        # Both CustomKVCache and CustomRingKVCache have the same update interface
        k_out, v_out = layer_cache.update(
            input_pos=cache_position,
            k_val=key_states,
            v_val=value_states,
        )

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if layer_idx is None:
            layer_idx = 0

        # For CustomRingKVCache, we need to handle the sequence length differently
        layer_cache = self.kv_cache[layer_idx]
        if self.layers[layer_idx].is_sliding:
            # CustomRingKVCache cache_position_manager which
            # maintains cache position for each slot in the kv cache
            # we return the max position + 1 to indicate max position
            # seen so far. Not sure if thats the correct interpretation
            # of sequence length
            return layer_cache.cache_positions_manager.cache_positions.max().item() + 1
        return (layer_cache.k_cache[0, :, 0].any(dim=-1)).sum()

    def get_layer_cache(self, layer_idx: int):
        """
        Get the cache for a specific layer. This method is dynamo-traceable.

        Args:
            layer_idx (int): The layer index

        Returns:
            The cache instance for the specified layer (CustomKVCache or CustomRingKVCache)
        """
        return self.kv_cache[layer_idx]


def replace_with_et_custom_kv_cache(module, config, generation_config, cache_dtype):
    """
    Replace all KV caches in the module with ETCustomStaticCache or ETCustomHybridCache.
    This modifies the model in place.

    Args:
        module: The module to modify
        config: The model configuration

    Returns:
        The modified module
    """
    # Recursively replace KV caches
    return _replace_with_et_custom_kv_cache(module, config, generation_config, cache_dtype)


def _replace_with_et_custom_kv_cache(module, config, generation_config, cache_dtype):
    """
    Helper function to recursively replace KV caches in the module.

    Args:
        module: The module to modify
        config: The model configuration

    Returns:
        The modified module
    """
    # Check if module has static_cache (TorchExportableModuleWithStaticCache)
    if hasattr(module, "static_cache"):
        assert isinstance(module.static_cache, StaticCache), f"Expected StaticCache, got {type(module.static_cache)}"

        # TODO: Add replace_cache to exported module
        # in transformer's executorch.py
        if getattr(module, "replace_cache", None) is not None:
            static_cache = ETCustomStaticCache(
                config=config,
                max_batch_size=generation_config.cache_config.get("batch_size"),
                max_cache_len=generation_config.cache_config.get("max_cache_len"),
                device=generation_config.cache_config.get("device"),
                dtype=cache_dtype,
            )
            module.replace_cache(static_cache)
        else:
            module.static_cache = ETCustomStaticCache(
                config=config,
                max_batch_size=generation_config.cache_config.get("batch_size"),
                max_cache_len=generation_config.cache_config.get("max_cache_len"),
                device=generation_config.cache_config.get("device"),
                dtype=cache_dtype,
            )
            # Dont know why we need to this even though
            # CustomKVCache registers the attributes
            for i in range(len(module.static_cache.kv_cache)):
                setattr(module, f"key_cache_{i}", module.static_cache.kv_cache[i].k_cache)
                setattr(module, f"value_cache_{i}", module.static_cache.kv_cache[i].v_cache)
            # The export wrapper registered `cumulative_length_{i}` buffers pointing at the
            # *original* StaticCache layers and mutates them in-place during forward (and we read
            # them in `update` to derive cache positions). Now that we've swapped in a new cache,
            # re-register those buffers against the new layers' tensors, otherwise they are seen as
            # mutated constants and `run_decompositions` rejects the program.
            for i, layer in enumerate(module.static_cache.layers):
                module.register_buffer(f"cumulative_length_{i}", layer.cumulative_length, persistent=False)

    # Check if module has cache (TorchExportableModuleWithHybridCache)
    elif hasattr(module, "cache"):
        # Replace with ETCustomHybridCache
        if getattr(module, "replace_cache", None) is not None:
            hybrid_cache = ETCustomHybridCache(
                config=config,
                max_batch_size=generation_config.cache_config.get("batch_size"),
                max_cache_len=generation_config.cache_config.get("max_cache_len"),
                device=generation_config.cache_config.get("device"),
                dtype=cache_dtype,
            )
            module.replace_cache(hybrid_cache)
        else:
            module.cache = ETCustomHybridCache(
                config=config,
                max_batch_size=generation_config.cache_config.get("batch_size"),
                max_cache_len=generation_config.cache_config.get("max_cache_len"),
                device=generation_config.cache_config.get("device"),
                dtype=cache_dtype,
            )
            # Register cache attributes for each layer
            for i in range(len(module.cache.kv_cache)):
                setattr(module, f"key_cache_{i}", module.cache.kv_cache[i].k_cache)
                setattr(module, f"value_cache_{i}", module.cache.kv_cache[i].v_cache)
                # Re-register the cumulative_length buffers against the new cache's layers (see the
                # static_cache branch above for the rationale) so their in-place mutation stays legal.
                module.register_buffer(
                    f"cumulative_length_{i}", module.cache.layers[i].cumulative_length, persistent=False
                )
                if module.cache.layers[i].is_sliding:
                    # Register cache_positions as buffer for sliding window layers
                    # This prevents it from being traced as a constant
                    module.register_buffer(
                        f"cache_positions_{i}",
                        module.cache.kv_cache[i].cache_positions_manager.cache_positions,
                        persistent=False,
                    )
    else:
        raise ValueError(
            "Module must have either 'static_cache' (TorchExportableModuleWithStaticCache) "
            "or 'cache' (TorchExportableModuleWithHybridCache) attribute"
        )

    return module
