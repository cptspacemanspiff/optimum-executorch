# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Export-friendly cross-attention for T5 that reuses the encoder-derived K/V across decode
# steps instead of recomputing the k/v projections every step.
#
# This is modeled on the idea behind the Whisper version
# (optimum/executorch/attentions/whisper_attention.py) -- a `torch.cond` "compute-once-then-
# reuse" branch -- but it deliberately uses ONLY standard ATen ops, because the Whisper path
# relies on the `executorch.{alias,update_cross_attn_cache}` custom ops, which have no portable
# out-variant (they are only registered with a fake/meta kernel + an Inductor lowering, i.e. they
# lower for CUDA/AOTInductor but NOT for the XNNPACK/portable `.pte` path). Hence WhisperCrossAttention
# is CUDA-gated.
#
# Mechanism (all export/XNNPACK-friendly):
#   * This module OWNS the cross-KV cache as its own registered buffers, so writing them is an
#     ordinary mutable-buffer mutation in the MAIN graph -- the same mechanism the self-attention
#     StaticCache already uses -- rather than an in-`torch.cond` mutation (which export forbids).
#   * `torch.cond(cache_initialized, reuse, recompute)` selects between cloning the cached K/V and
#     recomputing them via the k/v projections (as `matmul`, because the XNNPACK partitioner cannot
#     delegate -- and refuses to decompose -- `aten.linear` inside a cond subgraph). The recompute
#     branch only runs on the first decode step.
#   * Both branches return true-length (S = encoder length) K/V, so no padding mask is needed; the
#     buffers are sized to `cross_cache_len` (the max source length) and only their [0:S] slice is
#     used. The persist `copy_` is a dynamic-length slice write, mirroring the self-attention cache.

from typing import Optional

import torch
from torch import Tensor, nn


class T5CrossAttention(nn.Module):
    """Drop-in replacement for a decoder block's ``EncDecAttention`` (a ``T5Attention``)."""

    def __init__(self, t5_attention: nn.Module, layer_idx: int, cross_cache_len: int, dtype, device):
        super().__init__()
        # Reuse the original projection weights verbatim (all bias-free in T5).
        self.q = t5_attention.q
        self.k = t5_attention.k
        self.v = t5_attention.v
        self.o = t5_attention.o
        self.n_heads = t5_attention.n_heads
        self.key_value_proj_dim = t5_attention.key_value_proj_dim
        self.inner_dim = t5_attention.inner_dim
        self.layer_idx = layer_idx

        # Static cross-KV cache (this module owns it; written in the main graph). Sized to the max
        # source length; only [0:S] is used at runtime for an encoder of length S <= cross_cache_len.
        cache_shape = (1, self.n_heads, cross_cache_len, self.key_value_proj_dim)
        self.register_buffer("cross_k", torch.zeros(cache_shape, dtype=dtype, device=device), persistent=False)
        self.register_buffer("cross_v", torch.zeros(cache_shape, dtype=dtype, device=device), persistent=False)
        # The decode step index, stashed by the decoder wrapper each forward (transformers does not
        # thread cache_position into cross-attention). `cache_position[0] == 0` is the "first step"
        # predicate -- read from an input each call, so unlike a persisted boolean flag it does NOT
        # depend on a mutable buffer's (unpreserved) initial state under ExecuTorch.
        self.cache_position = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        **kwargs,
    ):
        bsz = hidden_states.shape[0]
        q_len = hidden_states.shape[1]
        s = key_value_states.shape[1]  # true encoder length (dynamic)

        # Query projection. T5 applies NO scaling (it is folded into initialization).
        query_states = self.q(hidden_states).view(bsz, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def reuse(cross_k: Tensor, cross_v: Tensor, kv_states: Tensor):
            # Reuse the cached cross K/V. Slice to the true encoder length and clone (torch.cond
            # branches may not return their operands directly).
            return cross_k[:, :, :s, :].clone(), cross_v[:, :, :s, :].clone()

        def recompute(cross_k: Tensor, cross_v: Tensor, kv_states: Tensor):
            # Compute fresh K/V. Use matmul against the (bias-free) weights rather than nn.Linear:
            # the XNNPACK partitioner can neither delegate nor decompose `aten.linear` inside a
            # torch.cond subgraph. Runs only on the first decode step.
            k = torch.matmul(kv_states, self.k.weight.t())
            v = torch.matmul(kv_states, self.v.weight.t())
            k = k.view(bsz, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2).contiguous()
            v = v.view(bsz, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2).contiguous()
            return k, v

        is_first = self.cache_position.reshape(-1)[0] == 0
        key_states, value_states = torch.cond(
            is_first,
            recompute,
            reuse,
            operands=(self.cross_k, self.cross_v, key_value_states),
        )

        # Persist into the owned cache buffers (main-graph mutable-buffer write). On reuse steps this
        # writes the just-read values back (idempotent); on the first step it stores the fresh K/V.
        self.cross_k[:, :, :s, :].copy_(key_states)
        self.cross_v[:, :, :s, :].copy_(value_states)

        # scores: [B, H, q_len, S]. No scaling (T5); no mask (K/V are true-length, no padding).
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_output = torch.matmul(attn_weights, value_states)  # [B, H, q_len, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.inner_dim)
        attn_output = self.o(attn_output)

        # T5 threads a (cross) position_bias between layers; cross-attention has no relative bias,
        # so a zeros tensor of the right shape keeps the plumbing happy.
        out_position_bias = scores.new_zeros((1, self.n_heads, q_len, s))
        outputs = (attn_output, out_position_bias)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
