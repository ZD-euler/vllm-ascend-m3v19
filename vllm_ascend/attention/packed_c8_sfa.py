# SPDX-License-Identifier: Apache-2.0
"""Call the torch_npu sparse attention operator for the M3 packed main KV."""

from __future__ import annotations

import torch

_PAGE_TOKENS = 128
_NOPE_DIM = 512
_ROPE_DIM = 64
_QUANT_TILE_SIZE = 128
_PACKED_ROW_BYTES = _NOPE_DIM + 2 * _ROPE_DIM + 4 * (_NOPE_DIM // _QUANT_TILE_SIZE)
_PER_TILE_QUANT_MODE = 2
_MLA_ABSORB_MODE = 2
_COMBINED_SCALE_REPO_MODE = 1
_CAUSAL_SPARSE_MODE = 3


def packed_c8_sparse_flash_attention(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    packed_main: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    scale: float,
    block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
) -> torch.Tensor:
    """Run TND attention over contiguous PA_BSND packed main pages.

    The caller supplies the validated main cache with INT8 NOPE, BF16 RoPE,
    and inline FP32 scales. Query lengths are cumulative TND end offsets.
    The block table, KV lengths, and topK must refer to the same request order
    and cache layout. This adapter validates tensor metadata without reading
    device values. torch_npu validates the supported topK shape.
    """
    if (
        packed_main.ndim != 4
        or packed_main.shape[1:] != (_PAGE_TOKENS, 1, _PACKED_ROW_BYTES)
        or packed_main.dtype != torch.int8
        or not packed_main.is_contiguous()
    ):
        raise ValueError("Packed C8 main KV must be contiguous INT8 [blocks, 128, 1, 656]")
    if (
        ql_nope.ndim != 3
        or q_pe.ndim != 3
        or ql_nope.shape[:-1] != q_pe.shape[:-1]
        or ql_nope.shape[-1] != _NOPE_DIM
        or q_pe.shape[-1] != _ROPE_DIM
        or ql_nope.dtype != torch.bfloat16
        or q_pe.dtype != torch.bfloat16
    ):
        raise ValueError("Packed C8 attention requires matching BF16 TND NOPE (512) and RoPE (64) queries")
    if (
        block_table.ndim != 2
        or actual_seq_lengths_query.ndim != 1
        or actual_seq_lengths_kv.ndim != 1
        or block_table.shape[0] != actual_seq_lengths_query.shape[0]
        or actual_seq_lengths_query.shape != actual_seq_lengths_kv.shape
    ):
        raise ValueError("Packed C8 block table and query/KV lengths must describe the same batch")
    if any(
        tensor.device != packed_main.device
        for tensor in (ql_nope, q_pe, topk_indices, block_table, actual_seq_lengths_query, actual_seq_lengths_kv)
    ):
        raise ValueError("Packed C8 queries, cache, and attention metadata must be on the same device")

    # Import the device backend only when this path is called.
    import torch_npu

    quant_sfa = getattr(torch_npu, "npu_kv_quant_sparse_flash_attention", None)
    if not callable(quant_sfa):
        raise RuntimeError("torch_npu does not provide npu_kv_quant_sparse_flash_attention")

    return quant_sfa(
        query=torch.cat((ql_nope, q_pe), dim=-1),
        key=packed_main,
        value=packed_main,
        sparse_indices=topk_indices,
        scale_value=scale,
        key_quant_mode=_PER_TILE_QUANT_MODE,
        value_quant_mode=_PER_TILE_QUANT_MODE,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=_CAUSAL_SPARSE_MODE,
        attention_mode=_MLA_ABSORB_MODE,
        quant_scale_repo_mode=_COMBINED_SCALE_REPO_MODE,
        tile_size=_QUANT_TILE_SIZE,
        rope_head_dim=_ROPE_DIM,
    )
