# SPDX-License-Identifier: Apache-2.0
"""Packed main C8 dispatches SFA to the quantized torch_npu adapter."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm_ascend.attention.sfa_v1 import AscendSFAImpl


def _inputs():
    packed = torch.zeros((2, 128, 1, 656), dtype=torch.int8)
    ql_nope = torch.zeros((2, 1, 512), dtype=torch.bfloat16)
    q_pe = torch.zeros((2, 1, 64), dtype=torch.bfloat16)
    topk = torch.zeros((2, 1, 8), dtype=torch.int32)
    block_table = torch.zeros((2, 4), dtype=torch.int32)
    query_lens = torch.tensor([2], dtype=torch.int32)
    key_lens = torch.tensor([256], dtype=torch.int32)
    return packed, ql_nope, q_pe, topk, block_table, query_lens, key_lens


def test_packed_main_attention_uses_quantized_adapter():
    packed, ql_nope, q_pe, topk, block_table, query_lens, key_lens = _inputs()
    impl = SimpleNamespace(
        _packed_main_cache=packed,
        scale=0.125,
        dsa_shrink_latent=0,
    )
    sentinel = object()
    with patch(
        "vllm_ascend.attention.sfa_v1.packed_c8_sparse_flash_attention",
        return_value=sentinel,
    ) as adapter:
        out = AscendSFAImpl._execute_sparse_flash_attention_process(
            impl,
            ql_nope,
            q_pe,
            (packed,),
            topk,
            None,
            query_lens,
            key_lens,
            block_table_override=block_table,
        )
    assert out is sentinel
    kwargs = adapter.call_args.kwargs
    assert adapter.call_args.args == (ql_nope, q_pe, packed, topk)
    assert kwargs["scale"] == 0.125
    assert kwargs["block_table"] is block_table
    assert kwargs["actual_seq_lengths_query"] is query_lens
    assert kwargs["actual_seq_lengths_kv"] is key_lens


def test_non_packed_attention_keeps_legacy_kernel():
    packed, ql_nope, q_pe, topk, block_table, query_lens, key_lens = _inputs()
    impl = SimpleNamespace(_packed_main_cache=None, scale=0.125, dsa_shrink_latent=0)
    sentinel = object()
    with patch(
        "torch.ops._C_ascend.npu_sparse_flash_attention",
        create=True,
        return_value=sentinel,
    ) as kernel:
        out = AscendSFAImpl._execute_sparse_flash_attention_process(
            impl,
            ql_nope,
            q_pe,
            (packed, packed),
            topk,
            None,
            query_lens,
            key_lens,
            block_table_override=block_table,
        )
    assert out is sentinel
    assert kernel.call_args.kwargs["key"] is packed
