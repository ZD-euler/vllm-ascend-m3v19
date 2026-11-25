# SPDX-License-Identifier: Apache-2.0
"""Check the real mask JIT cache after startup, without distributed services."""

import pytest
import torch

pytest.importorskip("torch_npu")
pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU")


def forbid_compile(*args, **kwargs):
    raise AssertionError("DCP mask requested compilation after startup warmup")


@torch.inference_mode()
def test_changing_batches_and_padding_reuse_warmed_mask(monkeypatch):
    from vllm_ascend.ops.triton import sfa_dcp as ops

    torch.npu.set_device(0)
    width = 2048
    # Warming only one row makes this test fail if row-count specialization
    # returns, even though production exercises every supported decode size.
    ops.warmup_sfa_dcp_mask(1, width, torch.device("npu:0"))
    monkeypatch.setattr(ops._mask_topk, "compile", forbid_compile)
    for dtype in (torch.int32, torch.int64):
        for rows in [0, *range(1, 34), 256]:
            raw_cpu = (torch.arange(rows * width, dtype=torch.int32) % 1027 - 2).reshape(rows, 1, width)
            positions_cpu = torch.arange(rows, dtype=dtype) * 31 - 1
            raw, positions = raw_cpu.npu(), positions_cpu.npu()
            output = torch.full((rows + 2, 1, width), -99, dtype=torch.int32, device="npu")
            counts = torch.full((rows, 16), -77, dtype=torch.int32, device="npu")
            for active in sorted({0, min(1, rows), rows // 2, rows}):
                ops.prepare_sfa_dcp_topk(raw, positions, output[:rows], counts, active, 8, 3, 128, local=False)
                valid = (raw_cpu >= 0) & (raw_cpu <= positions_cpu[:, None, None])
                valid &= torch.arange(rows)[:, None, None] < active
                expected = torch.where(valid, raw_cpu, -1)
                assert torch.equal(output[:rows].cpu(), expected)
                assert output[rows:].cpu().eq(-99).all()
                assert counts.cpu().eq(-77).all()
            assert torch.equal(raw.cpu(), raw_cpu)


@torch.inference_mode()
def test_warmed_mask_preserves_causality_in_graph_replay(monkeypatch):
    from vllm_ascend.ops.triton import sfa_dcp as ops

    torch.npu.set_device(0)
    ops.warmup_sfa_dcp_mask(1, 2048, torch.device("npu:0"))
    monkeypatch.setattr(ops._mask_topk, "compile", forbid_compile)
    raw_cpu = torch.arange(2048, dtype=torch.int32).repeat(8, 1, 1)
    raw = raw_cpu.npu()
    positions = torch.zeros(8, dtype=torch.int32, device="npu")
    output, counts = torch.empty_like(raw), torch.empty((8, 16), dtype=torch.int32, device="npu")
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            ops.prepare_sfa_dcp_topk(raw, positions, output, counts, 6, 4, 0, 128, local=False)
        for end in (-1, 0, 127, 1023, 2047):
            positions.fill_(end)
            graph.replay()
            expected = torch.where(raw_cpu <= end, raw_cpu, -1)
            expected[6:].fill_(-1)
            assert torch.equal(output.cpu(), expected)
    finally:
        torch.npu.synchronize()
        graph.reset()
