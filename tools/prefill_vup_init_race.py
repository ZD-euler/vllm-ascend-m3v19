#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe serving or legacy V-up against an outstanding same-stream read.

Run each mode in a fresh process on an idle NPU. No model or P server is used.
The read uses LMCache's raw-pointer DMA on the allocation's own stream. Freeing
the source then reusing its cached storage is safe for stream-ordered writes;
the legacy custom V-up used a direct, unstreamed aclrtMemcpy instead.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import struct
import time
from pathlib import Path
from types import SimpleNamespace

# csrc/batch_matmul_transpose/op_host/{batch_matmul_transpose.h,tiling/tiling_data.h}.
# A CPU regression checks these against the C++ declarations.
TILING_BYTES = 72
TILING_SLOTS = 1024
CANARY_BYTES = TILING_BYTES * TILING_SLOTS
CANARY_VALUE = 0xA5
READBACK_VALUE = 0x5A
K_WIDTH = 512
V_WIDTH = 128
GUARD_BYTES = 512
MODES = ("cold", "sync", "hold", "warm")


def emit(**record):
    print("[VUP_INIT] " + json.dumps(record), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--operator", choices=("serving", "legacy"), default="serving")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--backlog-size", type=int, default=4096)
    parser.add_argument("--backlog-ops", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    max_tokens = TILING_SLOTS if args.operator == "legacy" else 4096
    if not 1 <= args.tokens <= max_tokens:
        parser.error(f"tokens must be in [1, {max_tokens}]")
    if not 1 <= args.heads <= 128:
        parser.error("heads must be in [1, 128]")
    if not 16 <= args.backlog_size <= 8192 or args.backlog_size % 16:
        parser.error("backlog-size must be a multiple of 16 in [16, 8192]")
    if not 1 <= args.backlog_ops <= 2048:
        parser.error("backlog-ops must be in [1, 2048]")
    if args.device < 0:
        parser.error("device must be nonnegative")
    if os.getenv("ASCEND_LAUNCH_BLOCKING", "0") != "0":
        parser.error("ASCEND_LAUNCH_BLOCKING must be 0 for an asynchronous test")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


class NpuRuntime:
    def __init__(self, torch, ops, projection=None, projection_branch="legacy_batch_matmul_transpose"):
        self.torch, self.ops = torch, ops
        self.npu, self.device = torch.npu, "npu"
        self.projection, self.projection_branch = projection, projection_branch

    def copy_out(self, destination, source):
        # pybind.cpp submits aclrtMemcpyAsync on getCurrentNPUStream().stream().
        # Only integer addresses are passed, so no tensor owner is retained.
        self.ops.layerwise_prefill_dma_copy([(destination.data_ptr(), source.data_ptr(), source.numel())], True)

    def project(self, a, b, out):
        if self.projection is None:
            self.torch.ops._C_ascend.batch_matmul_transpose(a, b, out)
        else:
            state = SimpleNamespace(local_num_heads=a.shape[1], kv_lora_rank=a.shape[2], v_head_dim=b.shape[2], W_UV=b)
            out.copy_(self.projection(state, a).view_as(out))


def load_runtime(args):
    import torch
    import torch_npu

    from vllm_ascend.platform import NPUPlatform

    NPUPlatform.import_kernels()
    torch.npu.set_device(args.device)
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    ops = importlib.import_module("lmcache_ascend.c_ops")
    if args.operator == "legacy" and not hasattr(torch.ops._C_ascend, "batch_matmul_transpose"):
        raise RuntimeError("Rebuild vllm-ascend with batch_matmul_transpose before running this probe")
    if not hasattr(ops, "layerwise_prefill_dma_copy"):
        raise RuntimeError("Rebuild LMCache-Ascend with layerwise_prefill_dma_copy before running this probe")
    projection = None
    branch = "legacy_batch_matmul_transpose"
    if args.operator == "serving":
        from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

        projection = AscendSFAImpl._v_up_proj
        branch = "npu_transpose_batchmatmul" if hasattr(torch_npu, "npu_transpose_batchmatmul") else "native_torch_bmm"
    emit(
        event="runtime",
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        task_queue=os.getenv("TASK_QUEUE_ENABLE"),
        allocator_config=os.getenv("PYTORCH_NPU_ALLOC_CONF"),
        operator=args.operator,
        projection_branch=branch,
        scope="first V-up tiling allocation and same-stream temporary reuse",
    )
    return NpuRuntime(torch, ops, projection, branch)


def inspect_readback(payload, tokens, heads):
    if len(payload) != CANARY_BYTES:
        raise ValueError(f"expected {CANARY_BYTES} readback bytes, got {len(payload)}")
    # The retired custom operator only had 1024 slots. Larger serving probes
    # still audit all bytes, with no legacy slot/signature to interpret.
    offset = (tokens - 1) * TILING_BYTES if tokens <= TILING_SLOTS else None
    changed = [i for i, value in enumerate(payload) if value != CANARY_VALUE]
    outside = sum(offset is None or not offset <= i < offset + TILING_BYTES for i in changed)
    shape = list(struct.unpack_from("<4I", payload, offset)) if offset is not None else None
    signature = shape == [heads, tokens, K_WIDTH, V_WIDTH]
    return dict(
        changed_bytes=len(changed),
        first_changed_byte=changed[0] if changed else None,
        last_changed_byte=changed[-1] if changed else None,
        changed_bytes_outside_slot=outside,
        slot_offset=offset,
        slot_shape_words=shape,
        tiling_signature=signature,
        tiling_overwrite=bool(signature and changed and outside == 0),
    )


def classify(mode, audit, pending, output_zero):
    if not output_zero:
        return "projection_failed", 2
    if audit["changed_bytes"]:
        if mode == "cold" and audit["tiling_overwrite"]:
            return "tiling_overwrite_reproduced", 1
        return "control_failed" if mode != "cold" else "unexpected_readback", 2
    if not pending and mode != "sync":
        return "reader_finished_before_vup", 0
    return "control_clean" if mode != "cold" else "overwrite_not_observed", 0


def run_probe(rt, args):
    t, npu = rt.torch, rt.npu
    device = {"device": rt.device}
    # Allocate all large buffers first. No allocations are made between freeing
    # the canary and calling the real custom operator below.
    a = t.ones((args.tokens, args.heads, K_WIDTH), dtype=t.bfloat16, **device)
    b = t.zeros((args.heads, K_WIDTH, V_WIDTH), dtype=t.bfloat16, **device)
    out = t.full((args.tokens, args.heads, V_WIDTH), float("nan"), dtype=t.bfloat16, **device)
    size = args.backlog_size
    left = t.ones((size, size), dtype=t.float32, **device)
    right = t.zeros((size, size), dtype=t.float32, **device)
    product = t.empty_like(left)
    readback = t.full((CANARY_BYTES,), READBACK_VALUE, dtype=t.uint8, pin_memory=rt.device != "cpu")
    # Warm up ordinary matmul and raw DMA, keeping the target operator cold.
    t.mm(left, right, out=product)
    warm_source = t.full((CANARY_BYTES,), CANARY_VALUE, dtype=t.uint8, **device)
    rt.copy_out(readback, warm_source)
    npu.synchronize()
    del warm_source
    if args.mode == "warm":
        rt.project(a, b, out)
        npu.synchronize()
    npu.empty_cache()

    source = t.full((CANARY_BYTES,), CANARY_VALUE, dtype=t.uint8, **device)
    # Keep its neighbour alive to avoid merging the freed source into a larger
    # cached block. This favours exact-size reuse by V-up's first allocation.
    guard = t.empty((GUARD_BYTES,), dtype=t.uint8, **device)
    address = source.data_ptr()
    readback.fill_(READBACK_VALUE)
    out.fill_(float("nan"))
    event = npu.Event()
    npu.synchronize()
    emit(event="queued_test", mode=args.mode, source_address=hex(address))
    start = time.monotonic()
    try:
        for _ in range(args.backlog_ops):
            t.mm(left, right, out=product)
        rt.copy_out(readback, source)
        event.record(npu.current_stream())
        pending = not event.query()
        if args.mode == "sync":
            event.synchronize()
        owner = source if args.mode == "hold" else None
        del source
        project_start = time.monotonic()
        rt.project(a, b, out)
        project_call_ms = (time.monotonic() - project_start) * 1000
        pending_after = not event.query()
    finally:
        # Keep pinned destination, guard and held source alive even on failure.
        npu.synchronize()
    del owner, guard
    audit = inspect_readback(bytes(readback.tolist()), args.tokens, args.heads)
    output_zero = bool((out.cpu() == 0).all())
    status, exit_code = classify(args.mode, audit, pending, output_zero)
    return dict(
        mode=args.mode,
        operator=args.operator,
        projection_branch=getattr(rt, "projection_branch", "cpu_test"),
        project_call_ms=round(project_call_ms, 3),
        status=status,
        exit_code=exit_code,
        reader_pending_before_vup=pending,
        reader_pending_after_vup=pending_after,
        output_zero=output_zero,
        source_address=hex(address),
        candidate_write_address=hex(address + audit["slot_offset"]) if audit["slot_offset"] is not None else None,
        seconds=round(time.monotonic() - start, 3),
        **audit,
    )


def validate_projection(rt, args):
    """Check nonzero values after the cold probe, without a reference matmul.

    Each weight column selects one input column with a head-dependent sign.
    Small and >1024-token cases exercise both sides of the retired dispatch.
    """
    t = rt.torch
    token_counts = sorted({1, args.tokens, 1536} if args.operator == "serving" else {1, args.tokens})
    checks = []
    for tokens in token_counts:
        rows = (t.arange(tokens * args.heads).reshape(tokens, args.heads, 1) % 29).to(t.int16)
        columns = t.arange(K_WIDTH, dtype=t.int16)
        source = ((rows + columns) % 29 - 14).to(t.bfloat16) / 16
        selected = t.arange(V_WIDTH) * 3 % K_WIDTH
        signs = (1 - 2 * (t.arange(args.heads) % 2)).to(t.bfloat16)
        weight = t.zeros((args.heads, K_WIDTH, V_WIDTH), dtype=t.bfloat16)
        weight[:, selected, t.arange(V_WIDTH)] = signs[:, None]
        expected = source[:, :, selected] * signs[None, :, None]
        a, b = source.to(rt.device), weight.to(rt.device)
        out = t.empty((tokens, args.heads, V_WIDTH), dtype=t.bfloat16, device=rt.device)
        rt.project(a, b, out)
        actual = out.cpu()
        checks.append(dict(tokens=tokens, passed=bool(t.equal(actual, expected))))
    return dict(passed=all(check["passed"] for check in checks), cases=checks)


def main(argv=None):
    args = parse_args(argv)
    rt = load_runtime(args)
    with rt.torch.inference_mode():
        result = run_probe(rt, args)
        # Do this last so numerical validation never warms the race probe.
        if args.operator == "serving":
            result["numerical"] = validate_projection(rt, args)
            if not result["numerical"]["passed"]:
                result.update(status="projection_failed", exit_code=2)
    emit(event="result", **result)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
