#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe V-up's first tiling allocation against an outstanding same-stream read.

Run each mode in a fresh process on an idle NPU. No model or P server is used.
The read uses LMCache's raw-pointer DMA on the allocation's own stream. Freeing
the source then reusing its cached storage is safe for stream-ordered writes;
this probes the custom V-up's direct, unstreamed aclrtMemcpy instead.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import struct
import time
from pathlib import Path

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
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--backlog-size", type=int, default=4096)
    parser.add_argument("--backlog-ops", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.tokens <= TILING_SLOTS:
        parser.error(f"tokens must be in [1, {TILING_SLOTS}]")
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
    def __init__(self, torch, ops):
        self.torch, self.ops = torch, ops
        self.npu, self.device = torch.npu, "npu"

    def copy_out(self, destination, source):
        # pybind.cpp submits aclrtMemcpyAsync on getCurrentNPUStream().stream().
        # Only integer addresses are passed, so no tensor owner is retained.
        self.ops.layerwise_prefill_dma_copy([(destination.data_ptr(), source.data_ptr(), source.numel())], True)

    def project(self, a, b, out):
        self.torch.ops._C_ascend.batch_matmul_transpose(a, b, out)


def load_runtime(args):
    import torch
    import torch_npu

    from vllm_ascend.platform import NPUPlatform

    NPUPlatform.import_kernels()
    torch.npu.set_device(args.device)
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    ops = importlib.import_module("lmcache_ascend.c_ops")
    if not hasattr(torch.ops._C_ascend, "batch_matmul_transpose"):
        raise RuntimeError("Rebuild vllm-ascend with batch_matmul_transpose before running this probe")
    if not hasattr(ops, "layerwise_prefill_dma_copy"):
        raise RuntimeError("Rebuild LMCache-Ascend with layerwise_prefill_dma_copy before running this probe")
    emit(
        event="runtime",
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        task_queue=os.getenv("TASK_QUEUE_ENABLE"),
        allocator_config=os.getenv("PYTORCH_NPU_ALLOC_CONF"),
        scope="first V-up tiling allocation and same-stream temporary reuse",
    )
    return NpuRuntime(torch, ops)


def inspect_readback(payload, tokens, heads):
    if len(payload) != CANARY_BYTES:
        raise ValueError(f"expected {CANARY_BYTES} readback bytes, got {len(payload)}")
    offset = (tokens - 1) * TILING_BYTES
    changed = [i for i, value in enumerate(payload) if value != CANARY_VALUE]
    outside = sum(not offset <= i < offset + TILING_BYTES for i in changed)
    shape = list(struct.unpack_from("<4I", payload, offset))
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
        rt.project(a, b, out)
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
        status=status,
        exit_code=exit_code,
        reader_pending_before_vup=pending,
        reader_pending_after_vup=pending_after,
        output_zero=output_zero,
        source_address=hex(address),
        candidate_write_address=hex(address + audit["slot_offset"]),
        seconds=round(time.monotonic() - start, 3),
        **audit,
    )


def main(argv=None):
    args = parse_args(argv)
    rt = load_runtime(args)
    with rt.torch.inference_mode():
        result = run_probe(rt, args)
    emit(event="result", **result)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
