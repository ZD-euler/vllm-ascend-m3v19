#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise real SFA, V-up and production bank handoffs without model weights.

Fixed synthetic top-k deliberately excludes the indexer and full model.
Zero queries give an independent oracle: attention must return the mean of
the selected BF16 values. Layer/request/position-dependent payloads distinguish
stale banks, wrong request mappings and damaged rows. No payload is read on CPU
until the whole submission loop has finished. V-up uses the serving method and
fixed one-hot weights, so its expected output needs no reference matmul. Run on
an idle NPU.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

BLOCK = 128
WIDTH = 512
V_WIDTH = 128
ROPE = 64
PERIOD = 257
LAYER_SHIFT = 17


def emit(**record):
    print("[SFA_RACE] " + json.dumps(record, ensure_ascii=False), flush=True)


def emit_result(result):
    # Keep terminal output small enough to paste; full details stay in JSON.
    emit(event="result", **{key: value for key, value in result.items() if key != "first_failures"})


def memory_plan(args):
    """Lower bounds for explicit buffers; excludes runtime/operator workspace."""
    requests, total_q = len(args.queries), sum(args.queries)
    rows = math.ceil((args.history + max(args.queries)) / BLOCK) * BLOCK
    source_rows = rows + (args.layers - 1) * LAYER_SHIFT
    pinned = (requests * source_rows * WIDTH + args.history * ROPE + args.layers * (total_q + requests * 8) * WIDTH) * 2
    banks = args.banks * requests * rows * (WIDTH + ROPE) * 2
    outputs = args.layers * total_q * args.heads * V_WIDTH * 2
    sfa_output = total_q * args.heads * WIDTH * 2
    weights = args.heads * WIDTH * V_WIDTH * 2
    audit = args.layers * (requests * (args.topk - 1) * 3 + total_q * 2) * WIDTH * 2
    return dict(
        pinned_gib=round(pinned / 2**30, 2),
        device_gib_without_workspace=round((banks + outputs + sfa_output + weights) / 2**30, 2),
        audit_extra_device_gib=round(audit / 2**30, 2),
        hold_sfa_extra_device_gib=round(args.layers * sfa_output / 2**30, 2),
    )


def values(torch, positions, request, width=WIDTH):
    """Exactly representable inputs before the explicitly modelled BF16 cast."""
    columns = torch.arange(width, device=positions.device)
    return (((positions[:, None] % PERIOD) - 128).float() / 256 + request / 2 + (columns % 8).float()[None, :] / 64).to(
        torch.bfloat16
    )


def selected_positions(torch, history, count, query_sizes):
    """Spread distinct indices across history, plus each query's own new KV.

    The two residues avoid averaging away the layer fingerprint. Every index
    is causal, including for the first query in each packed request.
    """
    pairs = math.ceil((count - 1) / 2)
    blocks = (history - 129) // PERIOD
    if blocks < pairs - 1:
        raise ValueError("history too short for distinct patterned top-k positions")
    j = torch.arange(count - 1)
    base = (j // 2) * blocks // max(1, pairs - 1) * PERIOD + (j % 2) * 128
    return torch.cat(
        [torch.cat((base.expand(q, -1), (history + torch.arange(q))[:, None]), dim=1) for q in query_sizes]
    ).to(torch.int32)


def expected_attention(torch, indices, layer, request, column_count=WIDTH):
    # Sum history once; only the final selected (current-token) row varies.
    prefix = values(torch, indices[0, :-1].long() + layer * LAYER_SHIFT, request, column_count)
    tail = values(torch, indices[:, -1].long() + layer * LAYER_SHIFT, request, column_count)
    return (prefix.float().sum(0)[None, :] + tail.float()) / indices.shape[1]


def projection_columns(torch):
    return torch.arange(V_WIDTH) * 3 % WIDTH


def expected_projected(torch, attention, heads):
    """Account for the SFA output cast, then evaluate the one-hot projection."""
    selected = attention.to(torch.bfloat16)[:, projection_columns(torch)]
    signs = 1 - 2 * (torch.arange(heads) % 2)
    return (selected[:, None, :].float() * signs[None, :, None]).reshape(attention.shape[0], heads * V_WIDTH)


def compare(torch, actual, expected, *, exact=False):
    """Return an actual numerical failure, never label one changed bit corrupt."""
    a, e = actual.float(), expected.float().expand_as(actual)
    difference = (a - e).abs()
    bad = ~torch.isfinite(a) | (difference != 0 if exact else difference > 0.004 + e.abs() * 0.008)
    count = int(bad.sum())
    result = {"different_values": count, "max_abs_error": float(difference.nan_to_num().max())}
    if count:
        first = tuple(int(x) for x in bad.nonzero()[0])
        result.update(index=list(first), actual=float(a[first]), expected=float(e[first]))
    return result


class Fixture:
    """All host sources remain immutable and alive through final DMA completion."""

    def __init__(self, runtime, args):
        t = runtime.torch
        self.rt, self.args = runtime, args
        self.queries = tuple(args.queries)
        self.requests, self.total_q = len(self.queries), sum(self.queries)
        self.rows = math.ceil((args.history + max(self.queries)) / BLOCK) * BLOCK
        self.source_rows = self.rows + (args.layers - 1) * LAYER_SHIFT
        host = {"dtype": t.bfloat16, "pin_memory": runtime.device != "cpu"}
        self.sources = []
        for r in range(self.requests):
            source = t.empty((self.source_rows, WIDTH), **host)
            for start in range(0, self.source_rows, 16384):
                end = min(start + 16384, self.source_rows)
                source[start:end].copy_(values(t, t.arange(start, end), r))
            self.sources.append(source)
        # RoPE is held at zero; this test isolates latent bank payload/ordering.
        self.rope_source = t.zeros((args.history, ROPE), **host)
        self.indices_cpu = selected_positions(t, args.history, args.topk, self.queries)
        self.indices = self.indices_cpu.to(runtime.device).unsqueeze(1)
        self.q_lens = t.tensor(self.queries, dtype=t.int32).cumsum(0).to(t.int32).to(runtime.device)
        self.kv_lens = t.tensor([args.history + q for q in self.queries], dtype=t.int32, device=runtime.device)
        # Reverse request order physically, exercising block-table addressing.
        table = t.arange(self.requests * (self.rows // BLOCK), dtype=t.int32).reshape(self.requests, -1).flip(0)
        self.table = table.to(runtime.device)
        self.probes = t.linspace(0, args.history - 1, 8).long().unique()
        # The failure rerun checks every history row selected by SFA (about
        # 2048 of 500k), not eight samples pretending to cover all inputs.
        self.audit_history = self.indices_cpu[0, :-1].long().unique()
        self.audit_history_device = self.audit_history.to(runtime.device)
        self.audit_positions = [
            t.cat((self.audit_history, t.arange(args.history, args.history + q))) for q in self.queries
        ]
        self.audit_positions_device = [p.to(runtime.device) for p in self.audit_positions]
        self.a = t.ones((self.total_q, 128), dtype=t.bfloat16, device=runtime.device)
        self.b = t.ones((128, args.heads * WIDTH), dtype=t.bfloat16, device=runtime.device)
        self.b[64:].fill_(-1)  # Real matmul, exactly zero output.
        self.q_rope = t.zeros((self.total_q, args.heads, ROPE), dtype=t.bfloat16, device=runtime.device)
        weight = t.zeros((args.heads, WIDTH, V_WIDTH), dtype=t.bfloat16)
        weight[:, projection_columns(t), t.arange(V_WIDTH)] = (1 - 2 * (t.arange(args.heads) % 2))[:, None].to(
            t.bfloat16
        )
        self.v_up_weight = weight.to(runtime.device)
        runtime.npu.synchronize()

    def physical_request(self, request):
        return self.requests - 1 - request

    def run(self, mode, observe=False):
        with self.rt.torch.inference_mode():
            return self._run(mode, observe)

    def _run(self, mode, observe=False):
        rt, args = self.rt, self.args
        t, npu = rt.torch, rt.npu
        compute = npu.current_stream()
        streams = [npu.Stream() for _ in range(args.banks)]
        key = t.full((args.banks, self.requests * self.rows, 1, WIDTH), -7, dtype=t.bfloat16, device=rt.device)
        rope = t.full((args.banks, self.requests * self.rows, 1, ROPE), -7, dtype=t.bfloat16, device=rt.device)
        connector = object.__new__(rt.connector_cls)
        connector._layerwise_prefill_dma = True
        connector._layerwise_prefill_bank_counts = {0: args.banks}
        connector._layerwise_prefill_transfer_generations = {0: 0}
        connector._layerwise_prefill_dma_streams = {(0, b): s for b, s in enumerate(streams)}
        connector.load_stream = streams[0]
        fence_mode = "p_vup" if mode == "p_vup_hold" else mode
        connector.fence_debug = (
            rt.fence_cls("P", 0, {"LMCACHE_PREFILL_FENCE_DEBUG": fence_mode})
            if fence_mode in ("p_sfa", "p_use", "p_vup")
            else None
        )
        _, _, _, loads = connector._layerwise_prefill_transfer_state()
        request_ids = tuple(f"r{r}" for r in range(self.requests))
        host = {"dtype": t.bfloat16, "pin_memory": rt.device != "cpu"}
        saved = [t.full((args.layers, q + len(self.probes), WIDTH), -9, **host) for q in self.queries]
        outputs, observations, events = [], [], []
        held_sfa = []
        loaded_samples = {}
        npu.synchronize()  # Finish poison/constant setup before any H2D.

        def load(layer):
            b = layer % args.banks
            for r in reversed(range(self.requests)):
                dst = self.physical_request(r) * self.rows
                copies = [
                    (key[b, dst].data_ptr(), self.sources[r][layer * LAYER_SHIFT].data_ptr(), args.history * WIDTH * 2),
                    (rope[b, dst].data_ptr(), self.rope_source.data_ptr(), args.history * ROPE * 2),
                ]
                with npu.stream(streams[b]):
                    rt.ops.layerwise_prefill_dma_copy(copies, False)
                    if observe:
                        loaded_samples[layer, r] = key[b, dst : dst + self.rows, 0].index_select(
                            0, self.audit_history_device
                        )
                done = npu.Event()
                done.record(streams[b])
                events.append(done)
                loads.setdefault((0, layer), {})[request_ids[r]] = (0, done)

        def save_and_advance(layer):
            b = layer % args.banks
            connector._require_layerwise_prefill_bank_use(layer, 0, b)
            copies = []
            for r, q in enumerate(self.queries):
                src = self.physical_request(r) * self.rows
                copies.append((saved[r][layer, 0].data_ptr(), key[b, src + args.history].data_ptr(), q * WIDTH * 2))
                for i, pos in enumerate(self.probes.tolist()):
                    copies.append((saved[r][layer, q + i].data_ptr(), key[b, src + pos].data_ptr(), WIDTH * 2))
            with npu.stream(streams[b]):
                rt.ops.layerwise_prefill_dma_copy(copies, True)
            if layer + args.banks < args.layers:
                load(layer + args.banks)

        def snapshot(b):
            # Only the diagnostic rerun adds these device copies. No host wait.
            return [
                key[
                    b, self.physical_request(r) * self.rows : (self.physical_request(r) + 1) * self.rows, 0
                ].index_select(0, self.audit_positions_device[r])
                for r in range(self.requests)
            ]

        started = time.perf_counter()
        try:
            for layer in range(args.banks):
                load(layer)
            for layer in range(args.layers):
                if layer:
                    save_and_advance(layer - 1)  # Same next-layer submission point as serving.
                b = layer % args.banks
                query = t.mm(self.a, self.b).view(self.total_q, args.heads, WIDTH)
                connector.wait_for_layerwise_prefill_load(layer, 0, request_ids=request_ids)
                for r, q in enumerate(self.queries):
                    start = self.physical_request(r) * self.rows + args.history
                    new = values(t, t.arange(args.history, args.history + q, device=rt.device) + layer * LAYER_SHIFT, r)
                    key[b, start : start + q, 0].copy_(new)
                    rope[b, start : start + q].zero_()
                if observe:
                    before = snapshot(b)
                if mode == "serial":
                    npu.synchronize()
                elif mode == "p_sfa":
                    connector.synchronize_prefill_sfa_inputs(layer, 0)
                result = rt.sfa(
                    query=query,
                    key=key[b].view(-1, BLOCK, 1, WIDTH),
                    value=key[b].view(-1, BLOCK, 1, WIDTH),
                    sparse_indices=self.indices,
                    scale_value=1 / math.sqrt(WIDTH + ROPE),
                    sparse_block_size=1,
                    block_table=self.table,
                    actual_seq_lengths_query=self.q_lens,
                    actual_seq_lengths_kv=self.kv_lens,
                    query_rope=self.q_rope,
                    key_rope=rope[b].view(-1, BLOCK, 1, ROPE),
                    layout_query="TND",
                    layout_kv="PA_BSND",
                    sparse_mode=3,
                )
                if mode == "hold_sfa":
                    # Deliberately change only this tensor's lifetime; do not
                    # insert a copy or device fence into the control run.
                    held_sfa.append(result)
                if mode == "serial":
                    compute.synchronize()
                if observe:
                    observations.append((before, snapshot(b), query.clone()))
                used = npu.Event()
                used.record(compute)
                connector.record_layerwise_prefill_bank_use(layer, 0, used)
                if mode == "after_bank_wait":
                    # Distinguish p_use's wait-before-enqueue from waiting for
                    # the identical event immediately after the bank handoff.
                    used.synchronize()
                if layer == args.layers - 1:
                    # The final layer has no following entry callback. Serving
                    # submits its save before V-up, after the last-use event.
                    save_and_advance(layer)
                if mode == "before_vup":
                    # Complete the same SFA event after all bank handoff/save
                    # submissions, but before any projection work is submitted.
                    used.synchronize()
                # Assignment releases the original SFA output as in serving.
                # Do not keep it in outputs or another local in ordinary modes.
                sfa_owner = result if mode == "p_vup_hold" else None
                result = rt.projection(result, self.v_up_weight)
                if fence_mode == "p_vup":
                    connector.synchronize_prefill_vup_output(layer, 0)
                # This case has exactly p_vup's fences. Keep just this layer's
                # original output alive through the fence, not all layers.
                del sfa_owner
                outputs.append(result)
        finally:
            # Also retain raw-pointer owners until partial submissions finish on error.
            npu.synchronize()

        failures, checked = [], 0

        def check(stage, actual, expected, layer, r, exact=False):
            nonlocal checked
            checked += actual.numel()
            record = compare(t, actual, expected, exact=exact)
            if record["different_values"]:
                index = record["index"][0]
                token = args.history + index
                if stage == "save_history_probe":
                    token = int(self.probes[index])
                elif stage == "after_h2d_history":
                    token = int(self.audit_history[index])
                elif stage in ("before_sfa_history", "after_sfa_history"):
                    token = int(self.audit_positions[r][index])
                failures.append(dict(stage=stage, layer=layer, request=r, token=token, **record))

        for layer, output in enumerate(outputs):
            row = 0
            for r, q in enumerate(self.queries):
                expected = expected_attention(t, self.indices_cpu[row : row + q], layer, r)
                check("vup_output", output[row : row + q].cpu(), expected_projected(t, expected, args.heads), layer, r)
                if held_sfa:
                    check("sfa_output", held_sfa[layer][row : row + q].cpu(), expected[:, None, :], layer, r)
                tail_pos = t.arange(args.history, args.history + q) + layer * LAYER_SHIFT
                check("save_new_kv", saved[r][layer, :q], values(t, tail_pos, r), layer, r, True)
                history = values(t, self.probes + layer * LAYER_SHIFT, r)
                check("save_history_probe", saved[r][layer, q:], history, layer, r, True)
                if observe:
                    before, after, query = observations[layer]
                    selected_history = values(t, self.audit_history + layer * LAYER_SHIFT, r)
                    selected_kv = values(t, self.audit_positions[r] + layer * LAYER_SHIFT, r)
                    check("after_h2d_history", loaded_samples[layer, r].cpu(), selected_history, layer, r, True)
                    check("before_sfa_history", before[r].cpu(), selected_kv, layer, r, True)
                    check("after_sfa_history", after[r].cpu(), selected_kv, layer, r, True)
                    check("query", query[row : row + q].cpu(), t.tensor(0.0), layer, r, True)
                row += q
        first_by_stage = {}
        for failure in failures:
            first_by_stage.setdefault(failure["stage"], failure)
        return dict(
            mode=mode,
            projection_branch=getattr(rt, "projection_branch", "injected"),
            sfa_shape=[self.total_q, args.heads, WIDTH],
            vup_shape=[self.total_q, args.heads * V_WIDTH],
            output_stage="vup_output",
            retain_sfa_output=mode == "hold_sfa",
            retain_sfa_until_vup=mode == "p_vup_hold",
            observe_inputs=observe,
            passed=not failures,
            seconds=round(time.perf_counter() - started, 3),
            checked_values=checked,
            failures=len(failures),
            first_failures=failures[:8],
            first_by_stage=first_by_stage,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--history", type=int, default=499712)
    parser.add_argument("--queries", nargs="+", type=int, default=[378, 385, 383])
    parser.add_argument("--layers", type=int, default=79)
    parser.add_argument("--banks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "event",
            "p_sfa",
            "p_use",
            "after_bank_wait",
            "before_vup",
            "p_vup",
            "p_vup_hold",
            "hold_sfa",
        ),
        default=["event", "p_sfa", "p_use", "after_bank_wait", "p_vup", "hold_sfa"],
    )
    parser.add_argument("--output", default="prefill-sfa-race.json")
    args = parser.parse_args()
    if min(args.history, args.layers, args.banks, args.heads, args.repeat, *args.queries) <= 0:
        parser.error("sizes/repetitions must be positive")
    if args.layers <= args.banks or args.topk < 3:
        parser.error("layers must exceed banks; topk must be at least 3")
    if os.getenv("ASCEND_LAUNCH_BLOCKING", "0") != "0":
        parser.error("ASCEND_LAUNCH_BLOCKING must be 0 for an asynchronous test")
    import torch
    import torch_npu

    from vllm_ascend.platform import NPUPlatform

    # Match serving's custom-OPP registration before initializing the device.
    # Import extensions only after torch has loaded libtorch dependencies.
    NPUPlatform.import_kernels()
    torch.npu.set_device(args.device)
    importlib.import_module("vllm_ascend.vllm_ascend_C")
    ops = importlib.import_module("lmcache_ascend.c_ops")
    from lmcache_ascend.v1.kv_fence_debug import KVFenceDebug
    from lmcache_ascend.v1.npu_connector.npu_connectors import VLLMPagedMemLayerwiseNPUConnector

    from vllm_ascend.attention.sfa_v1 import BMM_TRANS_MAX_SUPPORTED_TOKENS, AscendSFAImpl

    # The serving method only needs these dimensions and W_UV; no model or
    # alternate implementation of the projection is constructed here.
    projection_state = SimpleNamespace(
        local_num_heads=args.heads,
        kv_lora_rank=WIDTH,
        v_head_dim=V_WIDTH,
    )

    def projection(x, weight):
        projection_state.W_UV = weight
        return AscendSFAImpl._v_up_proj(projection_state, x)

    projection_branch = (
        "custom_batch_matmul_transpose"
        if sum(args.queries) <= BMM_TRANS_MAX_SUPPORTED_TOKENS
        and hasattr(torch.ops._C_ascend, "batch_matmul_transpose")
        else "native_torch_bmm"
    )

    rt = SimpleNamespace(
        torch=torch,
        npu=torch.npu,
        device="npu",
        ops=ops,
        connector_cls=VLLMPagedMemLayerwiseNPUConnector,
        fence_cls=KVFenceDebug,
        sfa=torch.ops._C_ascend.npu_sparse_flash_attention,
        projection=projection,
        projection_branch=projection_branch,
    )
    emit(
        event="setup",
        config=vars(args),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        task_queue=os.getenv("TASK_QUEUE_ENABLE"),
        memory=memory_plan(args),
        projection_branch=projection_branch,
        scope=(
            "real SFA + serving V-up + bank DMA; synthetic Q/KV/topk/weights; "
            "no indexer, FFN, TP communication, graph capture/replay or PD transport"
        ),
    )
    fixture = Fixture(rt, args)
    results = []
    output = Path(args.output)
    emit(event="running", mode="serial", purpose="validate independent numerical oracle")
    control = fixture.run("serial")
    results.append(control)
    emit_result(control)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not control["passed"]:
        emit(event="control_failed", message="Serial control failed; stop, do not classify as an overlap race.")
        return 2
    for repeat in range(args.repeat):
        for mode in args.modes:
            emit(event="running", repeat=repeat, mode=mode)
            result = fixture.run(mode)
            result["repeat"] = repeat
            results.append(result)
            emit_result(result)
            output.write_text(json.dumps(results, indent=2), encoding="utf-8")
            if mode == "event" and not result["passed"]:
                observed = fixture.run(mode, observe=True)
                observed["repeat"] = repeat
                results.append(observed)
                emit_result(observed)
                output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tested_modes = {r["mode"] for r in results if r["mode"] != "serial" and not r["observe_inputs"]}
    failed_modes = sorted({r["mode"] for r in results if not r["passed"] and not r["observe_inputs"]})
    status = (
        "reproduced"
        if "event" in failed_modes
        else "selected_modes_failed"
        if failed_modes
        else "not_reproduced"
        if "event" in tested_modes
        else "selected_modes_passed"
    )
    emit(
        event="complete",
        status=status,
        event_tested="event" in tested_modes,
        failed_modes=failed_modes,
        report=str(output.resolve()),
        scope="SFA + serving V-up + bank handoff; not_reproduced does not certify full serving correctness",
    )
    return int(not all(r["passed"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
