# SPDX-License-Identifier: Apache-2.0
"""Installed MC2 V2 unequal-input/native-versus-capture probe, on 16 NPUs.

Launch pytest under torchrun on both hosts; see sfa_independent_dp.md. This is an
operator/graph contract test, NOT a full-SFA server, KV lifetime or throughput test.
The FC1 variant adds TP4 gather/unpad, row-parallel projection and reduce-scatter;
it includes communication padding and zero-input idle-like participation.
The decoder-DCP variant adds actual counted SFA and its query/output exchanges,
with dense-reference parity and whole three-layer pipeline completion timings.
"""

import os
import statistics
import time
from datetime import timedelta
from types import SimpleNamespace as NS

import pytest
import torch
import torch.distributed as dist

torch_npu = pytest.importorskip("torch_npu")


@torch.inference_mode()
@pytest.mark.parametrize("layout", ["ordinary", "flashcomm1", "decoder-dcp"])
def test_unequal_mc2_inputs_with_independent_native_and_graph_routes(layout):
    flashcomm1, decoder_dcp = layout == "flashcomm1", layout == "decoder-dcp"
    assert int(os.environ.get("WORLD_SIZE", "1")) == 16, "Launch this qualification probe with 16 torchrun ranks"
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    # The operator documentation requires a domain dedicated to this pair.
    mc2 = dist.new_group(backend="hccl", timeout=timedelta(seconds=120))
    group = mc2._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    tp_size, hidden = 4, 2048
    capacities = (2, 4, 6, 8, 10, 12, 24) if flashcomm1 else (1, 2, 4, 8)

    def local_count(tokens):
        return (tokens + tp_size - 1) // tp_size if flashcomm1 else tokens

    global_bs = local_count(max(capacities)) * world
    if flashcomm1 or decoder_dcp:
        # Create TP domains in the same global order; never reuse MC2's domain.
        for start in range(0, world, tp_size):
            subgroup = dist.new_group(
                list(range(start, start + tp_size)), backend="hccl", timeout=timedelta(seconds=120)
            )
            if start <= rank < start + tp_size:
                tp = subgroup
    if flashcomm1:
        begin = (rank % tp_size) * (hidden // tp_size)
        columns = slice(begin, begin + hidden // tp_size)
        weight = torch.eye(hidden, dtype=torch.bfloat16, device="npu")[columns].contiguous()
    assert hasattr(torch_npu, "npu_moe_distribute_dispatch_v2")
    assert hasattr(torch_npu, "npu_moe_distribute_combine_v2")
    if decoder_dcp:
        from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

        import vllm_ascend.vllm_ascend_C  # noqa: F401
        from vllm_ascend.ops.triton.sfa_dcp import execute_sfa_dcp

        # DCP=TP inside each DP replica; MC2 still spans all replicas.
        dcp = NS(device_group=tp, rank_in_group=rank % tp_size, world_size=tp_size)
        dcp.all_gather = lambda tensor, dim: DeviceCommunicatorBase.all_gather(dcp, tensor, dim)
        generator = torch.Generator().manual_seed(174 + rank // tp_size)
        cache = torch.randn((tp_size, 128, 576), generator=generator, dtype=torch.bfloat16)
        kv = tuple(
            cache[rank % tp_size, :, part].contiguous().reshape(1, 128, 1, dim).npu()
            for part, dim in ((slice(0, 512), 512), (slice(512, 576), 64))
        )
        dcp_batches = {}
        for tokens in capacities:
            count = 128 - (rank % tp_size) * 7
            indices = torch.full((tokens, 1, 2048), -1, dtype=torch.int32, device="npu")
            indices[:, 0, :count] = torch.arange(count, dtype=torch.int32, device="npu")
            counts = torch.zeros((tokens, 16), dtype=torch.int32, device="npu")
            counts[:, 0] = count
            dcp_batches[tokens] = (
                torch.zeros((tokens, hidden // 512, 64), dtype=torch.bfloat16, device="npu"),
                indices,
                counts,
                torch.tensor([[0]], dtype=torch.int32, device="npu"),
                torch.tensor([tokens], dtype=torch.int32, device="npu"),
                torch.tensor([128], dtype=torch.int32, device="npu"),
            )
        keys = torch.cat([cache[peer, : 128 - peer * 7] for peer in range(tp_size)]).float()

        def reference(value, idle):
            for _ in range(3):
                query = value.float().reshape(-1, hidden // 512, 512)
                scores = (query @ keys[:, :512].T) * (576**-0.5)
                value = (scores.softmax(-1) @ keys[:, :512]).reshape_as(value).bfloat16()
                if idle:
                    value.zero_()
            return value

    def transfer(x, ids, weights, quant):
        output = torch_npu.npu_moe_distribute_dispatch_v2(
            x=x,
            expert_ids=ids,
            scales=None,
            quant_mode=quant,
            group_ep=group,
            ep_world_size=world,
            ep_rank_id=rank,
            group_tp=group,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=world,
            global_bs=global_bs,
            expert_token_nums_type=0,
        )
        expanded, scales, assist, _, ep_counts, tp_counts, expand_scales = output[:7]
        if quant:
            # Identity expert computation after the dispatch's per-token quantization.
            expanded = (expanded.float() * scales.reshape(-1, 1)).to(x.dtype)
            tp_counts = torch.empty(1, dtype=torch.int32, device=x.device)
        return torch_npu.npu_moe_distribute_combine_v2(
            expand_x=expanded,
            expert_ids=ids,
            expert_scales=weights,
            assist_info_for_combine=assist,
            ep_send_counts=ep_counts,
            tp_send_counts=tp_counts,
            expand_scales=expand_scales,
            group_ep=group,
            ep_world_size=world,
            ep_rank_id=rank,
            group_tp=group,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            moe_expert_num=world,
            global_bs=global_bs,
        )

    def layer(x, ids, weights, quant, tokens):
        if decoder_dcp:
            rope, indices, counts, table, query_ends, kv_lens = dcp_batches[tokens]
            x = execute_sfa_dcp(
                x.reshape(tokens, hidden // 512, 512),
                rope,
                kv,
                indices,
                counts,
                table,
                query_ends,
                kv_lens,
                576**-0.5,
                dcp,
            ).reshape(tokens, hidden)
        if flashcomm1:
            gathered = torch.empty((x.shape[0] * tp_size, hidden), dtype=x.dtype, device=x.device)
            dist.all_gather_into_tensor(gathered, x, group=tp)
            # Each TP rank contributes disjoint hidden columns; RS reconstructs
            # the identity projection and returns the original local token rows.
            partial = gathered[:tokens, columns].contiguous() @ weight
            partial = torch.nn.functional.pad(partial, (0, 0, 0, gathered.shape[0] - tokens))
            x = torch.empty_like(x)
            dist.reduce_scatter_tensor(x, partial, group=tp)
        return transfer(x, ids, weights, quant)

    def step(x, ids, weights, quant, tokens):
        # Two target-like layers and one draft-like layer, in every route.
        for _ in range(3):
            x = layer(x, ids, weights, quant, tokens)
        return x

    failures, timings = [], {"graph": [], "native": []}
    try:
        # Retain every graph until the entire qualification finishes. Releasing
        # a graph while reusing its AIV communicator can affect later captures.
        retained = []
        for quant in (0, 2):
            captures = {}
            for tokens in capacities:
                size = local_count(tokens)
                x = torch.zeros((size, hidden), dtype=torch.bfloat16, device="npu")
                ids = (
                    torch.arange(size, device="npu", dtype=torch.int32)[:, None]
                    + torch.tensor([[rank, rank + 3]], device="npu", dtype=torch.int32)
                ) % world
                weights = torch.full((size, 2), 0.5, dtype=torch.float32, device="npu")
                for _ in range(2):
                    step(x, ids, weights, quant, tokens)
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    result = step(x, ids, weights, quant, tokens)
                captures[tokens] = (graph, x, ids, weights, result)
                retained.append(captures[tokens])
                torch.npu.synchronize()
            for iteration in range(2 * len(capacities)):
                dp = rank // 4
                tokens = capacities[(dp + iteration) % len(capacities)]
                size = local_count(tokens)
                graph, x, ids, weights, result = captures[tokens]
                values = torch.arange(size * hidden, device="npu", dtype=torch.float32).reshape(size, hidden)
                x.copy_(torch.sin(values * 0.17 + rank * 0.71 + iteration * 0.3).to(x.dtype))
                if decoder_dcp:
                    idle = (dp + iteration) % 5 == 0
                    dcp_batches[tokens][2][:, 0].fill_(0 if idle else 128 - (rank % tp_size) * 7)
                    expected = reference(x.cpu(), idle)
                if flashcomm1:
                    valid = max(0, min(size, tokens - (rank % tp_size) * size))
                    x[valid:].zero_()
                    if (dp + iteration) % 5 == 0:
                        x.zero_()  # All TP peers still execute all three phases.
                route = "native" if (dp + iteration) % 2 else "graph"
                started = time.perf_counter_ns()
                if route == "native":
                    observed = step(x, ids, weights, quant, tokens)
                else:
                    graph.replay()
                    observed = result
                torch.npu.synchronize()
                timings[route].append((time.perf_counter_ns() - started) / 1000)
                try:
                    torch.testing.assert_close(
                        observed.cpu(),
                        expected if decoder_dcp else x.cpu(),
                        rtol=0.03 if decoder_dcp else 0,
                        atol=0.04 if decoder_dcp or quant else 0,
                    )
                except AssertionError as error:
                    failures.append(f"rank={rank} quant={quant} iteration={iteration} route={route}: {error}")
            dist.barrier()  # Default group, never the MC2 communication domain.
        if decoder_dcp:
            print(
                f"DCP+MC2 pipeline rank={rank} median_us="
                f"{ {route: statistics.median(samples) for route, samples in timings.items()} }",
                flush=True,
            )
        assert not failures, "\n".join(failures)
    finally:
        torch.npu.synchronize()
        for captured, *_ in retained:
            captured.reset()
        dist.destroy_process_group()
