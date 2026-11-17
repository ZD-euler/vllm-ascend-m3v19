# SPDX-License-Identifier: Apache-2.0
"""Packet bits, feature tiling and deferred-query lifetime/ordering on CPU."""

import gc
import sys
import weakref
from dataclasses import dataclass
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
import torch
from sfa_test_support import ROOT, Pointer, definitions, extract, load_module
from test_sfa_dcp_forward import gather_queries as make_gather_queries
from test_sfa_dcp_merge import DcpTL

OPS = ROOT / "vllm_ascend/ops/triton/sfa_dcp.py"


@pytest.fixture
def gather_queries(monkeypatch):
    return make_gather_queries.__wrapped__(monkeypatch)


class Kernel:
    def __init__(self, name):
        self.tl = DcpTL()
        namespace = {"tl": self.tl, "get_element": lambda value, index: value[index]}
        extract(OPS, "_unpack_stat", namespace)
        self.body = extract(OPS, name, namespace)

    def __getitem__(self, grid):
        def run(*args):
            self.tl.programs = grid[0]
            for self.tl.pid in range(grid[0]):
                self.body(*(Pointer(arg) if isinstance(arg, torch.Tensor) else arg for arg in args))

        return run


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tile", [128, 256, 512])
@pytest.mark.parametrize("raw", [False, True])
def test_tuning_routes_and_merges_with_one_equal_sized_collective(world, dtype, tile, raw):
    rows, heads, dim = 5, world * 2, 512
    rng = torch.Generator().manual_seed(913)
    partial = torch.randn((world, rows, heads, dim), generator=rng, dtype=dtype)
    maxima = torch.randn((world, rows, heads), generator=rng) * 12
    sums = torch.rand((world, rows, heads), generator=rng) * 2048 + 1
    sums[:, 1] = 0  # all-empty row
    sums[0, 0] = 0  # empty rank
    maxima[sums == 0] = float("nan")
    maxima[:, 2] = 2**24  # max+log(sum) would lose the 1:2 mass ratio.
    sums[:, 2] = 0
    sums[0, 2], sums[1, 2] = 1, 2
    partial[0, 2], partial[1, 2] = 1, 7
    maxima[:, 3] = 0
    partial[:, 3] = 1e36 if dtype == torch.bfloat16 else 60000
    partial[sums == 0] = 0
    safe_max = torch.where(sums > 0, maxima.double(), -float("inf"))
    best = safe_max.amax(0)
    best[best == -float("inf")] = 0
    weights = sums.double() * (safe_max - best).exp()
    weights /= weights.sum(0).clamp_min(1e-300)
    reference = (weights[..., None] * partial.double()).sum(0).to(dtype)
    pack, combine = Kernel("_pack"), Kernel("_combine")
    width = 264 if raw else 528
    packets = torch.empty((world, world, rows, 2, width), dtype=torch.int32 if raw else dtype)
    for rank in range(world):
        pack[(3,)](
            partial[rank].view(torch.int32) if raw else partial[rank],
            maxima[rank],
            sums[rank],
            packets[rank],
            rows,
            heads,
            2,
            dim,
            raw,
        )
    for rank in range(world):
        calls = []

        def exchange(recv, send, *, group, rank=rank, calls=calls):
            assert group == "dcp" and send.is_contiguous()
            assert send.dtype == packets.dtype
            assert send.numel() * send.element_size() == world * rows * 2 * 1056
            torch.testing.assert_close(send, packets[rank], rtol=0, atol=0)
            recv.copy_(packets[:, rank])
            calls.append(1)

        merge = extract(
            OPS,
            "merge_sfa_dcp",
            {
                "torch": torch,
                "dist": NS(get_world_size=lambda _: world, all_to_all_single=exchange),
                "_pack": pack,
                "_combine": combine,
                "init_device_properties_triton": lambda: None,
                "get_vectorcore_num": lambda: 7,
            },
        )
        actual = merge(partial[rank], maxima[rank], sums[rank], "dcp", raw_packet=raw, combine_tile=tile)
        assert calls == [1]
        torch.testing.assert_close(actual, reference[:, rank * 2 : rank * 2 + 2], rtol=1e-2, atol=1e-3)
        assert not actual[1].any() and torch.all(actual[2] == 5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_raw_packet_preserves_every_halfword_and_full_stat_bits(dtype):
    # Exercise NaN payloads, signed zero and subnormals in raw transport.
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).reshape(8, 16, 512)
    partial = bits.view(dtype)
    maximum = (torch.arange(128, dtype=torch.int32) * 1664525 - 2**30).view(torch.float32).reshape(8, 16)
    sums = (torch.arange(128, dtype=torch.int32) * 1664525 + 2**26).view(torch.float32).reshape(8, 16)
    packet = torch.empty((8, 8, 2, 264), dtype=torch.int32)
    Kernel("_pack")[(7,)](partial.view(torch.int32), maximum, sums, packet, 8, 16, 2, 512, True)
    restored = packet.permute(1, 0, 2, 3).reshape(8, 16, 264)
    assert torch.equal(restored.view(dtype)[..., :512].view(torch.int16), bits)
    assert torch.equal(restored[..., 256], maximum.view(torch.int32))
    assert torch.equal(restored[..., 257], sums.view(torch.int32))
    assert not packet[..., 258:].any()


@pytest.mark.parametrize("tile", [0, 1, 64, 127, 1024])
def test_invalid_tile_fails_before_collective(tile):
    merge = extract(OPS, "merge_sfa_dcp", {"torch": torch, "dist": NS(get_world_size=lambda _: 8)})
    with pytest.raises(ValueError, match="combine tile"):
        merge(torch.zeros(2, 8, 512), None, None, None, combine_tile=tile)


def async_ops(gather_queries, events):
    namespace = dict(gather_queries.__globals__)
    namespace.update(dataclass=dataclass)
    definitions(OPS, ["SfaDcpQueryGather"], namespace)

    def collective(output, packet, *, group, async_op):
        assert group == "dcp" and async_op
        events.append("submit")
        source, destination = weakref.ref(packet), weakref.ref(output)

        def wait():
            events.append("wait")
            # Only SfaDcpQueryGather holds the DMA operands at this point.
            assert source() is not None and destination() is not None
            destination().copy_(torch.cat((source(), source()), dim=0))

        return NS(wait=wait)

    namespace["dist"] = NS(all_gather_into_tensor=collective)
    for name in ("start_sfa_dcp_query_gather", "execute_sfa_dcp"):
        extract(OPS, name, namespace)
    return namespace


def test_early_gather_retains_dma_operands_and_waits_once(gather_queries):
    events = []
    namespace = async_ops(gather_queries, events)
    group = NS(device_group="dcp", world_size=2)
    query, rope = torch.ones(2, 8, 512), torch.ones(2, 8, 64)
    context = namespace["start_sfa_dcp_query_gather"](query, rope, group)
    del query, rope
    gc.collect()
    assert events == ["submit"]
    context.wait()
    context.wait()  # cleanup after successful attention must not add a wait
    assert events == ["submit", "wait"]
    assert torch.equal(context.gathered, context.packet.repeat(2, 1, 1))


@pytest.mark.parametrize("raw,tile", [(False, 512), (True, 512), (False, 128), (True, 128)])
def test_early_attention_uses_one_gather_and_keeps_prefetch_order(gather_queries, monkeypatch, raw, tile):
    events = []
    namespace = async_ops(gather_queries, events)
    group = NS(device_group="dcp", world_size=2)
    query, rope = torch.ones(2, 8, 512, dtype=torch.bfloat16), torch.ones(2, 8, 64, dtype=torch.bfloat16)
    context = namespace["start_sfa_dcp_query_gather"](query, rope, group)
    events.append("independent_prepare")

    def native(q, r, *args):
        assert q.is_contiguous() and r.is_contiguous() and q.shape == (2, 16, 512)
        assert q.eq(1).all() and r.eq(1).all()
        events.append("attention")
        return q, r, None

    def merge(*args, **kwargs):
        events.append("merge")
        assert kwargs == ({"raw_packet": raw, "combine_tile": tile} if raw or tile != 512 else {})
        return "output"

    namespace["merge_sfa_dcp"] = merge
    monkeypatch.setattr(torch.ops._C_ascend, "npu_sparse_flash_attention_dcp", native, raising=False)
    metadata = torch.zeros(1, dtype=torch.int32)
    edge = NS(launch=lambda *args: events.append("prefetch"))
    assert (
        namespace["execute_sfa_dcp"](
            query,
            rope,
            (None, None),
            None,
            None,
            metadata,
            metadata,
            metadata,
            1.0,
            group,
            query_gather=context,
            raw_packet=raw,
            combine_tile=tile,
            prefetch=(edge, None, None, None, None),
        )
        == "output"
    )
    context.wait()
    assert events == ["submit", "independent_prepare", "wait", "attention", "prefetch", "merge"]


@pytest.mark.parametrize("failure", [None, "prepare", "retrieve", "post"])
@pytest.mark.parametrize("eligible", [False, True])
def test_full_operator_owns_early_gather_through_failure_cleanup(monkeypatch, failure, eligible):
    events, pending = [], []
    output = torch.zeros(2, 8, 512)
    counts = torch.ones(1, dtype=torch.int32)

    def start(query, rope, group):
        assert group == "dcp"
        events.append("gather")
        pending.append(NS(wait=lambda: events.append("join")))
        return pending[-1]

    module = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    module.start_sfa_dcp_query_gather = start
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def pre(*args, query_ready=None, **kwargs):
        events.append("project")
        assert (query_ready is not None) == eligible
        if query_ready:
            query_ready(output, output)
        events.append("prepare")
        if failure == "prepare":
            raise RuntimeError(failure)
        return output, output, output, output, counts, output

    def retrieve(*args):
        events.append("retrieve")
        if failure == "retrieve":
            raise RuntimeError(failure)

    def post(*args, **kwargs):
        events.append("post")
        assert kwargs.get("query_gather") is (pending[0] if eligible else None)
        if failure == "post":
            raise RuntimeError(failure)

    impl = NS(
        shared_resident_plan=NS(active=True),
        _dcp_early_gather=True,
        _staged_graph_content_diagnostic_enabled=lambda _: False,
        cross_layer_graph_pre=pre,
        cross_layer_lmcache_retrieve=retrieve,
        cross_layer_graph_post=post,
        _shared_retrieval_edges=lambda: ({}, {}),
    )
    run = extract(
        ROOT / "vllm_ascend/ops/mla.py",
        "sfa_forward_full",
        {
            "get_forward_context": lambda: NS(sfa_full_graph_active=eligible),
            "_mla_runtime_state": lambda _: (impl, "layer", (), NS(sfa_dcp=object())),
            "get_dcp_group": lambda: "dcp",
        },
    )
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            run(output, False, output, "layer", "next", [], [])
    else:
        run(output, False, output, "layer", "next", [], [])
    assert events[:3] == ["project", "gather", "prepare"] if eligible else events[:2] == ["project", "prepare"]
    assert events[-1] == "join" if eligible else "join" not in events


@pytest.fixture
def tuning_bench(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "benchmarks/ops"))
    return load_module(ROOT / "benchmarks/ops/benchmark_sfa_dcp_tuning.py", "dcp_tuning_bench", monkeypatch)


@pytest.mark.parametrize(
    "variant,raw,tile,early",
    [
        ("raw", True, 512, False),
        ("tile128", False, 128, False),
        ("tile256", False, 256, False),
        ("early", False, 512, True),
        ("raw_tile128", True, 128, False),
        ("combined", True, 128, True),
    ],
)
def test_benchmark_candidates_isolate_each_change(tuning_bench, monkeypatch, variant, raw, tile, early):
    monkeypatch.setattr(tuning_bench, "AttentionPipeline", lambda *args, **kwargs: kwargs)
    baseline = tuning_bench.tuning_pipeline(NS(variant=variant), None, None, None, False)
    candidate = tuning_bench.tuning_pipeline(NS(variant=variant), None, None, None, True)
    assert baseline == {"direct": True, "dcp_options": {}, "early_gather": False}
    assert candidate["direct"] is True
    assert candidate["early_gather"] == early
    assert candidate["dcp_options"].get("raw_packet", False) == raw
    assert candidate["dcp_options"].get("combine_tile", 512) == tile


@pytest.mark.parametrize("fail_prepare", [False, True])
def test_whole_pipeline_overlaps_only_independent_work_and_joins(monkeypatch, tuning_bench, fail_prepare):
    pipeline = tuning_bench.AttentionPipeline.__new__(tuning_bench.AttentionPipeline)
    events = []
    tensor = torch.zeros(2, 8, 512)
    pending = []
    namespace = {"dataclass": dataclass}
    definitions(OPS, ["SfaDcpQueryGather"], namespace)

    def start(*args):
        events.append("gather")
        context = namespace["SfaDcpQueryGather"](tensor, tensor, NS(wait=lambda: events.append("wait")), "group")
        pending.append(context)
        return context

    def prepare(frame):
        events.append("plan")
        if fail_prepare:
            raise RuntimeError("prepare failed")
        return (tensor,) * 6

    def attend(*args, query_gather, prefetch, raw_packet, combine_tile):
        assert raw_packet and combine_tile == 128
        query_gather.wait()
        events.append("attention")
        if prefetch is not None:
            prefetch[0].launch(*prefetch[1:])
        events.append("merge")
        return tensor

    module = ModuleType("vllm_ascend.ops.triton.sfa_dcp")
    module.start_sfa_dcp_query_gather, module.execute_sfa_dcp = start, attend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    pipeline.args, pipeline.group = NS(dcp_size=8), "group"
    pipeline.direct = pipeline.early_gather = True
    pipeline.dcp_options = {"raw_packet": True, "combine_tile": 128}
    pipeline.queries = [(tensor, tensor)] * 2
    pipeline.layer_caches = [(tensor, tensor)] * 2
    pipeline.bridges = [()] * 2
    pipeline.transfers = [NS(load=lambda *args: events.append("load"))] * 2
    pipeline.edges = [
        NS(join=lambda: events.append("join"), launch=lambda *args: events.append("prefetch"), destinations=())
    ]
    pipeline.counts = pipeline.attention_table = pipeline.attention_ends = pipeline.local_lens = tensor
    pipeline.prepare = prepare
    if fail_prepare:
        with pytest.raises(RuntimeError, match="prepare failed"):
            pipeline.step(None)
        assert events == ["gather", "plan", "wait"]
    else:
        pipeline.step(None)
        assert events == [
            "gather",
            "plan",
            "load",
            "wait",
            "attention",
            "prefetch",
            "merge",
            "join",
            "gather",
            "wait",
            "attention",
            "merge",
        ]
    assert all(context.work is None for context in pending)
