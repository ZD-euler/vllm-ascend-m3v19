# SPDX-License-Identifier: Apache-2.0
"""Check the race probe's oracle, raw DMA layout and failure reporting on CPU.

The CPU execution below validates the diagnostic, not NPU stream behavior.
"""

import ast
import contextlib
import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("sfa_race", ROOT / "tools/prefill_sfa_race.py")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_pattern_is_layer_request_and_position_specific():
    positions = torch.tensor([0, 128, 256, 499711])
    base = PROBE.values(torch, positions, 0)
    assert not torch.equal(base, PROBE.values(torch, positions + 4 * PROBE.LAYER_SHIFT, 0))
    assert not torch.equal(base, PROBE.values(torch, positions, 1))
    assert base[0, 0] != base[1, 0]


def test_memory_sources_do_not_scale_as_full_copies_of_each_layer():
    args = NS(queries=[378, 385, 383], history=499712, topk=2048, heads=16, banks=4, layers=79)
    memory = PROBE.memory_plan(args)
    assert memory["pinned_gib"] < 2
    assert 7 < memory["device_gib_without_workspace"] < 9
    assert memory["audit_extra_device_gib"] < 4


def test_topk_is_causal_distinct_and_spans_real_history():
    queries = [378, 385, 383]
    topk = PROBE.selected_positions(torch, 499712, 2048, queries)
    cursor = 0
    for q in queries:
        part = topk[cursor : cursor + q]
        assert part.shape == (q, 2048)
        assert torch.equal(part[:, -1], 499712 + torch.arange(q))
        assert part[:, :-1].max() < 499712
        assert part[0, :-1].unique().numel() == 2047
        assert part[0, :-1].max() > 490000
        cursor += q


def test_oracle_matches_explicit_attention_and_tolerates_roundoff():
    indices = PROBE.selected_positions(torch, 2048, 8, [3])
    expected = PROBE.expected_attention(torch, indices, 5, 2)
    for i in range(3):
        kv = PROBE.values(torch, indices[i].long() + 5 * PROBE.LAYER_SHIFT, 2).float()
        q = torch.zeros(1, 512)
        result = (q @ kv.T).softmax(-1) @ kv
        torch.testing.assert_close(result[0], expected[i])
    assert PROBE.compare(torch, expected.bfloat16(), expected)["different_values"] == 0
    assert PROBE.compare(torch, expected + 0.2, expected)["different_values"] > 0
    poisoned = expected.clone()
    poisoned[1, 3] = float("nan")
    assert PROBE.compare(torch, poisoned, expected)["index"] == [1, 3]


class Stream:
    def __init__(self, owner):
        self.owner = owner

    def wait_event(self, event):
        assert event.recorded

    def synchronize(self):
        self.owner.syncs += 1


class Event:
    recorded = False

    def record(self, stream):
        self.recorded = True

    def synchronize(self):
        assert self.recorded


class CPUDevice:
    def __init__(self):
        self.syncs = 0
        self.current = Stream(self)

    def current_stream(self):
        return self.current

    def Stream(self):
        return Stream(self)

    Event = Event

    @contextlib.contextmanager
    def stream(self, stream):
        old, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = old

    def synchronize(self):
        self.syncs += 1


@pytest.fixture(scope="module")
def production_classes():
    root = ROOT.parent / "LMCache-Ascend"
    if not root.exists():
        pytest.skip("requires sibling LMCache-Ascend checkout")
    names = {
        "_layerwise_prefill_transfer_state",
        "_layerwise_prefill_transfer_generation",
        "_layerwise_prefill_bank",
        "_layerwise_prefill_dma_stream",
        "_require_layerwise_prefill_bank_use",
        "record_layerwise_prefill_bank_use",
        "wait_for_layerwise_prefill_load",
        "synchronize_prefill_sfa_inputs",
    }
    source = root / "lmcache_ascend/v1/npu_connector/npu_connectors.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VLLMPagedMemLayerwiseNPUConnector")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    module = ast.parse("from __future__ import annotations")
    module.body.extend(methods)
    fence_spec = importlib.util.spec_from_file_location("race_fences", root / "lmcache_ascend/v1/kv_fence_debug.py")
    fences = importlib.util.module_from_spec(fence_spec)
    fence_spec.loader.exec_module(fences)
    return module, fences.KVFenceDebug


def cpu_runtime(production_classes, damage=None):
    device = CPUDevice()
    module, fence = production_classes
    namespace = {"torch": NS(npu=device), "prefill_start_timing_enabled": lambda: False}
    exec(compile(ast.fix_missing_locations(module), "<production bank methods>", "exec"), namespace)
    connector = type(
        "ProductionConnector", (), {n.name: namespace[n.name] for n in module.body if isinstance(n, ast.FunctionDef)}
    )

    def dma(copies, direction):
        for dst, src, count in copies:
            ctypes.memmove(dst, src, count)
            if damage == "save" and direction:
                ctypes.memset(dst, 0, count)
            if damage == "load" and not direction:
                ctypes.memset(dst, 0, count)

    def sfa(**kw):
        table, key, indices = kw["block_table"], kw["key"], kw["sparse_indices"][:, 0]
        output = torch.empty_like(kw["query"])
        begin = 0
        for r, end in enumerate(kw["actual_seq_lengths_query"].tolist()):
            for i in range(begin, end):
                selected = indices[i].long()
                kv = key[table[r, selected // PROBE.BLOCK].long(), selected % PROBE.BLOCK, 0].float()
                result = ((kw["query"][i].float() @ kv.T) * kw["scale_value"]).softmax(-1) @ kv
                output[i] = result.to(output.dtype)
            begin = end
        if damage == "sfa":
            output.fill_(-5)
        return output

    return NS(
        torch=torch,
        npu=device,
        device="cpu",
        ops=NS(layerwise_prefill_dma_copy=dma),
        connector_cls=connector,
        fence_cls=fence,
        sfa=sfa,
    )


@pytest.mark.parametrize("mode", ["serial", "event", "p_sfa", "p_use"])
def test_full_probe_with_real_bank_methods_and_cpu_attention(production_classes, mode):
    rt = cpu_runtime(production_classes)
    args = NS(queries=[2, 3, 1], history=2048, topk=8, heads=2, banks=2, layers=5)
    fixture = PROBE.Fixture(rt, args)
    result = fixture.run(mode, observe=True)
    assert result["passed"], result
    assert result["checked_values"] > 0


def test_unobserved_event_path_has_no_intermediate_host_fence(production_classes):
    rt = cpu_runtime(production_classes)
    args = NS(queries=[2, 3, 1], history=2048, topk=8, heads=2, banks=2, layers=5)
    fixture = PROBE.Fixture(rt, args)
    baseline = rt.npu.syncs
    result = fixture.run("event", observe=False)
    assert result["passed"]
    assert rt.npu.syncs - baseline == 2  # Setup poison + final drain only.


@pytest.mark.parametrize(
    "damage,stage", [("load", "after_h2d_history"), ("save", "save_new_kv"), ("sfa", "sfa_output")]
)
def test_probe_detects_actual_payload_and_output_damage(production_classes, damage, stage):
    rt = cpu_runtime(production_classes, damage=damage)
    args = NS(queries=[2, 3, 1], history=2048, topk=8, heads=2, banks=2, layers=5)
    fixture = PROBE.Fixture(rt, args)
    result = fixture.run("event", observe=True)
    assert not result["passed"]
    assert any(f["stage"] == stage for f in result["first_failures"]), result
