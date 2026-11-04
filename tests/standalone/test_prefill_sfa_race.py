# SPDX-License-Identifier: Apache-2.0
"""Check the race probe's oracle, raw DMA layout and failure reporting on CPU.

The CPU execution below validates the diagnostic, not NPU stream behavior.
"""

import ast
import contextlib
import ctypes
import importlib.util
import weakref
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
    assert 6 < memory["device_gib_without_workspace"] < 7
    assert memory["audit_extra_device_gib"] < 4
    assert 1 < memory["hold_sfa_extra_device_gib"] < 2


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
        self.index = owner.next_stream
        owner.next_stream += 1

    def wait_event(self, event):
        assert event.recorded
        self.owner.actions.append(("wait", self.index, event.index))

    def synchronize(self):
        self.owner.syncs += 1
        self.owner.actions.append(("stream_sync", self.index))


class Event:
    def __init__(self, owner):
        self.owner = owner
        self.index = owner.next_event
        owner.next_event += 1
        self.recorded = False
        self.stream_index = None
        owner.event_refs.append(weakref.ref(self))

    def record(self, stream):
        self.recorded = True
        self.stream_index = stream.index
        self.owner.actions.append(("record", stream.index, self.index))

    def synchronize(self):
        assert self.recorded
        self.owner.event_syncs += 1
        self.owner.actions.append(("event_sync", self.index))


class CPUDevice:
    def __init__(self):
        self.syncs = 0
        self.event_syncs = 0
        self.next_stream = self.next_event = 0
        self.actions = []
        self.event_refs = []
        self.current = Stream(self)

    def current_stream(self):
        return self.current

    def Stream(self):
        return Stream(self)

    def Event(self):
        return Event(self)

    @contextlib.contextmanager
    def stream(self, stream):
        old, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = old

    def synchronize(self):
        self.syncs += 1
        self.actions.append(("device_sync",))


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
        "synchronize_prefill_vup_output",
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
    sfa_refs, alive_at_sfa, alive_use_at_sfa = [], [], []
    module, fence = production_classes
    namespace = {"torch": NS(npu=device), "prefill_start_timing_enabled": lambda: False}
    exec(compile(ast.fix_missing_locations(module), "<production bank methods>", "exec"), namespace)
    connector = type(
        "ProductionConnector", (), {n.name: namespace[n.name] for n in module.body if isinstance(n, ast.FunctionDef)}
    )

    def dma(copies, direction):
        device.actions.append(("save" if direction else "load", device.current.index))
        for dst, src, count in copies:
            ctypes.memmove(dst, src, count)
            if damage == "save" and direction:
                ctypes.memset(dst, 0, count)
            if damage == "load" and not direction:
                ctypes.memset(dst, 0, count)

    def sfa(**kw):
        device.actions.append(("sfa", len(sfa_refs)))
        alive_at_sfa.append(sum(ref() is not None for ref in sfa_refs))
        alive_use_at_sfa.append(sum(ref() is not None and ref().stream_index == 0 for ref in device.event_refs))
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
        sfa_refs.append(weakref.ref(output))
        return output

    def projection(operand, weight):
        device.actions.append(("projection", len(sfa_refs) - 1))
        result = torch.bmm(operand.transpose(0, 1), weight).transpose(0, 1).reshape(operand.shape[0], -1)
        assert result.untyped_storage().data_ptr() != operand.untyped_storage().data_ptr()
        if damage == "projection":
            result.fill_(7)
        return result

    return NS(
        torch=torch,
        npu=device,
        device="cpu",
        ops=NS(layerwise_prefill_dma_copy=dma),
        connector_cls=connector,
        fence_cls=fence,
        sfa=sfa,
        projection=projection,
        projection_branch="cpu_reference",
        sfa_refs=sfa_refs,
        alive_at_sfa=alive_at_sfa,
        alive_use_at_sfa=alive_use_at_sfa,
    )


@pytest.mark.parametrize("mode", ["serial", "event", "p_sfa", "p_use", "p_vup", "after_bank_wait", "hold_sfa"])
def test_full_probe_with_real_bank_methods_and_cpu_attention(production_classes, mode):
    rt = cpu_runtime(production_classes)
    args = NS(queries=[2, 3, 1], history=2048, topk=8, heads=2, banks=2, layers=5)
    fixture = PROBE.Fixture(rt, args)
    result = fixture.run(mode, observe=True)
    assert result["passed"], result
    assert result["checked_values"] > 0
    assert result["projection_branch"] == "cpu_reference"
    assert result["sfa_shape"] == [6, 2, PROBE.WIDTH]
    assert result["vup_shape"] == [6, 256]


def test_unobserved_event_path_has_no_intermediate_host_fence(production_classes):
    rt = cpu_runtime(production_classes)
    args = NS(queries=[2, 3, 1], history=2048, topk=8, heads=2, banks=2, layers=5)
    fixture = PROBE.Fixture(rt, args)
    baseline = rt.npu.syncs
    result = fixture.run("event", observe=False)
    assert result["passed"]
    assert rt.npu.syncs - baseline == 2  # Setup poison + final drain only.
    assert rt.npu.event_syncs == 0


@pytest.mark.parametrize(
    "damage,stage",
    [
        ("load", "after_h2d_history"),
        ("save", "save_new_kv"),
        ("sfa", "vup_output"),
        ("projection", "vup_output"),
    ],
)
def test_probe_detects_actual_payload_and_output_damage(production_classes, damage, stage):
    rt = cpu_runtime(production_classes, damage=damage)
    args = NS(queries=[2, 3, 1], history=2048, topk=8, heads=2, banks=2, layers=5)
    fixture = PROBE.Fixture(rt, args)
    result = fixture.run("event", observe=True)
    assert not result["passed"]
    assert any(f["stage"] == stage for f in result["first_failures"]), result


def test_projected_oracle_matches_explicit_projection():
    # Nonconstant values distinguish column selection, head signs and BF16 rounding.
    attention = torch.linspace(-0.743, 0.981, 3 * PROBE.WIDTH).reshape(3, PROBE.WIDTH)
    expected = PROBE.expected_projected(torch, attention, 3)
    weight = torch.zeros(3, PROBE.WIDTH, 128, dtype=torch.bfloat16)
    for head in range(3):
        for column in range(128):
            weight[head, column * 3 % PROBE.WIDTH, column] = 1 if head % 2 == 0 else -1
    operand = attention.bfloat16()[:, None, :].expand(-1, 3, -1)
    actual = torch.bmm(operand.transpose(0, 1), weight).transpose(0, 1).reshape(3, -1)
    torch.testing.assert_close(expected.float(), actual.float(), rtol=0, atol=0)


@pytest.mark.parametrize("observe", [False, True])
def test_event_path_does_not_keep_original_sfa_output_alive(production_classes, observe):
    rt = cpu_runtime(production_classes)
    args = NS(queries=[2, 3, 1], history=2048, topk=8, heads=2, banks=2, layers=5)
    result = PROBE.Fixture(rt, args).run("event", observe=observe)
    assert result["passed"], result
    assert rt.alive_at_sfa == [0] * args.layers
    assert all(ref() is None for ref in rt.sfa_refs)


def test_hold_sfa_is_explicit_and_checks_original_output(production_classes):
    rt = cpu_runtime(production_classes, damage="sfa")
    args = NS(queries=[2, 3, 1], history=2048, topk=8, heads=2, banks=2, layers=5)
    result = PROBE.Fixture(rt, args).run("hold_sfa")
    assert rt.alive_at_sfa == list(range(args.layers))
    assert "sfa_output" in result["first_by_stage"]
    assert "vup_output" in result["first_by_stage"]


def test_used_events_have_only_the_production_bank_owners(production_classes):
    rt = cpu_runtime(production_classes)
    args = NS(queries=[2, 3, 1], history=2048, topk=8, heads=2, banks=2, layers=7)
    result = PROBE.Fixture(rt, args).run("event")
    assert result["passed"], result
    assert rt.alive_use_at_sfa == [min(layer, args.banks) for layer in range(args.layers)]
    assert all(ref() is None for ref in rt.npu.event_refs)


@pytest.mark.parametrize("mode", ["event", "p_sfa", "p_use", "p_vup", "after_bank_wait", "hold_sfa"])
def test_fence_and_final_save_order_matches_serving(production_classes, mode):
    rt = cpu_runtime(production_classes)
    args = NS(queries=[2, 3, 1], history=2048, topk=8, heads=2, banks=2, layers=5)
    fixture = PROBE.Fixture(rt, args)
    rt.npu.actions.clear()
    result = fixture.run(mode)
    assert result["passed"], result
    actions = rt.npu.actions
    assert sum(action[0] == "save" for action in actions) == args.layers
    for layer in range(args.layers):
        sfa = actions.index(("sfa", layer))
        projected = actions.index(("projection", layer))
        between = actions[sfa + 1 : projected]
        # One SFA-completion event is recorded, then each bank FIFO joins it.
        recorded = next(action for action in between if action[0] == "record")
        wait = ("wait", 1 + layer % args.banks, recorded[2])
        assert wait in between
        if mode in ("p_use", "after_bank_wait"):
            sync = ("event_sync", recorded[2])
            assert sync in between
            if mode == "p_use":
                assert between.index(sync) < between.index(wait)
            else:
                assert between.index(sync) > between.index(wait)
        else:
            assert not any(action[0] == "event_sync" for action in between)
        if mode == "p_sfa":
            assert actions[sfa - 1] == ("stream_sync", 0)
        if mode == "p_vup":
            assert actions[projected + 1] == ("stream_sync", 0)
        if layer == args.layers - 1:
            assert between[-1][0] == "save"
            assert not any(action[0] == "save" for action in actions[projected + 1 :])
        else:
            next_sfa = actions.index(("sfa", layer + 1))
            assert sum(action[0] == "save" for action in actions[projected + 1 : next_sfa]) == 1


@pytest.fixture(scope="module")
def real_vup_method():
    source = ROOT / "vllm_ascend/attention/sfa_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFAImpl")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_v_up_proj")
    return ast.Module(body=[method], type_ignores=[])


@pytest.mark.parametrize("query_count,expected_branch", [(6, "custom"), (1146, "bmm")])
def test_real_vup_dispatch_preserves_the_independent_oracle(real_vup_method, query_count, expected_branch):
    # Executes serving's actual Python method. Its NPU custom kernel is replaced
    # by CPU BMM here; this test verifies wiring, shapes and the oracle only.
    branches = []

    def cpu_custom(x, weight, output):
        branches.append("custom")
        output.copy_(torch.bmm(x.transpose(0, 1), weight).transpose(0, 1))

    def cpu_bmm(x, weight):
        branches.append("bmm")
        return torch.bmm(x, weight)

    namespace = {
        "torch": NS(
            float16=torch.float16,
            bfloat16=torch.bfloat16,
            empty=torch.empty,
            bmm=cpu_bmm,
            ops=NS(_C_ascend=NS(batch_matmul_transpose=cpu_custom)),
        ),
        "BMM_TRANS_MAX_SUPPORTED_TOKENS": 1024,
    }
    exec(compile(real_vup_method, "<serving _v_up_proj>", "exec"), namespace)
    attention = torch.linspace(-0.71, 1.21, query_count * PROBE.WIDTH).reshape(query_count, PROBE.WIDTH)
    operand = attention.bfloat16()[:, None, :].expand(-1, 2, -1).contiguous()
    weight = torch.zeros(2, PROBE.WIDTH, 128, dtype=torch.bfloat16)
    for head in range(2):
        for column in range(128):
            weight[head, column * 3 % PROBE.WIDTH, column] = 1 if head % 2 == 0 else -1
    module = NS(W_UV=weight, local_num_heads=2, kv_lora_rank=PROBE.WIDTH, v_head_dim=128)
    actual = namespace["_v_up_proj"](module, operand)
    assert branches == [expected_branch]
    assert actual.untyped_storage().data_ptr() != operand.untyped_storage().data_ptr()
    expected = PROBE.expected_projected(torch, attention, 2)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
