# SPDX-License-Identifier: Apache-2.0
"""Check the race probe's oracle, raw DMA layout and failure reporting on CPU.

The CPU execution below validates the diagnostic, not NPU stream behavior.
"""

import ast
import contextlib
import ctypes
import importlib.util
import json
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


@pytest.mark.parametrize(
    "history,queries",
    [(499712, [378, 385, 383]), (8192, [128, 127, 129]), (8192, [512, 511, 513])],
)
def test_topk_is_causal_distinct_and_spans_real_history(history, queries):
    topk = PROBE.selected_positions(torch, history, 2048, queries)
    cursor = 0
    for q in queries:
        part = topk[cursor : cursor + q]
        assert part.shape == (q, 2048)
        assert torch.equal(part[:, -1], history + torch.arange(q))
        assert part[:, :-1].min() >= 0
        assert part[:, :-1].max() < history
        assert part[0, :-1].unique().numel() == 2047
        assert part[0, :-1].max() > history * 0.98
        cursor += q


@pytest.mark.parametrize("queries", [[128, 127, 129], [512, 511, 513]])
def test_short_history_fixture_and_bank_reuse_fingerprint(production_classes, queries):
    args = NS(queries=queries, history=8192, topk=2048, heads=64, banks=4, layers=12)
    fixture = PROBE.Fixture(cpu_runtime(production_classes), args)
    assert fixture.indices.shape == (sum(queries), 1, args.topk)
    cursor = 0
    for request, q in enumerate(queries):
        # Check both ends of every request, including its last causal KV row.
        selected = fixture.indices_cpu[[cursor, cursor + q - 1]]
        expected = [
            PROBE.expected_attention(torch, selected, layer, request, column_count=8) for layer in range(args.layers)
        ]
        for layer in range(args.layers):
            explicit = (
                PROBE.values(torch, selected.long().flatten() + layer * PROBE.LAYER_SHIFT, request, width=8)
                .float()
                .reshape(2, args.topk, 8)
                .mean(1)
            )
            torch.testing.assert_close(expected[layer], explicit)
            # A stale or prematurely overwritten bank must fail the same
            # BF16-aware comparator used by the numerical probe.
            for other in range(layer % args.banks, args.layers, args.banks):
                if other != layer:
                    assert PROBE.compare(torch, expected[other].bfloat16(), expected[layer])["different_values"] > 0
        cursor += q


def test_topk_rejects_insufficient_distinct_history():
    with pytest.raises(ValueError, match="at least topk - 1"):
        PROBE.selected_positions(torch, 2046, 2048, [1])
    selected = PROBE.selected_positions(torch, 2047, 2048, [1])
    assert selected.unique().numel() == 2048


def test_cli_rejects_insufficient_history_before_loading_npu(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["prefill_sfa_race.py", "--history", "2046", "--topk", "2048"])
    with pytest.raises(SystemExit) as error:
        PROBE.main()
    assert error.value.code == 2
    assert "history must contain at least topk - 1" in capsys.readouterr().err


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


def cpu_runtime(production_classes, damage=None, structured=False):
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
        output = torch.empty_like(kw["query"][:, :1]) if structured else torch.empty_like(kw["query"])
        begin = 0
        for r, end in enumerate(kw["actual_seq_lengths_query"].tolist()):
            if structured:
                # Exact zero-query attention from the actual bank payloads.
                # Read the common history once instead of once per head/token.
                selected = indices[begin:end].long()
                assert torch.count_nonzero(kw["query"][begin:end]) == 0
                assert torch.equal(selected[:, :-1], selected[:1, :-1].expand(end - begin, -1))
                prefix, tail = selected[0, :-1], selected[:, -1]
                history = key[table[r, prefix // PROBE.BLOCK].long(), prefix % PROBE.BLOCK, 0].float()
                current = key[table[r, tail // PROBE.BLOCK].long(), tail % PROBE.BLOCK, 0].float()
                mean = (history.sum(0)[None, :] + current) / selected.shape[1]
                output[begin:end] = mean[:, None, :].to(output.dtype)
            else:
                for i in range(begin, end):
                    selected = indices[i].long()
                    kv = key[table[r, selected // PROBE.BLOCK].long(), selected % PROBE.BLOCK, 0].float()
                    result = ((kw["query"][i].float() @ kv.T) * kw["scale_value"]).softmax(-1) @ kv
                    output[i] = result.to(output.dtype)
            begin = end
        if damage == "sfa":
            output.fill_(-5)
        if structured:
            output = output.expand_as(kw["query"])
        sfa_refs.append(weakref.ref(output))
        return output

    def projection(operand, weight):
        device.actions.append(("projection", len(sfa_refs) - 1))
        if structured:
            # Evaluate the actual one-hot weight tensor as a sparse projection.
            # Do not call the diagnostic's expected_projected oracle.
            assert torch.all(torch.count_nonzero(weight, dim=1) == 1)
            columns = weight.abs().argmax(dim=1)
            coefficients = weight.gather(1, columns[:, None, :]).squeeze(1)
            result = (
                operand.gather(2, columns[None, :, :].expand(operand.shape[0], -1, -1)) * coefficients[None, :, :]
            ).reshape(operand.shape[0], -1)
        else:
            result = torch.bmm(operand.transpose(0, 1), weight).transpose(0, 1).reshape(operand.shape[0], -1)
        assert result.untyped_storage().data_ptr() != operand.untyped_storage().data_ptr()
        if damage == "projection":
            result.fill_(7)
        return result

    class StructuredCPUOps:
        def __getattr__(self, name):
            return getattr(torch, name)

        def mm(self, a, b):
            # All query-weight columns are identical in this fixture.
            assert torch.equal(b, b[:, :1].expand_as(b))
            column = torch.mm(a.float(), b[:, :1].float()).to(a.dtype)
            return column.expand(a.shape[0], b.shape[1])

    return NS(
        torch=StructuredCPUOps() if structured else torch,
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


@pytest.mark.parametrize(
    "mode",
    ["serial", "event", "p_sfa", "p_use", "p_vup", "after_bank_wait", "before_vup", "p_vup_hold", "hold_sfa"],
)
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


def test_structured_cpu_operators_match_dense_cpu_operators(production_classes):
    dense = cpu_runtime(production_classes)
    structured = cpu_runtime(production_classes, structured=True)
    a = torch.arange(24, dtype=torch.float32).reshape(3, 8).bfloat16()
    b = torch.arange(8, dtype=torch.float32)[:, None].expand(8, 16).bfloat16()
    torch.testing.assert_close(structured.torch.mm(a, b), dense.torch.mm(a, b), rtol=0, atol=0)
    key = torch.arange(2 * PROBE.BLOCK * 16, dtype=torch.float32).reshape(2, PROBE.BLOCK, 1, 16).bfloat16()
    kw = dict(
        query=torch.zeros(3, 2, 16, dtype=torch.bfloat16),
        key=key,
        sparse_indices=torch.tensor([[[0, 3, 129]], [[0, 3, 130]], [[131, 135, 8]]]),
        block_table=torch.tensor([[1, 0], [0, 1]]),
        actual_seq_lengths_query=torch.tensor([2, 3]),
        scale_value=0.125,
    )
    torch.testing.assert_close(structured.sfa(**kw), dense.sfa(**kw), rtol=0, atol=0)
    operand = torch.arange(3 * 2 * 16, dtype=torch.float32).reshape(3, 2, 16).bfloat16()
    weight = torch.zeros(2, 16, 4, dtype=torch.bfloat16)
    weight[0, [2, 7, 10, 15], torch.arange(4)] = 1
    weight[1, [1, 3, 5, 9], torch.arange(4)] = -1
    torch.testing.assert_close(
        structured.projection(operand, weight), dense.projection(operand, weight), rtol=0, atol=0
    )


@pytest.mark.parametrize("queries", [[128, 127, 129], [512, 511, 513]], ids=["small", "large"])
def test_exact_server_commands_on_cpu(production_classes, queries, monkeypatch, tmp_path, capsys):
    report = tmp_path / "results with spaces.json"
    modes = ["p_use", "p_vup", "before_vup", "p_vup_hold"]
    monkeypatch.setattr(
        "sys.argv",
        [
            "prefill_sfa_race.py",
            "--device",
            "0",
            "--history",
            "8192",
            "--layers",
            "12",
            "--banks",
            "4",
            "--heads",
            "64",
            "--queries",
            *map(str, queries),
            "--repeat",
            "2",
            "--modes",
            *modes,
            "--output",
            str(report),
        ],
    )
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "0")
    rt = cpu_runtime(production_classes, structured=True)
    monkeypatch.setattr(PROBE, "load_runtime", lambda args: rt)
    assert PROBE.main() == 0
    results = json.loads(report.read_text(encoding="utf-8"))
    assert [r["mode"] for r in results] == ["serial", *modes, *modes]
    assert [r["repeat"] for r in results[1:]] == [0] * 4 + [1] * 4
    checked = 12 * (sum(queries) * 64 * PROBE.V_WIDTH + (sum(queries) + 3 * 8) * PROBE.WIDTH)
    for result in results:
        assert result["passed"] and result["failures"] == 0
        assert result["sfa_shape"] == [sum(queries), 64, PROBE.WIDTH]
        assert result["vup_shape"] == [sum(queries), 64 * PROBE.V_WIDTH]
        assert result["checked_values"] == checked
        assert not result["observe_inputs"]
    records = [
        json.loads(line.removeprefix("[SFA_RACE] "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[SFA_RACE] ")
    ]
    assert sum(r["event"] == "result" for r in records) == 9
    assert records[-1]["event"] == "complete"
    assert records[-1]["status"] == "selected_modes_passed"
    assert records[-1]["failed_modes"] == []
    assert all(ref() is None for ref in rt.sfa_refs)


@pytest.mark.parametrize("failure", ["control", "selected", "event"])
def test_command_failure_exit_and_json(production_classes, failure, monkeypatch, tmp_path, capsys):
    report = tmp_path / "failure.json"
    mode = "event" if failure == "event" else "p_vup"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prefill_sfa_race.py",
            "--history",
            "2048",
            "--layers",
            "5",
            "--banks",
            "2",
            "--heads",
            "2",
            "--topk",
            "8",
            "--queries",
            "2",
            "3",
            "1",
            "--repeat",
            "1",
            "--modes",
            mode,
            "--output",
            str(report),
        ],
    )
    monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "0")
    rt = cpu_runtime(production_classes)
    project = rt.projection
    calls = 0

    def damage_projection(operand, weight):
        nonlocal calls
        calls += 1
        result = project(operand, weight)
        if failure == "control" or calls > 5:
            result.fill_(7)
        return result

    rt.projection = damage_projection
    monkeypatch.setattr(PROBE, "load_runtime", lambda args: rt)
    assert PROBE.main() == (2 if failure == "control" else 1)
    results = json.loads(report.read_text(encoding="utf-8"))
    records = [
        json.loads(line.removeprefix("[SFA_RACE] "))
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[SFA_RACE] ")
    ]
    if failure == "control":
        assert len(results) == 1 and not results[0]["passed"]
        assert records[-1]["event"] == "control_failed"
    else:
        assert results[0]["passed"] and not results[1]["passed"]
        assert results[1]["first_failures"][0]["stage"] == "vup_output"
        assert records[-1]["status"] == ("reproduced" if failure == "event" else "selected_modes_failed")
        assert records[-1]["failed_modes"] == [mode]
        assert len(results) == (3 if failure == "event" else 2)
        if failure == "event":
            assert results[2]["observe_inputs"] and not results[2]["passed"]


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


@pytest.mark.parametrize("mode,owners_at_fence", [("p_vup", 0), ("p_vup_hold", 1)])
def test_bounded_output_retention_changes_only_owner_lifetime(production_classes, mode, owners_at_fence):
    rt = cpu_runtime(production_classes)
    args = NS(queries=[2, 3, 1], history=2048, topk=8, heads=2, banks=2, layers=5)
    fixture = PROBE.Fixture(rt, args)
    alive_at_fence = []
    original_sync = rt.npu.current.synchronize

    def inspect_fence():
        alive_at_fence.append(sum(ref() is not None for ref in rt.sfa_refs))
        original_sync()

    rt.npu.current.synchronize = inspect_fence
    rt.npu.actions.clear()
    result = fixture.run(mode)
    assert result["passed"]
    assert result["retain_sfa_until_vup"] == (mode == "p_vup_hold")
    assert alive_at_fence == [owners_at_fence] * args.layers
    assert rt.alive_at_sfa == [0] * args.layers
    assert all(ref() is None for ref in rt.sfa_refs)
    assert sum(action[0] == "stream_sync" for action in rt.npu.actions) == args.layers
    assert sum(action[0] == "device_sync" for action in rt.npu.actions) == 2
    assert rt.npu.event_syncs == 0


def test_used_events_have_only_the_production_bank_owners(production_classes):
    rt = cpu_runtime(production_classes)
    args = NS(queries=[2, 3, 1], history=2048, topk=8, heads=2, banks=2, layers=7)
    result = PROBE.Fixture(rt, args).run("event")
    assert result["passed"], result
    assert rt.alive_use_at_sfa == [min(layer, args.banks) for layer in range(args.layers)]
    assert all(ref() is None for ref in rt.npu.event_refs)


@pytest.mark.parametrize(
    "mode",
    ["event", "p_sfa", "p_use", "p_vup", "after_bank_wait", "before_vup", "p_vup_hold", "hold_sfa"],
)
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
        if mode in ("p_use", "after_bank_wait", "before_vup"):
            sync = ("event_sync", recorded[2])
            assert sync in between
            if mode == "p_use":
                assert between.index(sync) < between.index(wait)
            else:
                assert between.index(sync) > between.index(wait)
            if mode == "before_vup":
                assert between[-1] == sync
        else:
            assert not any(action[0] == "event_sync" for action in between)
        if mode == "p_sfa":
            assert actions[sfa - 1] == ("stream_sync", 0)
        if mode in ("p_vup", "p_vup_hold"):
            assert actions[projected + 1] == ("stream_sync", 0)
        if layer == args.layers - 1:
            assert between[-2 if mode == "before_vup" else -1][0] == "save"
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
