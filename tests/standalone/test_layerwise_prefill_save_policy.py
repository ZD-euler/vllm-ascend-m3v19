# SPDX-License-Identifier: Apache-2.0
"""Execute the real runner/SFA transfer control flow without torch or an NPU."""

import ast
from copy import deepcopy
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend"


def source_class(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)


def assigns(node, name):
    if isinstance(node, ast.Assign):
        return any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    return isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name


def execute(nodes, namespace):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *deepcopy(nodes)], type_ignores=[]))
    exec(compile(module, "<real-layerwise-prefill-save-policy>", "exec"), namespace)


@pytest.fixture
def api():
    state = source_class("attention/attention_v1.py", "AscendAttentionState")
    runner = source_class("worker/model_runner_v1.py", "NPUModelRunner")
    classify = next(n for n in runner.body if getattr(n, "name", None) == "_build_attn_state")
    prepare = next(n for n in runner.body if getattr(n, "name", None) == "_prepare_inputs")
    # Keep the scheduler's actual valid-token calculation. A sampled token is
    # not num_valid_tokens: without drafts, valid == scheduled, even on P.
    prepare_start = next(
        i
        for i, n in enumerate(prepare.body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "not scheduler_output.scheduled_spec_decode_tokens"
    )
    prepare_end = next(i for i, n in enumerate(prepare.body) if assigns(n, "attn_state"))
    runner_class = ast.ClassDef(name="Runner", bases=[], keywords=[], body=[classify], decorator_list=[])

    sfa = source_class("attention/sfa_v1.py", "AscendSFAImpl")
    forward = next(n for n in sfa.body if getattr(n, "name", None) == "forward")
    entry_start = next(i for i, n in enumerate(forward.body) if assigns(n, "transfer_context"))
    entry_end = next(i for i, n in enumerate(forward.body) if assigns(n, "pending_transfer_names"))
    pure_decode = next(n for n in forward.body if assigns(n, "_is_pure_decode"))
    save_start = next(i for i, n in enumerate(forward.body) if assigns(n, "save_operations"))
    projection_start = next(
        i
        for i, n in enumerate(forward.body)
        if assigns(n, "attn_output") and ast.unparse(n.value) == "self._v_up_proj(attn_output)"
    )
    tail = next(
        n
        for n in forward.body
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "use_layerwise_transfer_window"
    )
    profile_return = next(
        n for n in forward.body if isinstance(n, ast.If) and ast.unparse(n.test) == "attn_metadata is None"
    )
    sfa_start = next(
        i for i, n in enumerate(forward.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "attn_output is None"
    )
    # Execute the actual transfer entry, post-SFA handoff, save policy and
    # projection tail together. Device math contributes stream-order markers.
    control = ast.parse(
        "def control(self, attn_metadata, layer_name, kv_cache, index_layer_name, "
        "index_lmcache_enabled, attn_output, output):\n    pass\n"
    ).body[0]
    control.body = [
        profile_return,
        *forward.body[entry_start : entry_end + 1],
        pure_decode,
        *ast.parse("stream_work('SFA', layer_name)").body,
        *forward.body[sfa_start + 1 : save_start],
        *forward.body[save_start:projection_start],
        *ast.parse("stream_work('v_up', layer_name)").body,
        tail,
    ]
    helper_names = {
        "_submit_sfa_save_operations",
        "_submit_sfa_transfer_window_save_operations",
        "_submit_sfa_post_transfer_window_save_operations",
        "_submit_sfa_layerwise_transfer_window",
        "_finish_sfa_layerwise_transfer_window",
    }
    helpers = [n for n in sfa.body if getattr(n, "name", None) in helper_names]
    assert len(helpers) == len(helper_names)
    impl_class = ast.ClassDef(name="Impl", bases=[], keywords=[], body=[*helpers, control], decorator_list=[])
    context = NS(additional_kwargs={})
    events, operations, bank_events = [], [], []

    class Event:
        def record(self, stream):
            assert stream is operations
            self.predecessors = tuple(stream)

    def handoff(name, event):
        bank_events.append((name, event))
        events.append(("use", name))
        operations.append(("use", name))

    def record(kind):
        def callback(name, *args):
            events.append((kind, name))
            operations.append((kind, name))
            return True

        return callback

    namespace = dict(
        np=np,
        sfa_full_graph_enabled=lambda config: False,
        Enum=Enum,
        torch=NS(npu=NS(Event=Event, current_stream=lambda: operations)),
        get_kv_transfer_group=lambda: NS(record_layerwise_prefill_bank_use=handoff),
        stream_work=lambda kind, name: operations.append((kind, name)),
        kv_hash_trace=False,
        get_forward_context=lambda: context,
        maybe_save_kv_layer_to_connector=record("legacy_save"),
        maybe_save_kv_layer_in_layerwise_prefill_transfer_window=record("save"),
        maybe_save_kv_layer_outside_layerwise_prefill_transfer_window=record("post_save"),
        maybe_submit_layerwise_prefill_load=record("load"),
        maybe_finish_layerwise_prefill_save=record("finish"),
    )
    utils = ast.parse((ROOT / "attention/utils.py").read_text(encoding="utf-8"))
    marker = next(n for n in utils.body if getattr(n, "name", None) == "record_layerwise_prefill_bank_use")
    execute([state, runner_class, marker, impl_class], namespace)
    return NS(
        namespace=namespace,
        states=namespace["AscendAttentionState"],
        Runner=namespace["Runner"],
        Impl=namespace["Impl"],
        prepare_nodes=prepare.body[prepare_start : prepare_end + 1],
        context=context,
        events=events,
        operations=operations,
        bank_events=bank_events,
    )


def classify(api, *, computed, scheduled, drafts=0, mtp=True):
    runner = api.Runner()
    runner.vllm_config = NS()
    runner.input_batch = NS(req_ids=["r"], num_computed_tokens_cpu=np.array([computed]))
    runner.speculative_config = NS(method="mtp") if mtp else None
    runner.scheduler_config = NS(enable_chunked_prefill=True)
    namespace = dict(
        api.namespace,
        self=runner,
        num_reqs=1,
        num_scheduled_tokens=np.array([scheduled]),
        scheduler_output=NS(
            num_scheduled_tokens={"r": scheduled},
            scheduled_spec_decode_tokens={"r": list(range(drafts))} if drafts else {},
        ),
    )
    execute(api.prepare_nodes, namespace)
    assert namespace["num_valid_tokens"].tolist() == [scheduled - drafts]
    return namespace["attn_state"]


def impl(api, *, p_node=True, window=0, shrink=2):
    obj = api.Impl()
    obj._layerwise_prefill_p_node = p_node
    obj._layerwise_prefill_mtp = True
    obj._first_layerwise_prefill_layer_index = 0
    obj._last_layerwise_prefill_layer_index = 77
    obj.layer_name = None
    obj.dsa_shrink_latent = shrink
    obj.dsa_offload_unbundle = True
    obj.enable_dsa_cp_with_o_proj_tp = False
    obj.enable_dsa_cp_with_layer_shard = False
    obj.o_proj = lambda value: (api.operations.append(("o_proj", obj.control_layer_name)) or value,)
    api.namespace["_decode_window_save_window_size"] = lambda: window
    api.namespace["layerwise_prefill_transfer_window_supported"] = lambda: p_node
    return obj


def step(api, obj, state, *, layer=0, has_indexer=True, decode_rows=0):
    name = f"model.layers.{layer}.self_attn.attn"
    indexer = f"model.layers.{layer}.self_attn.indexer.k_cache" if has_indexer else None
    caches = [object(), object(), object()]
    obj.control_layer_name = name
    obj.control(
        NS(attn_state=state, num_decode_tokens=decode_rows),
        name,
        caches,
        indexer,
        has_indexer,
        np.ones(1),
        np.zeros(1),
    )
    return [name, indexer] if has_indexer else [name]


@pytest.mark.parametrize(
    "computed,scheduled,mtp,expected",
    [
        (0, 8, True, "PrefillNoCache"),
        (4096, 8, True, "ChunkedPrefill"),
        (4096, 2, True, "ChunkedPrefill"),
        (4096, 1, True, "SpecDecoding"),
        (4096, 1, False, "DecodeOnly"),
    ],
)
@pytest.mark.parametrize("has_indexer", [True, False])
def test_p_prefill_keeps_saves_for_real_runner_state(api, computed, scheduled, mtp, expected, has_indexer):
    state = classify(api, computed=computed, scheduled=scheduled, mtp=mtp)
    assert state == getattr(api.states, expected)
    obj = impl(api)
    # These are prompt rows, including the one-token remainder. SFA's builder
    # therefore reports zero decode rows despite the runner's decode enum.
    names = step(api, obj, state, has_indexer=has_indexer, decode_rows=0)
    pending = api.context.additional_kwargs["sfa_layerwise_prefill_pending"]
    assert [name for name, _ in pending] == names
    step(api, obj, state, layer=1, has_indexer=has_indexer, decode_rows=0)
    assert [name for kind, name in api.events if kind == "save"] == names
    assert [name for kind, name in api.events if kind == "load"] == [-1, *names]
    assert [name for kind, name in api.events if kind == "finish"] == names


@pytest.mark.parametrize("num_layers", [1, 3, 78])
@pytest.mark.parametrize("has_indexer", [False, True])
@pytest.mark.parametrize("cached_layer_flags", [False, True])
def test_latent_release_at_next_layer_entry_and_final_projection(api, num_layers, has_indexer, cached_layer_flags):
    obj = impl(api)
    obj._last_layerwise_prefill_layer_index = num_layers - 1
    names = [f"model.layers.{i}.self_attn.attn" for i in range(num_layers)]
    for layer, name in enumerate(names):
        if cached_layer_flags:
            obj.layer_name = name
            obj._last_layerwise_prefill_layer = layer == num_layers - 1
        api.operations.append(("entry", name))
        step(api, obj, api.states.ChunkedPrefill, layer=layer, has_indexer=has_indexer)
        if layer < num_layers - 1:
            # The current bank stays owned throughout the rest of this layer.
            assert [n for n, _ in api.bank_events] == names[:layer]
            api.operations.append(("FFN", name))
            api.operations.append(("norm", names[layer + 1]))

    assert [name for name, _ in api.bank_events] == names
    assert "sfa_layerwise_prefill_pending" not in api.context.additional_kwargs
    for layer, (name, event) in enumerate(api.bank_events):
        assert ("SFA", name) in event.predecessors
        assert ("v_up", name) in event.predecessors
        assert ("o_proj", name) in event.predecessors
        if layer < num_layers - 1:
            assert ("FFN", name) in event.predecessors
            assert ("norm", names[layer + 1]) in event.predecessors
            assert ("entry", names[layer + 1]) in event.predecessors
            assert ("SFA", names[layer + 1]) not in event.predecessors
        assert api.operations.index(("use", name)) < api.operations.index(("save", name))
        assert api.operations.index(("save", name)) < api.operations.index(("load", name))
    # The helper records only the latent source, not the optional index sibling.
    assert all(name.endswith("self_attn.attn") for name, _ in api.bank_events)


def test_separate_prefill_chunks_record_fresh_events_and_leave_no_pending_release(api):
    obj = impl(api)
    obj._last_layerwise_prefill_layer_index = 2
    for chunk in range(2):
        api.operations.append(("chunk", chunk))
        for layer in range(3):
            step(api, obj, api.states.ChunkedPrefill, layer=layer)
        assert "sfa_layerwise_prefill_pending" not in api.context.additional_kwargs
    assert len(api.bank_events) == 6
    for layer in range(3):
        name, first = api.bank_events[layer]
        second_name, second = api.bank_events[layer + 3]
        assert name == second_name and first is not second
        assert ("chunk", 1) not in first.predecessors
        assert ("chunk", 1) in second.predecessors


def test_legacy_p_without_transfer_window_releases_after_projection(api):
    obj = impl(api)
    api.namespace["layerwise_prefill_transfer_window_supported"] = lambda: False
    names = step(api, obj, api.states.ChunkedPrefill)
    assert [name for name, _ in api.bank_events] == names[:1]
    assert ("o_proj", names[0]) in api.bank_events[0][1].predecessors
    assert api.operations.index(("use", names[0])) < api.operations.index(("legacy_save", names[0]))
    assert not api.context.additional_kwargs


def test_duplicate_transfer_names_fail_before_releasing_any_bank(api):
    obj = impl(api)
    with pytest.raises(RuntimeError, match="duplicate KV groups"):
        obj._submit_sfa_layerwise_transfer_window([("layer0", []), ("layer0", [])])
    assert api.bank_events == [] and api.events == []


@pytest.mark.parametrize("has_indexer", [True, False])
def test_one_token_p_tail_flushes_all_target_layers_before_mtp(api, has_indexer):
    state = classify(api, computed=4096, scheduled=1)
    assert state == api.states.SpecDecoding
    obj = impl(api)
    expected = []
    for layer in range(78):
        expected.extend(step(api, obj, state, layer=layer, has_indexer=has_indexer))
    assert "sfa_layerwise_prefill_pending" not in api.context.additional_kwargs
    for kind in ("save", "finish", "post_save"):
        assert [name for event, name in api.events if event == kind] == expected
    assert [name for kind, name in api.events if kind == "load"] == [-1, *expected]
    assert not any(kind == "legacy_save" for kind, _ in api.events)
    assert len(api.bank_events) == 78
    last_name, last_event = api.bank_events[-1]
    assert ("o_proj", last_name) in last_event.predecessors

    # The MTP proposer creates a fresh forward context. Its layer 78 entry
    # must not inherit target pending saves. MTP has no next layer callback;
    # its save must be submitted before this context is discarded.
    mtp_context = NS(additional_kwargs={}, is_draft_model=True)
    api.namespace["get_forward_context"] = lambda: mtp_context
    events_before_mtp = list(api.events)
    mtp_names = step(api, obj, state, layer=78, has_indexer=has_indexer, decode_rows=0)
    assert "sfa_layerwise_prefill_pending" not in mtp_context.additional_kwargs
    mtp_events = api.events[len(events_before_mtp) :]
    for kind in ("save", "finish", "post_save", "load"):
        assert [name for event, name in mtp_events if event == kind] == mtp_names
    assert len(api.bank_events) == 79
    mtp_name, mtp_event = api.bank_events[-1]
    assert mtp_name == mtp_names[0]
    assert ("o_proj", mtp_name) in mtp_event.predecessors


@pytest.mark.parametrize("scheduled,drafts,mtp", [(1, 0, False), (1, 0, True), (2, 1, True)])
@pytest.mark.parametrize("window", [0, 2048])
def test_d_decode_retains_existing_window_save_policy(api, scheduled, drafts, mtp, window):
    state = classify(api, computed=4096, scheduled=scheduled, drafts=drafts, mtp=mtp)
    assert state == (api.states.SpecDecoding if mtp else api.states.DecodeOnly)
    obj = impl(api, p_node=False, window=window)
    names = step(api, obj, state, decode_rows=scheduled)
    expected = [("legacy_save", name) for name in names] if window else []
    assert api.events == expected
    assert not api.context.additional_kwargs


def test_d_without_shrink_still_saves(api):
    state = classify(api, computed=4096, scheduled=1)
    obj = impl(api, p_node=False, shrink=0)
    names = step(api, obj, state, decode_rows=1)
    assert api.events == [("legacy_save", name) for name in names]


def test_d_does_not_consult_p_transfer_capability_or_move_save_policy_before_projection(api):
    obj = impl(api, p_node=False, window=2048)
    del obj._first_layerwise_prefill_layer_index
    del obj._last_layerwise_prefill_layer_index
    del obj._layerwise_prefill_mtp
    order = []
    obj.o_proj = lambda value: (order.append("projection") or value,)
    api.namespace["_decode_window_save_window_size"] = lambda: order.append("save_policy") or 2048

    def unexpected():
        raise AssertionError("D must not consult the P transfer-window capability")

    api.namespace["layerwise_prefill_transfer_window_supported"] = unexpected
    names = step(api, obj, api.states.SpecDecoding, decode_rows=1)
    assert order == ["projection", "save_policy"]
    assert api.events == [("legacy_save", name) for name in names]


def test_no_metadata_profile_forward_never_starts_transfers(api):
    obj = impl(api)
    filled = []
    output = NS(fill_=lambda value: filled.append(value))
    obj.control(None, "model.layers.0.self_attn.attn", [], None, False, None, output)
    assert filled == [0]
    assert api.events == []
    assert not api.context.additional_kwargs
