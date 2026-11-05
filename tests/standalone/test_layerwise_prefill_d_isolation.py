# SPDX-License-Identifier: Apache-2.0
"""Exercise production P gates without loading the Ascend runtime."""

import ast
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend"


def method(path, cls, name):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


def execute(nodes, namespace):
    tree = ast.parse("from __future__ import annotations")
    tree.body.extend(deepcopy(nodes))
    exec(compile(ast.fix_missing_locations(tree), "<production-D-isolation>", "exec"), namespace)


def assigns(node, name):
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    return any(isinstance(target, ast.Name) and target.id == name for target in targets)


def calls(node, name):
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == name for n in ast.walk(node)
    )


def initialized_shrink(p_node, configured, *, builder=False, unbundle=True):
    owner = "AscendSFAMetadataBuilder" if builder else "AscendSFAImpl"
    init = method("attention/sfa_v1.py", owner, "__init__")
    assignment = next(
        node
        for node in init.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "dsa_shrink_latent" for target in node.targets)
    )
    obj = NS(_layerwise_prefill_p_node=p_node, dsa_offload_unbundle=unbundle)
    execute(
        [assignment],
        dict(
            self=obj,
            envs=NS(
                VLLM_ASCEND_DSA_UNBUNDLE=unbundle,
                VLLM_ASCEND_DSA_SHRINK_LATENT=configured,
            ),
        ),
    )
    return obj


@pytest.mark.parametrize("builder", [False, True])
@pytest.mark.parametrize("p_node", [False, True])
@pytest.mark.parametrize("unbundle", [False, True])
@pytest.mark.parametrize("configured", [0, 1, 2, 3])
def test_compact_scratch_policy_excludes_only_banked_p(builder, p_node, unbundle, configured):
    obj = initialized_shrink(p_node, configured, builder=builder, unbundle=unbundle)
    assert obj.dsa_shrink_latent == (configured if unbundle and not p_node else 0)


@pytest.mark.parametrize("p_node", [False, True])
@pytest.mark.parametrize("mlapo", [False, True])
@pytest.mark.parametrize("has_indexer", [False, True])
@pytest.mark.parametrize("decode_rows", [0, 1, 2, 4])
def test_bank_waits_precede_p_qkv_and_preserve_d_order(p_node, mlapo, has_indexer, decode_rows):
    # Single-token prefill tails and P-side MTP can be classified as decode.
    # Execute the shared wait gate, actual QKV dispatch and both old wait sites.
    forward = method("attention/sfa_v1.py", "AscendSFAImpl", "forward")
    dispatch_index = next(
        i
        for i, node in enumerate(forward.body)
        if isinstance(node, ast.If) and "self.enable_mlapo" in ast.unparse(node.test)
    )
    early_wait = forward.body[dispatch_index - 1]
    assert ast.unparse(early_wait.test) == "self._layerwise_prefill_p_node"
    dispatch = deepcopy(forward.body[dispatch_index])
    first_cache_write = next(i for i, node in enumerate(dispatch.orelse) if calls(node, "exec_kv"))
    dispatch.orelse = dispatch.orelse[:first_cache_write]
    index_write = next(
        node
        for node in forward.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "kv_cache is not None and self.has_indexer"
    )
    late_index_wait = deepcopy(index_write)
    late_index_wait.body = [
        next(
            node
            for node in index_write.body
            if isinstance(node, ast.If) and "wait_for_kv_layer_from_connector" in ast.unparse(node)
        )
    ]

    obj = initialized_shrink(p_node, 2)
    obj.enable_mlapo = mlapo
    obj.has_indexer = has_indexer
    obj.q_lora_rank = 1
    obj.kv_lora_rank = 1
    obj.qk_rope_head_dim = 1
    obj.q_a_layernorm = lambda tensor: tensor
    operations = []

    def wait(layer_name):
        assert layer_name not in operations, "Duplicate wait advances a retriever twice"
        operations.append(layer_name)

    def prepare_qkv(kind):
        if p_node:
            assert operations == (["latent", "indexer"] if has_indexer else ["latent"])
        operations.append(kind)

    def project(tensor):
        prepare_qkv("native_qkv")
        return (NS(split=lambda *args, **kwargs: (object(), object())),)

    project.weight = object()
    obj.fused_qkv_a_proj = project

    def preprocess(**kwargs):
        prepare_qkv("mlapo_qkv")
        return 1, 2, 3, 4

    obj._sfa_preprocess_with_mlapo = preprocess

    def indexer_preprocess(**kwargs):
        operations.append("indexer_qkv")
        return object(), None

    obj.indexer_select_pre_process = indexer_preprocess
    namespace = dict(
        self=obj,
        attn_metadata=NS(num_decode_tokens=decode_rows),
        layer_name="latent",
        index_layer_name="indexer",
        index_lmcache_enabled=has_indexer,
        hidden_states=object(),
        kv_cache=object(),
        cos=object(),
        sin=object(),
        slot_mapping=object(),
        num_input_tokens=4,
        MLAPO_MAX_SUPPORTED_TOKENS=16,
        kv_hash_trace=False,
        _dsa_prof=NS(section=lambda name: nullcontext()),
        get_weight_prefetch_method=lambda: NS(maybe_prefetch_mla_or_sla_weight_in_current_stream=lambda **kwargs: None),
        wait_for_kv_layer_from_connector=wait,
    )
    execute([early_wait, dispatch, late_index_wait], namespace)
    expected = ["mlapo_qkv" if mlapo else "native_qkv"]
    if has_indexer:
        expected.append("indexer_qkv")
    if p_node:
        expected = (["latent", "indexer"] if has_indexer else ["latent"]) + expected
    else:
        if not mlapo and decode_rows == 0:
            expected.append("latent")
        if has_indexer:
            expected.append("indexer")
    assert operations == expected


@pytest.mark.parametrize("p_node,remaps", [(False, True), (True, False)])
def test_p_does_not_remap_full_bank_topk_to_decode_scratch(p_node, remaps):
    obj = initialized_shrink(p_node, 2)
    forward = method("attention/sfa_v1.py", "AscendSFAImpl", "forward")
    compact = next(
        node
        for node in forward.body
        if isinstance(node, ast.If) and "attn_metadata.need_sparse_lmcache_payload" in ast.unparse(node.test)
    )
    # Execute the production dispatch condition; no fake NPU remap implementation.
    branch = deepcopy(compact)
    branch.body = ast.parse("selected.append('decode_scratch')").body
    selected = []
    execute(
        [branch],
        dict(
            self=obj,
            selected=selected,
            attn_metadata=NS(
                split_boundary=object(),
                num_decode_tokens=2,
                need_sparse_lmcache_payload=True,
            ),
        ),
    )
    assert bool(selected) is remaps


@pytest.mark.parametrize("p_node", [False, True])
def test_mlapo_uses_same_preprocessing_after_shared_wait_gate(p_node):
    forward = method("attention/sfa_v1.py", "AscendSFAImpl", "forward")
    outer = next(n for n in forward.body if isinstance(n, ast.If) and "self.enable_mlapo" in ast.unparse(n.test))
    dispatch = outer.body[0]
    seen = []

    def preprocess(**kwargs):
        seen.append(("legacy", kwargs))
        return 1, 2, 3, 4

    obj = NS(_layerwise_prefill_p_node=p_node, _sfa_preprocess_with_mlapo=preprocess)
    ns = dict(
        self=obj,
        hidden_states=object(),
        kv_cache=object(),
        cos=object(),
        sin=object(),
        slot_mapping=object(),
        num_input_tokens=1,
        layer_name="layer",
        attn_metadata=object(),
    )
    execute([dispatch], ns)
    assert [entry[0] for entry in seen] == ["legacy"]
    assert ns["ql_nope"] == 2
    assert "layer_name" not in seen[0][1]


@pytest.mark.parametrize("p_node", [False, True])
def test_input_preparation_uses_single_legacy_table_on_d(p_node):
    prepare = method("worker/model_runner_v1.py", "NPUModelRunner", "_prepare_inputs")
    nodes = [
        node
        for node in prepare.body
        if isinstance(node, ast.If)
        and ast.unparse(node.test) == "self.layerwise_prefill_p_node"
        and (calls(node, "commit_block_table") or calls(node, "compute_slot_mapping"))
    ]
    assert len(nodes) == 2
    seen = []

    def table(name):
        return NS(
            commit_block_table=lambda count: seen.append((name, "blocks", count)),
            compute_slot_mapping=lambda requests, positions: seen.append((name, "slots", requests, positions)),
            commit_slot_mapping=lambda count: seen.append((name, "commit", count)),
        )

    primary, shadow = table("primary"), table("shadow")
    obj = NS(layerwise_prefill_p_node=p_node, input_batch=NS(block_table=primary))
    if p_node:
        obj._refresh_layerwise_prefill_block_tables = lambda: (primary, shadow)
    execute(nodes, dict(self=obj, num_reqs=1, req_indices="req", positions_np="pos", total_num_scheduled_tokens=3))
    assert seen == (
        [
            ("primary", "blocks", 1),
            ("shadow", "blocks", 1),
            ("primary", "slots", "req", "pos"),
            ("primary", "commit", 3),
            ("shadow", "slots", "req", "pos"),
            ("shadow", "commit", 3),
        ]
        if p_node
        else [("primary", "blocks", 1), ("primary", "slots", "req", "pos"), ("primary", "commit", 3)]
    )


@pytest.mark.parametrize("p_node", [False, True])
def test_shared_pool_only_p_reuses_layer_tensor_objects(p_node):
    reshape = method("worker/model_runner_v1.py", "NPUModelRunner", "_reshape_kv_cache_tensors")
    gate = next(
        n
        for n in ast.walk(reshape)
        if isinstance(n, ast.If) and ast.unparse(n.test).startswith("self.dsa_shared_pool and")
    )
    function = ast.parse("def reshape(self, layers):\n    for layer_name in layers:\n        pass\n").body[0]
    function.body[0].body = [gate]
    records = []

    def views(*args, is_indexer, **kwargs):
        result = (object(),)
        records.append((is_indexer, result))
        return result

    caches = {}
    ns = dict(
        reshape_dsa_shared_pool_raw=views,
        raws=[object()],
        spec=NS(dtype="dtype"),
        bs=128,
        nh=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        kv_caches=caches,
        layer_c8=False,
        dsa_shared_views={},
    )
    execute([function], ns)
    ns["reshape"](
        NS(
            dsa_shared_pool=True,
            layerwise_prefill_p_node=p_node,
            use_sparse_c8_indexer=False,
            _mixed_indexer_c8_names=None,
            kv_cache_config=NS(),
        ),
        ["layer.0.attn", "layer.1.attn", "layer.1.indexer"],
    )
    assert [index for index, _ in records] == ([False, True] if p_node else [False, False, True])
    assert (caches["layer.0.attn"] is caches["layer.1.attn"]) is p_node


@pytest.mark.parametrize(
    "path,cls,v2",
    [
        ("core/recompute_scheduler.py", "RecomputeScheduler", False),
        ("core/recompute_scheduler.py", "RecomputeScheduler", True),
        ("core/scheduler_dynamic_batch.py", "SchedulerDynamicBatch", False),
        ("patch/platform/patch_balance_schedule.py", "BalanceScheduler", False),
        ("patch/platform/patch_balance_schedule.py", "BalanceScheduler", True),
    ],
)
@pytest.mark.parametrize("p_node", [False, True])
def test_scheduler_d_constructs_baseline_payload_directly(path, cls, v2, p_node):
    schedule = method(path, cls, "schedule")
    start = next(i for i, n in enumerate(schedule.body) if assigns(n, "layerwise_prefill"))
    nodes = schedule.body[start : start + 2]
    seen = []
    req = NS(request_id="r", _all_token_ids=[10, 11])
    blocks = NS(get_block_ids=lambda: ("physical",))
    obj = NS(use_v2_model_runner=v2, kv_cache_manager=NS(coordinator=NS(layerwise_prefill_p_node=p_node)))
    if p_node:
        obj._make_new_request_data = lambda *args: seen.append(("banked", args)) or args
    ns = dict(
        self=obj,
        NewRequestData=NS(from_request=lambda *args: seen.append(("legacy", args)) or args),
        scheduled_new_reqs=[req],
        scheduled_resumed_reqs=[],
        req_to_new_blocks={"r": blocks},
    )
    execute(nodes, ns)
    assert seen == [
        (
            "banked" if p_node else "legacy",
            (req, blocks if p_node else ("physical",), *([req._all_token_ids] if v2 else [])),
        )
    ]
    assert len(ns["new_reqs_data"]) == 1


@pytest.mark.parametrize("p_node,expected", [(False, True), (True, False)])
def test_d_cold_resume_classification_keeps_baseline_without_shrink(p_node, expected):
    route = method("worker/model_runner_v1.py", "NPUModelRunner", "_staged_sfa_local_route")
    start = next(i for i, n in enumerate(route.body) if assigns(n, "possible_cold_resume"))
    ns = dict(
        self=NS(layerwise_prefill_p_node=p_node, dsa_shrink_latent=0),
        np=np,
        is_decode_state=False,
        graph_configured=False,
        num_computed_tokens=[4],
        prompt_lens=[5],
        num_reqs=1,
    )
    execute(route.body[start : start + 2], ns)
    assert ns["possible_cold_resume"] is expected


def test_d_full_graph_callback_check_ignores_p_capability():
    validate = method("worker/model_runner_v1.py", "NPUModelRunner", "_validate_sfa_layerwise_connector_cudagraph_mode")
    node = next(n for n in validate.body if assigns(n, "uses_layerwise_callbacks"))

    class Connector:
        uses_layerwise_model_callbacks = False

        @property
        def supports_layerwise_prefill_transfer_window(self):
            raise AssertionError("P capability was evaluated on D")

    ns = dict(connector=Connector(), layerwise_prefill_p_node=False)
    execute([node], ns)
    assert ns["uses_layerwise_callbacks"] is False
