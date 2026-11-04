# SPDX-License-Identifier: Apache-2.0
"""Exercise the production projection tail with asynchronous CPU operations."""

import ast
import weakref
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend"


def projection_tail(namespace):
    tree = ast.parse((ROOT / "attention/sfa_v1.py").read_text(encoding="utf-8"))
    forward = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "forward"
        and any(
            isinstance(child, ast.Assign) and ast.unparse(child.value) == "self._v_up_proj(attn_output)"
            for child in node.body
        )
    )
    start = next(
        i
        for i, node in enumerate(forward.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.value) == "self._v_up_proj(attn_output)"
    )
    wrapper = ast.parse("def run(self, layer_name, output, transfer_context=None, save_operations=()): pass")
    wrapper.body[0].body = ast.parse("attn_output = make_attention_output()").body + forward.body[start:]
    exec(compile(ast.fix_missing_locations(wrapper), "<production-projection-tail>", "exec"), namespace)
    return namespace["run"]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("layer", ["model.layers.0", "model.layers.78", "model.layers.79"])
def test_vup_finishes_before_prefetch_and_oproj_only_when_enabled(enabled, layer):
    pending, completed, calls = ["SFA"], [], []
    result = {}
    references = []

    class AttentionOutput:
        pass

    def make_attention_output():
        value = AttentionOutput()
        references.append(weakref.ref(value))
        return value

    def vup(value):
        assert isinstance(value, AttentionOutput)
        pending.append("v_up")
        calls.append("v_up")
        return "projected output"

    def synchronize(name):
        assert name == layer
        # Match serving's lifetime: the SFA output was released by the
        # assignment, before this diagnostic fence was reached.
        assert references[0]() is None
        completed.extend(pending)
        pending.clear()
        calls.append("fence")

    def prefetch(**kwargs):
        assert kwargs["dependency"] == "projected output"
        assert references[0]() is None
        assert completed == (["SFA", "v_up"] if enabled else [])
        calls.append("prefetch")

    class OProj:
        weight = object()

        def __call__(self, value):
            assert value == "projected output"
            pending.append("o_proj")
            calls.append("o_proj")
            return ("model output",)

    namespace = {
        "make_attention_output": make_attention_output,
        "synchronize_prefill_vup_output": synchronize,
        "get_weight_prefetch_method": lambda: NS(maybe_prefetch_mla_or_sla_weight_in_current_stream=prefetch),
        "MAX_O_PROJ_PREFETCH_SIZE": 1,
        "use_layerwise_transfer_window": True,
        "pending_transfers": None,
        "is_last_transfer_layer": False,
        "get_forward_context": lambda: NS(additional_kwargs={}),
        "_dsa_prof": NS(end=lambda _: None),
        "_sfa_t": None,
        "output_padded": result,
    }
    impl = NS(
        _v_up_proj=vup,
        _prefill_vup_output_fence=enabled,
        enable_dsa_cp_with_o_proj_tp=False,
        enable_dsa_cp_strict_accuracy=False,
        o_proj=OProj(),
    )
    assert projection_tail(namespace)(impl, layer, result) is result
    assert result[Ellipsis] == "model output"
    assert calls == (["v_up", "fence", "prefetch", "o_proj"] if enabled else ["v_up", "prefetch", "o_proj"])
    assert pending == (["o_proj"] if enabled else ["SFA", "v_up", "o_proj"])


@pytest.mark.parametrize("p_node", [False, True])
@pytest.mark.parametrize("mode", ["", "0", "all", "p_load", "p_use", "p_sfa", "p_vup", "p_vup,check"])
def test_cached_selector_only_enables_vup_mode_on_p(p_node, mode):
    tree = ast.parse((ROOT / "attention/sfa_v1.py").read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "self._prefill_vup_output_fence" for target in node.targets)
    )
    impl = NS(_layerwise_prefill_p_node=p_node)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])), "<selector>", "exec"),
        {"self": impl, "envs": NS(LMCACHE_PREFILL_FENCE_DEBUG=mode)},
    )
    assert impl._prefill_vup_output_fence == (p_node and "p_vup" in mode.split(","))


def test_helper_rejects_missing_api_and_propagates_sync_failure():
    tree = ast.parse((ROOT / "attention/utils.py").read_text(encoding="utf-8"))
    helper = next(node for node in tree.body if getattr(node, "name", None) == "synchronize_prefill_vup_output")
    calls = []
    connector = NS(synchronize_prefill_vup_output=calls.append)
    namespace = {"get_kv_transfer_group": lambda: connector}
    module = ast.parse("from __future__ import annotations")
    module.body.append(helper)
    exec(compile(ast.fix_missing_locations(module), "<vup-helper>", "exec"), namespace)
    run = namespace[helper.name]
    run("layer7")
    assert calls == ["layer7"]
    with pytest.raises(RuntimeError, match="p_vup requires"):
        run(None)

    def failed_sync(_):
        raise RuntimeError("device sync failed")

    connector.synchronize_prefill_vup_output = failed_sync
    with pytest.raises(RuntimeError, match="device sync failed"):
        run("layer7")
    del connector.synchronize_prefill_vup_output
    with pytest.raises(RuntimeError, match="update LMCache"):
        run("layer7")
