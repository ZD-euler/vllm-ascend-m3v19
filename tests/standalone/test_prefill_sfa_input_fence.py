# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual SFA dispatch with an asynchronous CPU test stream."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2] / "vllm_ascend"


def compile_function(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.parse("from __future__ import annotations")
    module.body.append(node)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("layer", ["model.layers.0", "model.layers.78", "model.layers.79"])
def test_real_sfa_call_waits_after_inputs_and_before_attention(enabled, layer):
    pending = ["load", "Q", "new KV", "topk"]
    completed, calls = [], []
    query, rope, key, key_rope, indices, table = (object() for _ in range(6))

    def synchronize(name):
        assert name == layer
        completed.extend(pending)
        pending.clear()
        calls.append("fence")

    def sfa(**kwargs):
        assert kwargs["query"] is query and kwargs["query_rope"] is rope
        assert kwargs["key"] is key and kwargs["value"] is key
        assert kwargs["key_rope"] is key_rope
        assert kwargs["sparse_indices"] is indices and kwargs["block_table"] is table
        assert completed == (["load", "Q", "new KV", "topk"] if enabled else [])
        pending.append("SFA")
        calls.append("SFA")
        return "attention output"

    namespace = {
        "torch": NS(ops=NS(_C_ascend=NS(npu_sparse_flash_attention=sfa))),
        "synchronize_prefill_sfa_inputs": synchronize,
    }
    forward = compile_function(ROOT / "attention/sfa_v1.py", "_execute_sparse_flash_attention_process", namespace)
    result = forward(
        NS(dsa_shrink_latent=0, scale=1.0, _prefill_sfa_input_fence=enabled),
        query,
        rope,
        (key, key_rope),
        indices,
        NS(block_table=table),
        None,
        None,
        layer_name=layer,
    )
    assert result == "attention output"
    assert calls == (["fence", "SFA"] if enabled else ["SFA"])
    assert pending[-1] == "SFA"  # The pre-fence never executes the SFA itself.


def test_helper_routes_to_live_connector_and_rejects_missing_api():
    calls = []
    connector = NS(synchronize_prefill_sfa_inputs=calls.append)
    helper = compile_function(
        ROOT / "attention/utils.py",
        "synchronize_prefill_sfa_inputs",
        {"get_kv_transfer_group": lambda: connector},
    )
    helper("layer7")
    assert calls == ["layer7"]
    with pytest.raises(RuntimeError, match="p_sfa requires"):
        helper(None)
    del connector.synchronize_prefill_sfa_inputs
    with pytest.raises(RuntimeError, match="update LMCache"):
        helper("layer7")


@pytest.mark.parametrize("p_node", [False, True])
@pytest.mark.parametrize("mode", ["", "0", "p_load", "p_use", "p_sfa", "p_sfa,check"])
def test_mode_is_cached_at_initialization_and_never_enabled_on_d(p_node, mode):
    tree = ast.parse((ROOT / "attention/sfa_v1.py").read_text(encoding="utf-8"))
    assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(target) == "self._prefill_sfa_input_fence" for target in n.targets)
    )
    impl = NS(_layerwise_prefill_p_node=p_node)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])), "<mode>", "exec"),
        {"self": impl, "envs": NS(LMCACHE_PREFILL_FENCE_DEBUG=mode)},
    )
    assert impl._prefill_sfa_input_fence == (p_node and "p_sfa" in mode.split(","))
