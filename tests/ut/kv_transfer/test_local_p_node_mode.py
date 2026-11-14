# SPDX-License-Identifier: Apache-2.0
"""The local P-node mode skips connector-only checks, nothing else."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _validator(local_only: bool):
    source = Path(__file__).parents[3] / "vllm_ascend/worker/model_runner_v1.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_sfa_layerwise_connector_cudagraph_mode"
    )
    namespace = {
        "envs_ascend": SimpleNamespace(
            VLLM_ASCEND_LAYERWISE_PREFILL_LOCAL_ONLY=local_only,
        ),
        "has_kv_transfer_group": lambda: False,
        "staged_sfa_graph_configured": lambda config: False,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    runner = SimpleNamespace(
        layerwise_prefill_p_node=True,
        vllm_config=SimpleNamespace(),
        model_config=SimpleNamespace(enforce_eager=True),
        _profiling_cudagraph_memory=False,
    )
    return namespace[method.name], runner


def test_local_only_allows_the_p_node_without_a_connector():
    validate, runner = _validator(local_only=True)
    validate(runner)


def test_local_only_does_not_relax_the_connector_requirement_when_off():
    validate, runner = _validator(local_only=False)
    with pytest.raises(ValueError, match="active KV connector"):
        validate(runner)


def test_local_only_mode_is_not_used_by_default():
    source = Path(__file__).parents[3] / "vllm_ascend/envs.py"
    assert 'os.getenv("VLLM_ASCEND_LAYERWISE_PREFILL_LOCAL_ONLY", "0")' in source.read_text(
        encoding="utf-8"
    )
