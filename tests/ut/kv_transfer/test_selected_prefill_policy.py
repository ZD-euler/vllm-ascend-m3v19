# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from vllm_ascend.attention.selected_prefill_policy import (
    packed_main_c8_enabled,
    selected_prefill_ineligibility_reason,
    selected_prefill_requested,
)


class TestSelectedPrefillPolicy(unittest.TestCase):
    def setUp(self):
        self.supported = {
            "requested": True,
            "layerwise_prefill_p_node": True,
            "enforce_eager": True,
            "tensor_parallel_size": 1,
            "kv_group_count": 2,
            "dsa_shared_pool": True,
            "is_lmcache_connector": True,
            "main_kv_layout_supported": True,
            "registered_source_ready": True,
            "bank_writer_ready": True,
            "reader_retirement_ready": True,
        }

    def test_only_explicit_true_requests_selected_prefill(self):
        self.assertFalse(selected_prefill_requested({}))
        for value in (False, None, "true", 1):
            with self.subTest(value=value):
                self.assertFalse(selected_prefill_requested({"enable_selected_prefill": value}))
        self.assertTrue(selected_prefill_requested({"enable_selected_prefill": True}))

    def test_native_remains_selected_without_opt_in(self):
        facts = {**self.supported, "requested": selected_prefill_requested({})}
        self.assertEqual(selected_prefill_ineligibility_reason(**facts), "Selected prefill is disabled.")

    def test_each_missing_requirement_retains_native(self):
        unsupported = (
            ("layerwise_prefill_p_node", False, "P node"),
            ("enforce_eager", False, "eager"),
            ("tensor_parallel_size", 8, "parallel size 1"),
            ("kv_group_count", 1, "two KV groups"),
            ("dsa_shared_pool", False, "shared pool"),
            ("is_lmcache_connector", False, "LMCache connector"),
            ("main_kv_layout_supported", False, "main KV layout"),
            ("registered_source_ready", False, "registered LMCache sources"),
            ("bank_writer_ready", False, "bank writer"),
            ("reader_retirement_ready", False, "reader retirement"),
        )
        for field, value, expected_reason in unsupported:
            with self.subTest(field=field):
                reason = selected_prefill_ineligibility_reason(**{**self.supported, field: value})
                self.assertIsNotNone(reason)
                self.assertIn(expected_reason, reason)

    def test_connector_evidence_is_required(self):
        missing_evidence = (
            ("is_lmcache_connector", "LMCache connector"),
            ("main_kv_layout_supported", "main KV layout"),
            ("registered_source_ready", "registered LMCache sources"),
            ("bank_writer_ready", "bank writer"),
            ("reader_retirement_ready", "reader retirement"),
        )
        for field, expected_reason in missing_evidence:
            with self.subTest(field=field):
                facts = {**self.supported}
                del facts[field]
                reason = selected_prefill_ineligibility_reason(**facts)
                self.assertIsNotNone(reason)
                self.assertIn(expected_reason, reason)

    def test_validated_capabilities_allow_step_preparation(self):
        self.assertIsNone(selected_prefill_ineligibility_reason(**self.supported))

    def test_packed_main_marker_defaults_to_disabled(self):
        self.assertFalse(packed_main_c8_enabled(SimpleNamespace()))
        self.assertFalse(packed_main_c8_enabled(SimpleNamespace(cache_sparse_sfa_c8=False)))
        self.assertTrue(packed_main_c8_enabled(SimpleNamespace(cache_sparse_sfa_c8=True)))

    def test_packed_main_marker_rejects_non_boolean_values(self):
        for marker in (0, 1, None, "false"):
            with self.subTest(marker=marker):
                with self.assertRaisesRegex(ValueError, "marker must be a boolean"):
                    packed_main_c8_enabled(SimpleNamespace(cache_sparse_sfa_c8=marker))


class TestRunnerSelectedPrefillGate(unittest.TestCase):
    @staticmethod
    def _gate():
        import ast
        from pathlib import Path

        source = Path(__file__).parents[3] / "vllm_ascend/worker/model_runner_v1.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_validate_selected_prefill_eligibility"
        )
        namespace = {
            "get_kv_transfer_group": lambda: SimpleNamespace(
                supports_dsa_index_lmcache=True,
                supports_selected_prefill_sources=False,
                supports_selected_prefill_bank_writer=False,
                supports_selected_prefill_reader_retirement=False,
            ),
            "has_kv_transfer_group": lambda: True,
            "KVCacheConfig": object,
        }
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        return namespace[method.name]

    def _runner(self, requested):
        return SimpleNamespace(
            ascend_config=SimpleNamespace(enable_selected_prefill=requested),
            layerwise_prefill_p_node=True,
            model_config=SimpleNamespace(enforce_eager=True),
            parallel_config=SimpleNamespace(tensor_parallel_size=1),
            enable_sparse_sfa_c8=True,
            dsa_shared_pool=True,
        )

    def _config(self):
        return SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(cache_sparse_sfa_c8=True)
                )
            ]
        )

    def test_disabled_request_is_a_noop(self):
        self._gate()(self._runner(False), self._config())

    def test_requested_runtime_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "unavailable"):
            self._gate()(self._runner(True), self._config())
