# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AscendSFAMetadataBuilder FlashComm1 DSA-CP optimizations."""

import sys
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import torch
from vllm.config import set_current_vllm_config

from tests.ut.attention.utils import patch_distributed_groups
from vllm_ascend.attention.attention_v1 import AscendAttentionState

sfa_v1: Any = None
AscendSFAMetadata: Any = None
AscendSFAMetadataBuilder: Any = None
_SENTINEL = object()
_ORIG_TORCH_NPU_INDUCTOR: Any = _SENTINEL


def setUpModule() -> None:
    global sfa_v1, AscendSFAMetadata, AscendSFAMetadataBuilder, _ORIG_TORCH_NPU_INDUCTOR
    _ORIG_TORCH_NPU_INDUCTOR = sys.modules.get("torch_npu._inductor", _SENTINEL)
    if _ORIG_TORCH_NPU_INDUCTOR is _SENTINEL:
        sys.modules["torch_npu._inductor"] = MagicMock()
    import vllm_ascend.attention.sfa_v1 as _sfa_v1
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadata as _Meta
    from vllm_ascend.attention.sfa_v1 import AscendSFAMetadataBuilder as _Builder

    sfa_v1 = _sfa_v1
    AscendSFAMetadata = _Meta
    AscendSFAMetadataBuilder = _Builder


def tearDownModule() -> None:
    if _ORIG_TORCH_NPU_INDUCTOR is _SENTINEL:
        sys.modules.pop("torch_npu._inductor", None)
    else:
        sys.modules["torch_npu._inductor"] = _ORIG_TORCH_NPU_INDUCTOR


class TestAscendSFADSACPPNode(unittest.TestCase):
    @staticmethod
    def _make_vllm_config(*, max_seqs: int = 8, max_tokens: int = 64):
        cfg = MagicMock()
        cfg.cache_config.block_size = 128
        cfg.model_config.max_model_len = 2048
        cfg.model_config.get_head_size.return_value = 64
        cfg.model_config.dtype = torch.bfloat16
        cfg.model_config.hf_text_config.qk_rope_head_dim = 64
        cfg.model_config.hf_text_config.topk_tokens = 2048
        cfg.speculative_config = None
        cfg.parallel_config.tensor_parallel_size = 4
        cfg.parallel_config.pipeline_parallel_size = 1
        cfg.parallel_config.prefill_context_parallel_size = 1
        cfg.parallel_config.decode_context_parallel_size = 1
        cfg.parallel_config.cp_kv_cache_interleave_size = 1
        cfg.kv_transfer_config = SimpleNamespace(is_kv_producer=True)
        cfg.scheduler_config.max_num_seqs = max_seqs
        cfg.scheduler_config.max_num_batched_tokens = max_tokens
        return cfg

    @staticmethod
    def _make_builder(
        layer_names: list[str],
        *,
        max_seqs: int = 8,
        max_tokens: int = 64,
        c8_factor: int = 1,
        full_graph: bool = False,
    ) -> AscendSFAMetadataBuilder:
        vcfg = TestAscendSFADSACPPNode._make_vllm_config(max_seqs=max_seqs, max_tokens=max_tokens)
        if full_graph:
            vcfg.kv_transfer_config = None
        ascend_cfg = SimpleNamespace(
            indexer_c8_shared_block_factor=c8_factor,
            indexer_hbm_block_map=None,
            indexer_c8_layer_mask=lambda names: [True for _ in names],
        )
        env_patch = (
            patch.dict("os.environ", {"VLLM_ASCEND_DSA_UNBUNDLE": "1", "VLLM_ASCEND_DSA_SHRINK_LATENT": "2"})
            if full_graph
            else patch.dict("os.environ", {})
        )
        with (
            env_patch,
            set_current_vllm_config(vcfg),
            patch.object(sfa_v1, "get_current_vllm_config", return_value=vcfg),
            patch.object(sfa_v1, "get_ascend_config", return_value=ascend_cfg),
            patch.object(sfa_v1, "enable_dsa_cp", return_value=True),
        ):
            builder = AscendSFAMetadataBuilder(
                kv_cache_spec=MagicMock(),
                layer_names=layer_names,
                vllm_config=vcfg,
                device=torch.device("cpu"),
            )
        builder.enable_dsa_cp = True
        if full_graph:
            builder._layerwise_prefill_p_node = False
        return builder

    @staticmethod
    def _make_common(
        *,
        qsl: list[int],
        seq_lens: list[int],
        num_actual_tokens: int,
        num_input_tokens: int,
        attn_state: AscendAttentionState = AscendAttentionState.PrefillNoCache,
        slot_offset: int = 100,
        block_offset: int = 1,
        with_cpu_qsl: bool = True,
        with_indexer: bool = False,
        sfa_full_graph: bool = False,
    ) -> SimpleNamespace:
        num_reqs = len(seq_lens)
        qsl_t = torch.tensor(qsl, dtype=torch.int32)
        seq_t = torch.tensor(seq_lens, dtype=torch.int32)
        slots = torch.arange(slot_offset, slot_offset + num_input_tokens, dtype=torch.int32)
        blocks = torch.arange(block_offset, block_offset + num_reqs * 4, dtype=torch.int32).reshape(num_reqs, 4)
        return SimpleNamespace(
            num_reqs=num_reqs,
            num_actual_tokens=num_actual_tokens,
            num_input_tokens=num_input_tokens,
            block_table_tensor=blocks,
            slot_mapping=slots,
            positions=torch.arange(num_input_tokens, dtype=torch.long),
            indexer_block_table_tensor=blocks + 1000 if with_indexer else None,
            indexer_slot_mapping=slots + 1000 if with_indexer else None,
            prompt_lens_cpu=None,
            request_ids=[f"req{i}" for i in range(num_reqs)],
            query_start_loc=qsl_t,
            query_start_loc_cpu=qsl_t.clone() if with_cpu_qsl else None,
            seq_lens=seq_t,
            seq_lens_cpu=seq_t.clone(),
            attn_state=attn_state,
            sfa_full_graph=sfa_full_graph,
        )

    @staticmethod
    def _no_item_patch(*tensors: torch.Tensor):
        orig_item = torch.Tensor.item
        guarded_storages = {t.untyped_storage().data_ptr() for t in tensors}

        def _checked_item(self_t: torch.Tensor):
            if self_t.untyped_storage().data_ptr() in guarded_storages:
                raise AssertionError("Unexpected synchronous Tensor.item() on cum_query_lens / seq_lens")
            return orig_item(self_t)

        return patch.object(torch.Tensor, "item", _checked_item)

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_dsa_cp_build_avoids_item_calls_across_4_ranks_straddling_empty_padding_and_full_graph(self) -> None:
        builder = self._make_builder(["layer0", "layer1"])
        cos_sin = lambda pos, *_a, **_k: (
            torch.zeros((pos.shape[0], 1, 1, 64), dtype=torch.bfloat16),
            torch.zeros((pos.shape[0], 1, 1, 64), dtype=torch.bfloat16),
        )

        # 1) Multi-request straddling batch across all 4 TP ranks:
        #    qsl=[0, 3, 10, 14], seq_lens=[100, 200, 300], actual=14, input=16 -> 4 tokens/rank
        expected_straddle = {
            0: ([3, 4, 4], [100, 194, 0], 0, 4, 4),
            1: ([0, 4, 4], [0, 198, 0], 4, 8, 8),
            2: ([0, 2, 4], [0, 200, 298], 8, 12, 12),
            3: ([0, 0, 2], [0, 0, 300], 12, 14, 16),
        }
        with patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=cos_sin):
            for rank in range(4):
                common = self._make_common(
                    qsl=[0, 3, 10, 14],
                    seq_lens=[100, 200, 300],
                    num_actual_tokens=14,
                    num_input_tokens=16,
                )
                tp_grp = SimpleNamespace(world_size=4, rank_in_group=rank)
                with (
                    patch.object(sfa_v1, "get_tp_group", return_value=tp_grp),
                    self._no_item_patch(common.query_start_loc, common.seq_lens),
                ):
                    meta = builder.build(0, common)
                exp_q, exp_k, exp_ls, exp_le, exp_lep = expected_straddle[rank]
                ctx = meta.dsa_cp_context
                assert ctx is not None
                self.assertEqual(ctx.actual_seq_lengths_query.tolist(), exp_q)
                self.assertEqual(ctx.actual_seq_lengths_key.tolist(), exp_k)
                self.assertEqual((ctx.local_start, ctx.local_end, ctx.local_end_with_pad), (exp_ls, exp_le, exp_lep))

        # 2) Short batch with empty rank slices (actual=2, input=2, TP=4 -> pad=4, 1 token/rank)
        with patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=cos_sin):
            for rank, exp_q, exp_k in ((0, [1], [9]), (1, [1], [10]), (2, [0], [0]), (3, [0], [0])):
                common_short = self._make_common(
                    qsl=[0, 2],
                    seq_lens=[10],
                    num_actual_tokens=2,
                    num_input_tokens=2,
                )
                with (
                    patch.object(
                        sfa_v1, "get_tp_group", return_value=SimpleNamespace(world_size=4, rank_in_group=rank)
                    ),
                    self._no_item_patch(common_short.query_start_loc, common_short.seq_lens),
                ):
                    meta = builder.build(0, common_short)
                ctx = meta.dsa_cp_context
                assert ctx is not None
                self.assertEqual(ctx.actual_seq_lengths_query.tolist(), exp_q)
                self.assertEqual(ctx.actual_seq_lengths_key.tolist(), exp_k)
                self.assertEqual(ctx.slot_mapping_cp.tolist(), [100 + rank] if rank < 2 else [-1])

        # 3) SP padding (num_input_tokens=7 > num_actual_tokens=5, TP=4 -> num_tokens_pad=8, 2/rank)
        with patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=cos_sin):
            common_sp = self._make_common(
                qsl=[0, 5],
                seq_lens=[20],
                num_actual_tokens=5,
                num_input_tokens=7,
                slot_offset=10,
            )
            with (
                patch.object(sfa_v1, "get_tp_group", return_value=SimpleNamespace(world_size=4, rank_in_group=3)),
                self._no_item_patch(common_sp.query_start_loc, common_sp.seq_lens),
            ):
                meta_sp = builder.build(0, common_sp)
            ctx_sp = meta_sp.dsa_cp_context
            assert ctx_sp is not None
            self.assertEqual((ctx_sp.num_tokens, ctx_sp.num_tokens_pad, ctx_sp.local_end), (7, 8, 5))
            self.assertEqual(meta_sp.slot_mapping.tolist(), [10, 11, 12, 13, 14, 15, 16, -1])
            self.assertEqual(ctx_sp.slot_mapping_cp.tolist(), [16, -1])

        # 4) sfa_full_graph=True (appends dummy segment to q_ends_cpu without .item())
        builder_fg = self._make_builder(["layer0"], max_seqs=4, max_tokens=16)
        builder_fg._layerwise_prefill_p_node = False
        with patch.dict(
            "os.environ",
            {"VLLM_ASCEND_DSA_UNBUNDLE": "1", "VLLM_ASCEND_DSA_SHRINK_LATENT": "2"},
        ):
            vcfg_fg = self._make_vllm_config(max_seqs=4, max_tokens=16)
            vcfg_fg.kv_transfer_config = None
            ascend_cfg = SimpleNamespace(indexer_c8_shared_block_factor=1)
            with (
                set_current_vllm_config(vcfg_fg),
                patch.object(sfa_v1, "get_current_vllm_config", return_value=vcfg_fg),
                patch.object(sfa_v1, "get_ascend_config", return_value=ascend_cfg),
                patch.object(sfa_v1, "enable_dsa_cp", return_value=True),
            ):
                builder_fg = AscendSFAMetadataBuilder(
                    kv_cache_spec=MagicMock(),
                    layer_names=["layer0"],
                    vllm_config=vcfg_fg,
                    device=torch.device("cpu"),
                )
            builder_fg.enable_dsa_cp = True
        with patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=cos_sin):
            common_fg = self._make_common(
                qsl=[0, 3, 5],
                seq_lens=[10, 20],
                num_actual_tokens=5,
                num_input_tokens=8,
                with_indexer=True,
                sfa_full_graph=True,
            )
            with (
                patch.object(sfa_v1, "get_tp_group", return_value=SimpleNamespace(world_size=4, rank_in_group=2)),
                self._no_item_patch(common_fg.query_start_loc, common_fg.seq_lens),
            ):
                meta_fg = builder_fg.build(0, common_fg)
            ctx_fg = meta_fg.dsa_cp_context
            assert ctx_fg is not None
            # Rank 2 owns [4, 6): req1 covers [4, 5) (1 token) + full-graph pad [5, 6)
            self.assertEqual(ctx_fg.actual_seq_lengths_query.tolist(), [0, 1])
            self.assertEqual(ctx_fg.actual_seq_lengths_key.tolist(), [0, 20])

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_dsa_cp_build_fallback_when_cpu_lengths_incomplete(self) -> None:
        builder = self._make_builder(["layer0"])
        cos_sin = lambda pos, *_a, **_k: (
            torch.zeros((pos.shape[0], 1, 1, 64), dtype=torch.bfloat16),
            torch.zeros((pos.shape[0], 1, 1, 64), dtype=torch.bfloat16),
        )
        common_incomplete = self._make_common(
            qsl=[0, 3, 8],
            seq_lens=[12, 24],
            num_actual_tokens=8,
            num_input_tokens=8,
        )
        # Short seq_lens_cpu (len < num_segs) triggers the .item() fallback path cleanly.
        common_incomplete.seq_lens_cpu = torch.tensor([12], dtype=torch.int32)
        with (
            patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=cos_sin),
            patch.object(sfa_v1, "get_tp_group", return_value=SimpleNamespace(world_size=4, rank_in_group=1)),
        ):
            meta_fb = builder.build(0, common_incomplete)
        ctx_fb = meta_fb.dsa_cp_context
        assert ctx_fb is not None
        self.assertEqual(ctx_fb.actual_seq_lengths_query.tolist(), [1, 2])
        self.assertEqual(ctx_fb.actual_seq_lengths_key.tolist(), [12, 20])

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_dsa_cp_build_fallback_when_query_start_loc_cpu_is_none(self) -> None:
        builder = self._make_builder(["layer0"])
        cos_sin = lambda pos, *_a, **_k: (
            torch.zeros((pos.shape[0], 1, 1, 64), dtype=torch.bfloat16),
            torch.zeros((pos.shape[0], 1, 1, 64), dtype=torch.bfloat16),
        )
        common_fallback = self._make_common(
            qsl=[0, 3, 8],
            seq_lens=[12, 24],
            num_actual_tokens=8,
            num_input_tokens=8,
            with_cpu_qsl=False,
        )
        with (
            patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=cos_sin),
            patch.object(sfa_v1, "get_tp_group", return_value=SimpleNamespace(world_size=4, rank_in_group=1)),
        ):
            meta_fb = builder.build(0, common_fallback)
        ctx_fb = meta_fb.dsa_cp_context
        assert ctx_fb is not None
        # Rank 1 owns [2, 4): req0 has [2, 3) (offset=0, k=12), req1 has [3, 4) (offset=4, k=20, cum_q=2)
        self.assertEqual(ctx_fb.actual_seq_lengths_query.tolist(), [1, 2])
        self.assertEqual(ctx_fb.actual_seq_lengths_key.tolist(), [12, 20])

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_dsa_cp_rebind_78_layers_2_banks_caches_pad_and_handles_indexer_and_guards(self) -> None:
        builder = self._make_builder([f"layer{i}" for i in range(78)], c8_factor=1)
        builder._layerwise_prefill_p_node = True
        cos_sin = lambda pos, *_a, **_k: (
            torch.zeros((pos.shape[0], 1, 1, 64), dtype=torch.bfloat16),
            torch.zeros((pos.shape[0], 1, 1, 64), dtype=torch.bfloat16),
        )

        # SP-padded input (num_input_tokens=6, TP=4 -> num_tokens_pad=8) triggers slot_mapping pad once per bank!
        common_b0 = self._make_common(
            qsl=[0, 6],
            seq_lens=[32],
            num_actual_tokens=6,
            num_input_tokens=6,
            slot_offset=10,
            block_offset=1,
            with_indexer=True,
        )
        common_b1 = self._make_common(
            qsl=[0, 6],
            seq_lens=[32],
            num_actual_tokens=6,
            num_input_tokens=6,
            slot_offset=200,
            block_offset=100,
            with_indexer=False,
        )

        orig_pad = torch.nn.functional.pad
        pad_1d_calls: list[tuple[int, int]] = []

        def _counting_pad(tensor: torch.Tensor, *args, **kwargs):
            if tensor.ndim == 1:
                pad_1d_calls.append((int(tensor[0]), int(tensor.shape[0])))
            return orig_pad(tensor, *args, **kwargs)

        tp_grp = SimpleNamespace(world_size=4, rank_in_group=2)  # owns [4, 6)
        with (
            patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=cos_sin),
            patch.object(sfa_v1, "get_tp_group", return_value=tp_grp),
            patch.object(sfa_v1.nn.functional, "pad", side_effect=_counting_pad),
        ):
            template = builder.build(0, common_b0)
            self.assertEqual(len(pad_1d_calls), 1)
            metas: list[AscendSFAMetadata] = [template]
            for layer_idx in range(1, 78):
                common_l = common_b1 if layer_idx % 2 == 1 else common_b0
                rebound = builder.rebind_layerwise_prefill_metadata(template, common_l)
                assert rebound is not None
                metas.append(rebound)

        # Layer 0 (build) padded bank 0 once; Layer 1 (rebind) padded bank 1 once; Layers 2..77 hit cache!
        self.assertEqual(pad_1d_calls, [(10, 6), (200, 6)])
        self.assertEqual(len({id(m) for m in metas}), 78)
        for layer_idx, m in enumerate(metas):
            expected_bank = common_b1 if layer_idx % 2 == 1 else common_b0
            expected_base_slot = 200 if layer_idx % 2 == 1 else 10
            self.assertTrue(torch.equal(m.block_table, expected_bank.block_table_tensor[:1]))
            self.assertEqual(
                m.slot_mapping.tolist(), list(range(expected_base_slot, expected_base_slot + 6)) + [-1, -1]
            )
            assert m.dsa_cp_context is not None
            self.assertEqual(
                m.dsa_cp_context.slot_mapping_cp.tolist(), [expected_base_slot + 4, expected_base_slot + 5]
            )
            if layer_idx % 2 == 1:
                self.assertIsNone(m.indexer_block_table)
                self.assertIsNone(m.indexer_slot_mapping)
                self.assertIsNone(m.indexer_c8_block_table)
                self.assertIsNone(m.indexer_c8_slot_mapping)
            else:
                assert m.indexer_block_table is not None and m.indexer_slot_mapping is not None
                self.assertTrue(torch.equal(m.indexer_block_table, common_b0.indexer_block_table_tensor[:1]))
                self.assertTrue(torch.equal(m.indexer_slot_mapping, common_b0.indexer_slot_mapping[:6]))

        # MixedIndexerMetadata update on rebind clones indexer_c8 views cleanly per bank
        builder_c8 = self._make_builder(["layer0", "layer1"], c8_factor=2)
        builder_c8._layerwise_prefill_p_node = True
        with (
            patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=cos_sin),
            patch.object(sfa_v1, "get_tp_group", return_value=tp_grp),
        ):
            t_c8 = builder_c8.build(0, common_b0)
            common_b1_idx = self._make_common(
                qsl=[0, 6],
                seq_lens=[32],
                num_actual_tokens=6,
                num_input_tokens=6,
                slot_offset=200,
                block_offset=5,
                with_indexer=True,
            )
            r_c8 = builder_c8.rebind_layerwise_prefill_metadata(t_c8, common_b1_idx)
        assert r_c8 is not None and r_c8.indexer_c8_block_table is not None and t_c8.indexer_c8_block_table is not None
        self.assertNotEqual(r_c8.indexer_c8_block_table.data_ptr(), t_c8.indexer_c8_block_table.data_ptr())
        self.assertNotEqual(r_c8.indexer_c8_slot_mapping.data_ptr(), t_c8.indexer_c8_slot_mapping.data_ptr())

        # Fail-closed None returns
        for bad_t_attr, bad_c_attr in (
            ({"attn_state": AscendAttentionState.DecodeOnly}, {}),
            ({"num_decode_tokens": 1}, {}),
            ({"dsa_cp_context": None}, {}),
            ({}, {"sfa_full_graph": True}),
            ({}, {"num_reqs": 2}),
            ({}, {"num_input_tokens": 7}),
            ({}, {"num_actual_tokens": 5}),
        ):
            t_bad = SimpleNamespace(**{**vars(template), **bad_t_attr})
            c_bad = SimpleNamespace(**{**vars(common_b1), **bad_c_attr})
            self.assertIsNone(builder.rebind_layerwise_prefill_metadata(t_bad, c_bad))

    @staticmethod
    def _det_cos_sin(pos: torch.Tensor, *_a: Any, **_k: Any) -> tuple[torch.Tensor, torch.Tensor]:
        base = pos.to(torch.float32)[:, None, None, None].expand(-1, 1, 1, 64)
        return base.to(torch.bfloat16).clone(), (base + 1.0).to(torch.bfloat16).clone()

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_dsa_cp_build_cpu_mirror_matches_legacy_item_path_exactly(self) -> None:
        scenarios = (
            (
                "prefill_straddle",
                {"qsl": [0, 3, 10, 14], "seq_lens": [100, 200, 300], "num_actual_tokens": 14, "num_input_tokens": 16},
                False,
                None,
            ),
            (
                "decode_batch",
                {
                    "qsl": [0, 1, 2, 3, 4],
                    "seq_lens": [50, 60, 70, 80],
                    "num_actual_tokens": 4,
                    "num_input_tokens": 4,
                    "attn_state": AscendAttentionState.DecodeOnly,
                },
                False,
                None,
            ),
            (
                "sp_padding",
                {"qsl": [0, 5], "seq_lens": [20], "num_actual_tokens": 5, "num_input_tokens": 7},
                False,
                None,
            ),
            (
                "full_graph",
                {
                    "qsl": [0, 3, 5],
                    "seq_lens": [10, 20],
                    "num_actual_tokens": 5,
                    "num_input_tokens": 8,
                    "with_indexer": True,
                    "sfa_full_graph": True,
                },
                True,
                None,
            ),
            (
                "mtp_1tok_draft",
                {"qsl": [0, 2, 4], "seq_lens": [17, 29], "num_actual_tokens": 4, "num_input_tokens": 4},
                False,
                SimpleNamespace(
                    method="deepseek_mtp",
                    num_speculative_tokens=1,
                    parallel_drafting=False,
                    needs_extra_input_slots=False,
                ),
            ),
        )
        with patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=self._det_cos_sin):
            for name, cm_kw, full_graph_builder, spec_cfg in scenarios:
                for rank in range(4):
                    b_fast = self._make_builder(["layer0"], max_seqs=8, max_tokens=32, full_graph=full_graph_builder)
                    b_legacy = self._make_builder(["layer0"], max_seqs=8, max_tokens=32, full_graph=full_graph_builder)
                    b_fast.speculative_config = spec_cfg
                    b_legacy.speculative_config = spec_cfg

                    c_fast = self._make_common(**cm_kw, with_cpu_qsl=True)
                    c_legacy = self._make_common(**cm_kw, with_cpu_qsl=False)
                    tp_grp = SimpleNamespace(world_size=4, rank_in_group=rank)
                    with patch.object(sfa_v1, "get_tp_group", return_value=tp_grp):
                        with self._no_item_patch(c_fast.query_start_loc, c_fast.seq_lens):
                            m_fast = b_fast.build(0, c_fast)
                        m_legacy = b_legacy.build(0, c_legacy)

                    cf, cl = m_fast.dsa_cp_context, m_legacy.dsa_cp_context
                    assert cf is not None and cl is not None
                    msg = f"mismatch in {name} rank={rank}"
                    self.assertTrue(torch.equal(cf.actual_seq_lengths_query, cl.actual_seq_lengths_query), msg)
                    self.assertTrue(torch.equal(cf.actual_seq_lengths_key, cl.actual_seq_lengths_key), msg)
                    self.assertTrue(torch.equal(m_fast.slot_mapping, m_legacy.slot_mapping), msg)
                    self.assertTrue(torch.equal(cf.slot_mapping_cp, cl.slot_mapping_cp), msg)
                    self.assertTrue(torch.equal(m_fast.cos, m_legacy.cos), msg)
                    self.assertTrue(torch.equal(m_fast.sin, m_legacy.sin), msg)
                    self.assertEqual(
                        (cf.local_start, cf.local_end, cf.local_end_with_pad),
                        (cl.local_start, cl.local_end, cl.local_end_with_pad),
                        msg,
                    )

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_dsa_cp_build_bypasses_stale_cpu_mirror_when_speculative_unsafe(self) -> None:
        unsafe_specs = (
            SimpleNamespace(num_speculative_tokens=2, parallel_drafting=False),
            SimpleNamespace(num_speculative_tokens=1, parallel_drafting=True),
            SimpleNamespace(
                method="draft_model",
                num_speculative_tokens=1,
                parallel_drafting=False,
                uses_draft_model=lambda: True,
            ),
            SimpleNamespace(
                method="custom_spec",
                num_speculative_tokens=1,
                parallel_drafting=False,
                uses_draft_model=lambda: False,
            ),
        )
        tp_grp = SimpleNamespace(world_size=4, rank_in_group=3)
        with (
            patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=self._det_cos_sin),
            patch.object(sfa_v1, "get_tp_group", return_value=tp_grp),
        ):
            for spec_cfg in unsafe_specs:
                builder = self._make_builder(["layer0"])
                builder.speculative_config = spec_cfg
                common = self._make_common(
                    qsl=[0, 4],
                    seq_lens=[2048],
                    num_actual_tokens=4,
                    num_input_tokens=4,
                    with_cpu_qsl=True,
                )
                # Inject stale clamped CPU lengths (as when multi-token/parallel draft reuses seq_lens_cpu=1)
                common.seq_lens_cpu = torch.tensor([1], dtype=torch.int32)
                common.query_start_loc_cpu = torch.tensor([0, 2], dtype=torch.int32)
                meta = builder.build(0, common)
                ctx = meta.dsa_cp_context
                assert ctx is not None
                # Must read GPU seq_lens=[2048] and GPU qsl=[0, 4] (Rank 3 owns [3, 4) -> cum_q=1, seq=2048)
                self.assertEqual(ctx.actual_seq_lengths_query.tolist(), [1])
                self.assertEqual(ctx.actual_seq_lengths_key.tolist(), [2048])

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_dsa_cp_seq_lens_cpu_ring_wraps_and_syncs_pending_events_only(self) -> None:
        builder = self._make_builder(["layer0"], max_seqs=4, max_tokens=16)
        assert builder._dsa_cp_seq_lens_cpu_ring is not None
        self.assertEqual(builder._dsa_cp_seq_lens_cpu_ring.shape, (4, 2, 5))
        self.assertEqual(builder._dsa_cp_seq_lens_cpu_ring.dtype, torch.int32)

        sync_log: list[str] = []

        class _FakeEv:
            def __init__(self, tag: str = "new", *, done: bool = False) -> None:
                self.tag = tag
                self.done = done
                self.recorded = False

            def record(self) -> None:
                self.recorded = True

            def query(self) -> bool:
                return self.done

            def synchronize(self) -> None:
                sync_log.append(self.tag)
                self.done = True

        # Pre-populate slots 2 (already done -> no sync) and 3 (pending -> must sync).
        builder._dsa_cp_seq_lens_events[2] = _FakeEv("slot2_ready", done=True)
        builder._dsa_cp_seq_lens_events[3] = _FakeEv("slot3_pending", done=False)

        tp_grp = SimpleNamespace(world_size=4, rank_in_group=0)
        with (
            patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=self._det_cos_sin),
            patch.object(sfa_v1, "get_tp_group", return_value=tp_grp),
        ):
            for step in range(6):
                expected_slot = step & 3
                self.assertEqual(builder._dsa_cp_seq_lens_slot, expected_slot)
                seq_val = 20 + step
                common = self._make_common(
                    qsl=[0, 4],
                    seq_lens=[seq_val],
                    num_actual_tokens=4,
                    num_input_tokens=4,
                )
                with self._no_item_patch(common.query_start_loc, common.seq_lens):
                    meta = builder.build(0, common)
                ctx = meta.dsa_cp_context
                assert ctx is not None
                # Rank 0 owns [0, 1) out of [0, 4): key_len = seq_val - 3
                self.assertEqual(ctx.actual_seq_lengths_query.tolist(), [1])
                self.assertEqual(ctx.actual_seq_lengths_key.tolist(), [seq_val - 3])
                self.assertEqual(int(builder._dsa_cp_seq_lens_cpu_ring[expected_slot, 1, 0]), seq_val - 3)

        self.assertEqual(builder._dsa_cp_seq_lens_slot, 2)
        self.assertEqual(sync_log, ["slot3_pending"])

        # Non-CPU device branch: records a torch.npu.Event into the active ring slot and
        # awaits pending prev_ev when the ring wraps back to that slot.
        builder._dsa_cp_seq_lens_slot = 0
        builder._dsa_cp_seq_lens_events = [_FakeEv("wrap_pending", done=False), None, None, None]
        sync_log.clear()
        created_evs: list[_FakeEv] = []

        class _DevWrap:
            type = "npu"

            def __eq__(self, other: object) -> bool:
                return getattr(other, "type", None) == "cpu" and False

        class _BufProxy:
            def __init__(self, inner: torch.Tensor) -> None:
                self._t = inner
                self.device = _DevWrap()
                self.shape = inner.shape
                self.dtype = inner.dtype
                self.data_ptr = inner.data_ptr

            def __getitem__(self, item: Any) -> "_SliceProxy":
                return _SliceProxy(self._t[item])

        class _SliceProxy:
            def __init__(self, target: torch.Tensor) -> None:
                self._t = target

            def copy_(self, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
                self._t.copy_(src, non_blocking=False)
                return self._t

            def __getitem__(self, item: Any) -> torch.Tensor:
                return self._t[item]

            def tolist(self) -> list[Any]:
                return self._t.tolist()

        builder.actual_seq_lengths_query = _BufProxy(builder.actual_seq_lengths_query)  # type: ignore[assignment]
        builder.actual_seq_lengths_key = _BufProxy(builder.actual_seq_lengths_key)  # type: ignore[assignment]
        fake_npu_ns = SimpleNamespace(
            Event=lambda: created_evs.append(_FakeEv(f"rec_{len(created_evs)}")) or created_evs[-1]
        )
        with (
            patch.object(torch, "npu", fake_npu_ns, create=True),
            patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=self._det_cos_sin),
            patch.object(sfa_v1, "get_tp_group", return_value=tp_grp),
        ):
            common_npu = self._make_common(qsl=[0, 4], seq_lens=[42], num_actual_tokens=4, num_input_tokens=4)
            builder.build(0, common_npu)
        self.assertEqual(sync_log, ["wrap_pending"])
        self.assertEqual(len(created_evs), 1)
        self.assertTrue(created_evs[0].recorded)
        self.assertIs(builder._dsa_cp_seq_lens_events[0], created_evs[0])

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_dsa_cp_rebind_in_place_slot_mutation_no_stale_cache_and_zero_pad(self) -> None:
        builder = self._make_builder(["layer0", "layer1"])
        builder._layerwise_prefill_p_node = True
        tp_grp = SimpleNamespace(world_size=4, rank_in_group=2)  # owns [4, 6) for n=6, [4, 6) for n=8

        # Mutate the SAME slot_mapping storage across two consecutive forwards:
        # Step 1 (slots start at 10 / 200) -> Step 2 (in-place overwrite to 500 / 800).
        common_b0 = self._make_common(
            qsl=[0, 6], seq_lens=[32], num_actual_tokens=6, num_input_tokens=6, slot_offset=10
        )
        common_b1 = self._make_common(
            qsl=[0, 6], seq_lens=[32], num_actual_tokens=6, num_input_tokens=6, slot_offset=200
        )
        ptr0, ptr1 = common_b0.slot_mapping.data_ptr(), common_b1.slot_mapping.data_ptr()

        with (
            patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=self._det_cos_sin),
            patch.object(sfa_v1, "get_tp_group", return_value=tp_grp),
        ):
            tpl1 = builder.build(0, common_b0)
            reb1 = builder.rebind_layerwise_prefill_metadata(tpl1, common_b1)
            assert reb1 is not None and tpl1.dsa_cp_context is not None and reb1.dsa_cp_context is not None
            self.assertEqual(tpl1.dsa_cp_context.slot_mapping_cp.tolist(), [14, 15])
            self.assertEqual(reb1.dsa_cp_context.slot_mapping_cp.tolist(), [204, 205])

            common_b0.slot_mapping.copy_(torch.arange(500, 506, dtype=torch.int32))
            common_b1.slot_mapping.copy_(torch.arange(800, 806, dtype=torch.int32))
            self.assertEqual((common_b0.slot_mapping.data_ptr(), common_b1.slot_mapping.data_ptr()), (ptr0, ptr1))

            tpl2 = builder.build(0, common_b0)
            reb2 = builder.rebind_layerwise_prefill_metadata(tpl2, common_b1)
            assert reb2 is not None and tpl2.dsa_cp_context is not None and reb2.dsa_cp_context is not None
            self.assertEqual(tpl2.dsa_cp_context.slot_mapping_cp.tolist(), [504, 505])
            self.assertEqual(reb2.dsa_cp_context.slot_mapping_cp.tolist(), [804, 805])

            # Lazy _dsa_cp_rebind_cache=None initialization + pad_size_slot == 0 (8 tokens, TP=4 -> 0 pad):
            common_aligned0 = self._make_common(
                qsl=[0, 8], seq_lens=[32], num_actual_tokens=8, num_input_tokens=8, slot_offset=300
            )
            common_aligned1 = self._make_common(
                qsl=[0, 8], seq_lens=[32], num_actual_tokens=8, num_input_tokens=8, slot_offset=900
            )
            tpl_aligned = builder.build(0, common_aligned0)
            tpl_aligned._dsa_cp_rebind_cache = None  # type: ignore[attr-defined]
            pad_calls: list[int] = []
            orig_pad = torch.nn.functional.pad
            with patch.object(
                sfa_v1.nn.functional,
                "pad",
                side_effect=lambda t, *a, **k: (pad_calls.append(t.ndim), orig_pad(t, *a, **k))[1],
            ):
                reb_aligned = builder.rebind_layerwise_prefill_metadata(tpl_aligned, common_aligned1)
            assert reb_aligned is not None and reb_aligned.dsa_cp_context is not None
            self.assertEqual(pad_calls, [])
            self.assertIsNotNone(getattr(tpl_aligned, "_dsa_cp_rebind_cache", None))
            self.assertEqual(reb_aligned.slot_mapping.tolist(), list(range(900, 908)))
            self.assertEqual(reb_aligned.dsa_cp_context.slot_mapping_cp.tolist(), [904, 905])

    @patch_distributed_groups(dcp_size=1, pcp_size=1, needs_mocks=False)
    def test_dsa_cp_ensure_seq_lens_ring_growth_syncs_pending_events(self) -> None:
        builder = self._make_builder(["layer0"], max_seqs=2, max_tokens=16)
        assert builder._dsa_cp_seq_lens_cpu_ring is not None
        self.assertEqual(builder._dsa_cp_seq_lens_cpu_ring.shape, (4, 2, 3))

        sync_calls: list[str] = []

        class _PendingEv:
            def __init__(self, tag: str) -> None:
                self.tag = tag

            def query(self) -> bool:
                return False

            def synchronize(self) -> None:
                sync_calls.append(self.tag)

        builder._dsa_cp_seq_lens_events[1] = _PendingEv("slot1_pending")
        builder._dsa_cp_seq_lens_events[3] = _PendingEv("slot3_pending")

        # Grow ring via num_segs=6 > old cols=3
        common_big = self._make_common(
            qsl=[0, 1, 2, 3, 4, 5, 6],
            seq_lens=[10, 11, 12, 13, 14, 15],
            num_actual_tokens=6,
            num_input_tokens=8,
        )
        tp_grp = SimpleNamespace(world_size=4, rank_in_group=1)  # owns [2, 4) -> reqs 2, 3
        with (
            patch.object(sfa_v1, "get_cos_and_sin_mla", side_effect=self._det_cos_sin),
            patch.object(sfa_v1, "get_tp_group", return_value=tp_grp),
            self._no_item_patch(common_big.query_start_loc, common_big.seq_lens),
        ):
            meta_big = builder.build(0, common_big)

        self.assertEqual(sync_calls, ["slot1_pending", "slot3_pending"])
        self.assertEqual(builder._dsa_cp_seq_lens_events, [None, None, None, None])
        self.assertEqual(builder._dsa_cp_seq_lens_cpu_ring.shape, (4, 2, 6))
        ctx_big = meta_big.dsa_cp_context
        assert ctx_big is not None
        self.assertEqual(ctx_big.actual_seq_lengths_query.tolist(), [0, 0, 1, 2, 2, 2])
        self.assertEqual(ctx_big.actual_seq_lengths_key.tolist(), [0, 0, 12, 13, 0, 0])


if __name__ == "__main__":
    unittest.main()
