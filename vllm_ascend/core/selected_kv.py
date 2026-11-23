# SPDX-License-Identifier: Apache-2.0
"""Configuration, HBM budget reservation, and runner hooks for ondemand-selected."""

from __future__ import annotations

import os
import sys
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass
from typing import Any

from vllm.logger import logger

import vllm_ascend.envs as envs_ascend

PAGE_ROWS = 128
NOPE_ROW_DIM = 512
PE_ROW_DIM = 64
TOPK = 2048
MLA_NOPE_ROW_BYTES = 1024
MLA_PE_ROW_BYTES = 128
MAIN_ROW_BYTES = MLA_NOPE_ROW_BYTES + MLA_PE_ROW_BYTES
MIN_HISTORY_TOKENS = 16384
MAX_HISTORY_TOKENS = 524288
MAX_REQUESTS = 256
MAX_BATCHED_TOKENS = 16384
NATIVE_ALIGNMENT = 2 * 1024 * 1024
PAGE_BASES_ENTRY_BYTES = 16

_ACTIVE_RUNTIMES: weakref.WeakSet[Any] = weakref.WeakSet()


def _pages(tokens: int) -> int:
    return (tokens + PAGE_ROWS - 1) // PAGE_ROWS


def _check(ok: object, msg: str, exc: type[Exception] = ValueError) -> None:
    if not ok:
        raise exc(msg)


def _rcheck(ok: object, msg: str) -> None:
    _check(ok, msg, RuntimeError)


def _ga(obj: Any, path: str, default: Any = None) -> Any:
    for key in path.split("."):
        obj = obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)
    return default if obj is None else obj


def _enabled(runner: Any) -> bool:
    return _ga(runner, "ascend_config.selected_kv_config.enabled") is True


def _active_rt(runner: Any) -> Any | None:
    return getattr(runner, "selected_kv_runtime", None) if _enabled(runner) else None


def _slices(runner: Any) -> tuple[Any, Any, Any]:
    batch = runner.input_batch
    num_reqs = int(batch.num_reqs)
    return batch.req_ids[:num_reqs], batch.num_computed_tokens_cpu[:num_reqs], runner.query_start_loc.np[: num_reqs + 1]


def _run_cleanup(fn: Any, err: BaseException | None = None) -> None:
    try:
        fn()
    except BaseException as cleanup_err:
        if err is not None and hasattr(err, "add_note"):
            err.add_note(f"Selected KV cleanup failed: {cleanup_err!r}")


@dataclass(frozen=True)
class SelectedKVConfig:
    enabled: bool = False

    def __post_init__(self) -> None:
        _check(type(self.enabled) is bool, f"SelectedKVConfig.enabled must be bool, got {self.enabled!r}", TypeError)

    @classmethod
    def from_dict(cls, raw: object) -> SelectedKVConfig:
        default_enabled = bool(envs_ascend.VLLM_ASCEND_ENABLE_SELECTED_KV_PREFILL)
        if raw is None:
            return cls(default_enabled)
        if isinstance(raw, cls):
            return raw
        _check(isinstance(raw, Mapping), f"selected_kv_config must be Mapping, got {type(raw).__name__}", TypeError)
        extra = set(raw) - {"enabled"}  # type: ignore[arg-type]
        _check(not extra, f"Unknown keys in selected_kv_config: {sorted(extra)}")
        return cls(raw.get("enabled", default_enabled))  # type: ignore[union-attr]

    def validate_deployment(self, cfg: Any) -> None:
        if not self.enabled:
            return

        _check(
            not bool(getattr(envs_ascend, "VLLM_ASCEND_ENABLE_DSA_LATENT_OFFLOAD", False)),
            "Selected KV prefill cannot be combined with DSA latent offload",
        )

        _check(
            not bool(getattr(envs_ascend, "VLLM_ASCEND_DSA_USE_ADAPTER_CACHE", False)),
            "Selected KV prefill cannot be combined with DSA adapter cache",
        )

        fc2_size = getattr(envs_ascend, "VLLM_ASCEND_FLASHCOMM2_PARALLEL_SIZE", 0)
        _check(
            not isinstance(fc2_size, bool) and int(fc2_size or 0) == 0,
            "Selected KV prefill cannot be combined with FLASHCOMM2",
        )

        def collect_lmc_roles(spec: Any) -> list[bool]:
            if spec is None:
                return [False]
            name = _ga(spec, "kv_connector")
            role = _ga(spec, "kv_role")
            is_prod_attr = _ga(spec, "is_kv_producer")
            role_valid = role in ("kv_producer", "kv_consumer", "kv_both") or type(is_prod_attr) is bool
            if not isinstance(name, str) or not name or not role_valid:
                return [False]
            spec_prod = role in ("kv_producer", "kv_both") if role is not None else bool(is_prod_attr)
            if name in ("MultiConnector", "AscendMultiConnector"):
                children = _ga(spec, "kv_connector_extra_config.connectors")
                if not isinstance(children, (list, tuple)) or not children:
                    return [False]
                return [item and spec_prod for child in children for item in collect_lmc_roles(child)]
            direct = name in (
                "LMCacheConnectorV1",
                "LMCacheConnectorV1Dynamic",
                "LMCacheAscendConnector",
                "LMCacheAscendConnectorV1Dynamic",
            )
            direct = direct or "lmcache" in str(_ga(spec, "kv_connector_module_path", "")).lower()
            return [spec_prod] if direct else []

        def _is_truthy_flag(val: Any) -> bool:
            if isinstance(val, bool):
                return val
            if isinstance(val, int):
                return val != 0
            if isinstance(val, str):
                return val.strip().lower() in ("1", "true", "yes", "on")
            return False

        def has_preemption_ckpt(spec: Any) -> bool:
            if spec is None:
                return False
            for path in (
                "decode_preemption_checkpoint",
                "kv_connector_extra_config.decode_preemption_checkpoint",
                "kv_connector_extra_config.lmcache_config.decode_preemption_checkpoint",
            ):
                if _is_truthy_flag(_ga(spec, path)):
                    return True
            children = _ga(spec, "kv_connector_extra_config.connectors")
            if isinstance(children, (list, tuple)):
                return any(has_preemption_ckpt(child) for child in children)
            return False

        tr = cfg.kv_transfer_config
        _check(
            not _is_truthy_flag(os.environ.get("LMCACHE_DECODE_PREEMPTION_CHECKPOINT", ""))
            and not has_preemption_ckpt(tr),
            "Selected KV prefill cannot be combined with decode_preemption_checkpoint",
        )
        sc = cfg.scheduler_config
        mc = cfg.model_config
        pc = cfg.parallel_config
        sp = cfg.speculative_config
        env_ok = (
            bool(envs_ascend.VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE)
            and int(envs_ascend.VLLM_ASCEND_LAYERWISE_PREFILL_NUM_BANKS) >= 2
        )
        eager_v1 = (
            type(getattr(sc, "async_scheduling", False)) is bool
            and bool(mc.enforce_eager)
            and not getattr(cfg, "use_v2_model_runner", False)
            and not getattr(pc, "enable_dbo", False)
        )
        tp_size = getattr(pc, "tensor_parallel_size", 1)
        sp_tokens = getattr(sp, "num_speculative_tokens", None)
        draft_tp = getattr(sp, "draft_tensor_parallel_size", None)
        spec_ok = sp is None or (
            getattr(sp, "method", None) in ("mtp", "deepseek_mtp")
            and type(sp_tokens) is int
            and sp_tokens == 1
            and (draft_tp is None or (type(draft_tp) is int and draft_tp == tp_size))
        )
        pp_ok = (
            type(tp_size) is int
            and tp_size >= 1
            and all(
                getattr(pc, f"{axis}_parallel_size", 1) == 1
                for axis in ("pipeline", "prefill_context", "decode_context")
            )
        )
        model_ok = (
            not getattr(mc, "is_encoder_decoder", False)
            and not getattr(mc, "enable_sleep_mode", False)
            and int(mc.max_model_len) > 0
        )
        toks = int(sc.max_num_batched_tokens)
        batch_ok = 0 < toks <= MAX_BATCHED_TOKENS and 0 < min(int(sc.max_num_seqs), toks) <= MAX_REQUESTS
        _check(
            bool(tr)
            and collect_lmc_roles(tr) == [True]
            and env_ok
            and eager_v1
            and spec_ok
            and pp_ok
            and model_ok
            and batch_ok,
            "Unsupported deployment",
        )


def init_selected_kv_config(additional_config: Mapping[str, Any], vllm_config: Any) -> SelectedKVConfig:
    cfg = SelectedKVConfig.from_dict(additional_config.get("selected_kv_config"))
    cfg.validate_deployment(vllm_config)
    return cfg


class SelectedKVBudget:
    _topology: tuple[Any, ...] = ((), frozenset(), {}, False)

    def __init__(self, avail: int, page: int, fixed: int, per_blk: int, pad: int = 0) -> None:
        args_ok = (
            all(type(v) is int and v >= 0 for v in (avail, fixed, per_blk, pad)) and type(page) is int and page > 0
        )
        _check(args_ok, "Invalid workspace byte arguments")
        self._avail = avail
        self._page = page
        self._pad = pad
        self._fixed = fixed
        self._per_blk = per_blk
        max_blocks = (avail - pad) // page
        self.maximum_blocks = max_blocks
        self.native_budget_bytes = avail - pad - fixed - per_blk * max_blocks
        _check(max_blocks > 0 and self.native_budget_bytes >= page, "Insufficient selected-KV memory")

    def validate(self, num_blocks: int, native_bytes: int | None = None) -> None:
        _check(type(num_blocks) is int and 0 < num_blocks <= self.maximum_blocks, "Block count exceeds reservation")
        native = num_blocks * self._page + self._pad if native_bytes is None else native_bytes
        _check(native + self._fixed + self._per_blk * num_blocks <= self._avail, "Budget exceeded")


reserve_workspace_bytes = SelectedKVBudget


def _topology_diagnostic_type(value: Any) -> str:
    # Bypass custom metaclass hooks and str-subclass slicing on class names.
    name = type.__dict__["__name__"].__get__(type(value))
    return str.__getitem__(name, slice(0, 80))


def _topology_diagnostic_value(value: Any) -> str:
    """Describe small builtins without inspecting device values or custom reprs."""
    value_type = type(value)
    type_name = _topology_diagnostic_type(value)
    if value_type is str:
        rendered = repr(value[:80])[:120]
        if len(value) > 80:
            rendered += "..."
    elif value_type is int:
        rendered = repr(value) if value.bit_length() <= 256 else "<more than 256 bits>"
    elif value_type is bool or value_type is float or value is None:
        rendered = repr(value)
    else:
        return f"<type={type_name}>"
    return f"{rendered} (type={type_name})"


def _cp_tp_size(vllm_config: Any, dsa_cp: bool) -> int:
    if not dsa_cp:
        return 1
    tp_size = _ga(vllm_config, "parallel_config.tensor_parallel_size", 1)
    _check(type(tp_size) is int and tp_size >= 1, "Invalid tensor_parallel_size")
    return tp_size


def _resolve_topology(runner: Any) -> tuple[tuple[str, ...], frozenset[str], dict[str, str], bool]:
    ctx = runner.compilation_config.static_forward_context

    def layer_index(parts: list[str]) -> int:
        after_layers = [b for a, b in zip(parts, parts[1:]) if a == "layers"]
        return next((int(tok) for tok in after_layers + parts if tok.isdigit()), -1)

    indexed = sorted(
        (layer_index(name.split(".")), name)
        for name, mod in ctx.items()
        if "indexer" not in name and _ga(mod, "impl.kv_lora_rank") is not None
    )
    names = tuple(name for _, name in indexed)
    draft = frozenset(map(str, _ga(runner, "drafter.attn_layer_names", ())))
    _check(
        bool(names) and [idx for idx, _ in indexed] == list(range(len(names))) and draft < set(names), "Bad MLA/draft"
    )

    owners: dict[str, str] = {}
    prev_owner: str | None = None
    dsa_cp_flags: set[bool] = set()
    impl_tp_sizes: set[int | None] = set()
    for name in names:
        mod = ctx[name]
        impl = mod.impl
        parent_mod = ctx.get(name.removesuffix(".attn")) if name.endswith(".attn") else None
        skip_topk = bool(getattr(impl, "skip_topk", False))
        raw_dsa_cp = getattr(impl, "enable_dsa_cp", False)
        impl_dsa_cp = bool(raw_dsa_cp)
        raw_o_proj_tp = getattr(impl, "enable_dsa_cp_with_o_proj_tp", False)
        impl_o_proj_tp = bool(raw_o_proj_tp)
        vl_first = any(
            bool(getattr(obj, "is_vl_first_layer", False)) for obj in (parent_mod, mod, impl) if obj is not None
        )
        impl_tp = getattr(impl, "tp_size", None)
        tp_ok = not impl_dsa_cp or impl_tp is None or (type(impl_tp) is int and impl_tp >= 1)
        kv_lora_rank = getattr(impl, "kv_lora_rank", 0)
        qk_rope_head_dim = getattr(impl, "qk_rope_head_dim", 0)
        dims_ok = (kv_lora_rank, qk_rope_head_dim) == (NOPE_ROW_DIM, PE_ROW_DIM)
        layer_ok = (
            dims_ok
            and not impl_o_proj_tp
            and not (impl_dsa_cp and vl_first)
            and tp_ok
            and not (name in draft and skip_topk)
        )
        if not layer_ok:
            raise ValueError(
                f"Bad MLA/draft {name!r}: impl_type={_topology_diagnostic_type(impl)}, "
                f"kv_lora_rank={_topology_diagnostic_value(kv_lora_rank)} (expected {NOPE_ROW_DIM}), "
                f"qk_rope_head_dim={_topology_diagnostic_value(qk_rope_head_dim)} (expected {PE_ROW_DIM}), "
                f"impl.enable_dsa_cp={_topology_diagnostic_value(raw_dsa_cp)}, "
                f"impl.enable_dsa_cp_with_o_proj_tp={_topology_diagnostic_value(raw_o_proj_tp)}, "
                f"is_vl_first_layer={vl_first}, "
                f"impl.tp_size={_topology_diagnostic_value(impl_tp)}, "
                f"is_draft={name in draft}, skip_topk={skip_topk}"
            )
        dsa_cp_flags.add(impl_dsa_cp)
        impl_tp_sizes.add(impl_tp)
        if not skip_topk:
            candidates = (
                getattr(impl, "index_topk", None),
                _ga(impl, "indexer.topk_tokens"),
                _ga(impl, "indexer.topk_output_width"),
            )
            topk_val = next((val for val in candidates if type(val) is int), None)
            _check(topk_val == TOPK, "Selected KV requires TopK=2048")
            prev_owner = name
        _check(prev_owner is not None, "Selected KV followers require an earlier actual TopK owner")
        owners[name] = str(prev_owner)
    _check(len(dsa_cp_flags) == 1, "Selected KV layers must agree on enable_dsa_cp")
    dsa_cp = next(iter(dsa_cp_flags))
    if dsa_cp:
        _check(len(impl_tp_sizes) == 1, "Selected KV DSA-CP layers must agree on tp_size")
        resolved_tp = next(iter(impl_tp_sizes))
        vcfg = getattr(runner, "vllm_config", None)
        if vcfg is not None:
            cfg_tp = _cp_tp_size(vcfg, True)
            if resolved_tp is not None:
                _check(resolved_tp == cfg_tp, "Selected KV DSA-CP layer tp_size mismatch")
            _check(
                not (draft and cfg_tp > 1 and bool(_ga(vcfg, "model_config.is_multimodal_model", False))),
                "Selected KV multimodal MTP draft is unsupported under DSA-CP",
            )
    return names, draft, owners, dsa_cp


def reserve_selected_kv_workspace(worker: Any, available_bytes: int) -> int:
    runner = worker.model_runner
    if not _enabled(runner):
        return available_bytes
    runner.ascend_config.selected_kv_config.validate_deployment(worker.vllm_config)
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import prefill_selected_runtime as psr
    from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

    _check(
        get_ascend_device_type() in (AscendDeviceType.A2, AscendDeviceType.A3),
        "Selected KV kernels require an A2 or A3 build",
    )
    p_node_ok = (
        getattr(runner, "pcp_size", 1) == 1
        and getattr(runner, "dcp_size", 1) == 1
        and bool(getattr(runner, "layerwise_prefill_p_node", False))
        and int(getattr(runner, "layerwise_prefill_bank_count", 0)) >= 2
    )
    _check(p_node_ok, "Requires P-node (>=2 banks) without PCP/DCP")

    names, draft, owners, dsa_cp = _resolve_topology(runner)
    sc = worker.vllm_config.scheduler_config
    avail = int(available_bytes)
    toks = int(sc.max_num_batched_tokens)
    cp_tp_size = _cp_tp_size(worker.vllm_config, dsa_cp)
    fixed = psr.selected_workspace_bytes(
        toks, min(int(sc.max_num_seqs), toks), dsa_cp=dsa_cp, cp_tp_size=cp_tp_size
    ) + 3 * PAGE_BASES_ENTRY_BYTES * len(names)
    budget = SelectedKVBudget(
        avail, PAGE_ROWS * MAIN_ROW_BYTES, fixed, PAGE_BASES_ENTRY_BYTES * len(names), NATIVE_ALIGNMENT
    )
    budget._topology = (names, draft, owners, dsa_cp)
    override_blocks = _ga(worker, "cache_config.num_gpu_blocks_override")
    if override_blocks is not None:
        budget.validate(int(override_blocks))
    runner._selected_kv_budget = budget
    reserved_mib = (avail - budget.native_budget_bytes) / 1048576.0
    logger.info_once("Reserved %.2f MiB for selected-KV prefill workspace", reserved_mib, scope="local")
    return budget.native_budget_bytes


def _resolve_lmcache_target(conn: Any) -> tuple[Any, tuple[tuple[int, int], ...]]:
    found: list[tuple[Any, tuple[tuple[int, int], ...]]] = []

    def walk(node: Any, path: tuple[tuple[int, int], ...]) -> None:
        if node is None:
            return
        if (eng := getattr(node, "_lmcache_engine", None)) is not None:
            found.append((eng, path))
        sub_conns = getattr(node, "_connectors", None)
        if isinstance(sub_conns, (list, tuple)):
            total = len(sub_conns)
            for idx, child in enumerate(sub_conns):
                walk(child, (*path, (idx, total)))

    walk(conn, ())
    _rcheck(len(found) == 1, "Bad LMCache engine")
    impl, meta_spec = found[0]
    _rcheck(getattr(impl, "selected_kv_source", None) is None, "Bad LMCache engine")
    return impl, meta_spec


def _unwrap_connector_metadata(conn_meta: Any, src: Any) -> Any:
    cur = conn_meta
    for idx, expected_len in getattr(src, "_connector_meta_spec", ()):
        if cur is None:
            return None
        sub_metas = getattr(cur, "metadata", None)
        _rcheck(
            isinstance(sub_metas, (tuple, list)) and len(sub_metas) == expected_len and 0 <= idx < len(sub_metas),
            "MultiConnector metadata does not match child connectors",
        )
        cur = sub_metas[idx]
    return cur


def initialize_selected_kv(runner: Any, kv_cache_config: Any, kv_caches: Mapping[str, Any]) -> None:
    if not _enabled(runner) or bool(getattr(runner, "_profiling_cudagraph_memory", False)):
        return
    from vllm.distributed.kv_transfer import get_kv_transfer_group

    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import lmcache_selected_kv_source as lsks
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import prefill_selected_runtime as psr

    budget = runner._selected_kv_budget
    _rcheck(budget is not None and runner.selected_kv_runtime is None, "Budget once")
    names, draft, owners, dsa_cp = budget._topology if budget._topology[0] else _resolve_topology(runner)

    def as_tuple(val: Any) -> tuple[Any, ...]:
        return tuple(val) if isinstance(val, (tuple, list)) else (val,)

    raw_planes = {name: as_tuple(kv_caches[name]) for name in names}
    _check(
        bool(names) and all(len(planes) >= 2 for planes in raw_planes.values()),
        "Selected KV requires (k_nope, k_pe) planes",
    )
    main_caches = {name: planes[:2] for name, planes in raw_planes.items()}
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for val in kv_caches.values()
        for tensor in as_tuple(val)
    }
    budget.validate(int(main_caches[names[0]][0].shape[0]), sum(storages.values()))

    impl, meta_spec = _resolve_lmcache_target(get_kv_transfer_group())
    src = lsks.LMCacheSelectedKVSource(runner, impl, names, main_caches)
    src._connector_meta_spec = meta_spec
    try:
        rt = psr.SelectedKVRuntime(runner, owners, main_caches, src, draft_layer_names=draft, dsa_cp=dsa_cp)
    except BaseException as err:
        _run_cleanup(src.close, err)
        raise
    impl.selected_kv_source = src
    runner.selected_kv_runtime = rt


@contextmanager
def _guard_step(rt: Any, step: Any | None) -> Iterator[None]:
    try:
        yield
    except BaseException as err:
        rt.fail(step, err)
        raise


@contextmanager
def selected_kv_prepare_context(runner: Any, sched_out: Any, attn_meta: Any = None) -> Iterator[None]:
    rt = _active_rt(runner)
    try:
        if rt is not None:
            rt._handoff_previous()
            src = rt.source
            prune = getattr(src, "_prune_retired_steps", None)
            if callable(prune):
                prune()
            batch = runner.input_batch
            src.step = None
            src._staged_step = None
            conn_meta = _unwrap_connector_metadata(sched_out.kv_connector_metadata, src)
            if conn_meta is not None and int(batch.num_reqs) > 0 and isinstance(attn_meta, Mapping):
                bank_tables = getattr(batch, "layerwise_prefill_block_tables", None)
                if bank_tables is not None:
                    _rcheck(len(bank_tables) == src.num_banks, "Banks")
                else:
                    bank_tables = (batch.block_table,)
                    _rcheck(len(bank_tables) >= src.num_banks, "Banks")
                cpu_tables = [bank_tables[bank][0].get_cpu_tensor().numpy() for bank in range(src.num_banks)]
                src.begin_runner_step(conn_meta, *_slices(runner), cpu_tables)
        yield
    except BaseException as err:
        if rt is not None:
            step = rt.source.step
            if (step is not None and not getattr(step, "_finished", False)) or getattr(rt.source, "_failed", False):
                rt.fail(step, err)
        raise
    finally:
        if rt is not None:
            rt.source._staged_step = None


@contextmanager
def selected_kv_forward(runner: Any, sched: Any, attn_meta: Any = None, *, defer_finish: bool = False) -> Iterator[Any]:
    rt = _active_rt(runner)
    if rt is None or rt.source.step is None:
        yield rt.bind_step(None, (), (), (), attn_meta) if rt is not None else None
        return
    step = rt.source.step
    with _guard_step(rt, step):
        _rcheck(
            step.connector_metadata is _unwrap_connector_metadata(sched.kv_connector_metadata, rt.source),
            "Connector metadata mismatch",
        )
        yield rt.bind_step(step, *_slices(runner), attn_meta)
        if defer_finish and rt.draft_layer_names:
            rt.defer_to_draft_step()
        else:
            rt.finish_step()


def bind_selected_kv_draft_metadata(runner: Any, draft_meta: Mapping[str, Any], common_meta: Any) -> Any | None:
    rt = _active_rt(runner)
    if rt is None:
        return None
    with _guard_step(rt, rt.source_step):
        return rt.bind_draft_step(draft_meta, common_meta)


@contextmanager
def selected_kv_draft_forward(runner: Any) -> Iterator[None]:
    rt = _active_rt(runner)
    active_draft = rt is not None and rt._awaiting_draft
    with _guard_step(rt, rt.source_step) if active_draft else nullcontext():
        yield
        if active_draft:
            rt.finish_step()


def retire_selected_kv_if_idle(runner: Any) -> None:
    rt = _active_rt(runner)
    if rt is not None:
        rt._retire_previous()
        prune = _ga(rt, "source._prune_retired_steps")
        if callable(prune):
            prune()
        rt.source.step = None


def abort_active_selected_kv_steps(*, deferred_only: bool = False) -> None:
    err = sys.exc_info()[1] if _ACTIVE_RUNTIMES else None
    for rt in tuple(_ACTIVE_RUNTIMES):
        step = rt.source_step or getattr(rt.source, "step", None)
        src_failed = not deferred_only and bool(getattr(rt.source, "_failed", False)) and not rt._failed
        rt_unsynced = (
            not deferred_only
            and rt._failed
            and hasattr(rt.source, "_failed")
            and not bool(getattr(rt.source, "_failed", False))
        )
        if (
            rt._awaiting_draft
            or src_failed
            or rt_unsynced
            or (not deferred_only and step is not None and not getattr(step, "_finished", False))
        ):
            rt.fail(step, err)


def close_selected_kv_worker(worker: Any) -> None:
    rt = _ga(worker, "model_runner.selected_kv_runtime")
    if rt is None:
        return
    try:
        rt.close()
    except RuntimeError:
        if not (getattr(rt, "_failed", False) or getattr(getattr(rt, "source", None), "_failed", False)):
            raise
        eng = _ga(rt, "source.engine")
        dev = getattr(rt, "device", None) or _ga(rt, "source.device")
        synced = False
        if dev is not None and getattr(dev, "type", None) == "npu":
            sync_fn = _ga(sys.modules.get("torch"), "npu.synchronize")
            if callable(sync_fn):
                with suppress(BaseException):
                    sync_fn(dev)
                    synced = True
        if not synced and eng is not None:
            eng._retain_unfenced_external_dma = True
        logger.warning("Retaining selected-KV workspace/leases during shutdown after failed step.")
