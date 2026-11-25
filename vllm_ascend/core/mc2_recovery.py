# SPDX-License-Identifier: Apache-2.0
"""Process-independent token limits for decoder recovery.

Keep this module free of device and distributed initialization: EngineCore and
the model worker must derive the same limit before either schedules recovery.
"""

from typing import Any

from vllm_ascend import envs


def independent_decoder_dcp(config: Any) -> bool:
    """Select a static candidate; workers must still validate the loaded layout.

    EngineCore computes its recovery budget before the worker sets the validated
    decoder-DCP marker. Keep this opt-in separate from generic context parallelism.
    """
    parallel, model = config.parallel_config, getattr(config, "model_config", None)
    transfer = config.kv_transfer_config
    return bool(
        envs.VLLM_ASCEND_SFA_INDEPENDENT_DP
        and envs.VLLM_ASCEND_SFA_FULL_GRAPH
        and envs.VLLM_ASCEND_DSA_UNBUNDLE
        and envs.VLLM_ASCEND_DSA_TWO_GROUPS
        and not envs.VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE
        and parallel.distributed_executor_backend == "mp"
        and parallel.nnodes_within_dp == 1
        and not getattr(parallel, "enable_elastic_ep", False)
        and parallel.decode_context_parallel_size in (2, 4, 8)
        and parallel.decode_context_parallel_size == parallel.tensor_parallel_size == parallel.local_world_size
        and transfer is not None
        and transfer.is_kv_consumer
        and getattr(model, "use_mla", False)
        and not model.enforce_eager
        and hasattr(model.hf_text_config, "index_topk")
    )


def calculate_mc2_tokens_capacity(config: Any, max_num_reqs: int, query_width: int) -> int:
    """Return the existing MC2 allocation size, rounded to a complete TP row."""
    compilation = config.compilation_config
    if compilation.cudagraph_capture_sizes:
        capacity = int(compilation.max_cudagraph_capture_size)
    else:
        capacity = min(max_num_reqs * query_width, 512)
    tp = int(config.parallel_config.tensor_parallel_size)
    if capacity <= 0 or tp <= 0:
        raise ValueError("MC2 requires positive token capacity and TP size")
    return (capacity + tp - 1) // tp * tp


def decoder_recovery_budget(config: Any) -> int | None:
    """Bound recompute-scheduler consumers without changing prefiller budgets.

    Workers still validate the communication method for their hardware. This
    function establishes the token-bound contract, not collective eligibility.
    """
    transfer = config.kv_transfer_config
    if not transfer or not transfer.is_kv_consumer:
        return None
    parallel = config.parallel_config
    if (
        not parallel.enable_expert_parallel
        or parallel.data_parallel_size <= 1
        or parallel.pipeline_parallel_size != 1
        or parallel.prefill_context_parallel_size != 1
        or (parallel.decode_context_parallel_size != 1 and not independent_decoder_dcp(config))
        or parallel.enable_dbo
    ):
        return None
    spec = config.speculative_config
    if spec is not None and (
        getattr(spec, "method", None) not in ("mtp", "deepseek_mtp")
        or getattr(spec, "parallel_drafting", False)
        or getattr(spec, "disable_padded_drafter_batch", False)
    ):
        # Extra-input-slot draft paths can expand a full target batch beyond
        # MC2 capacity. They must keep the existing cross-DP agreement path.
        return None
    width = 1 + (int(spec.num_speculative_tokens) if spec else 0)
    capacity = calculate_mc2_tokens_capacity(config, config.scheduler_config.max_num_seqs, width)
    budget = min(int(config.scheduler_config.max_num_batched_tokens), capacity)
    if not config.scheduler_config.enable_chunked_prefill:
        raise ValueError("Bounded decoder recovery requires enable_chunked_prefill")
    return budget
