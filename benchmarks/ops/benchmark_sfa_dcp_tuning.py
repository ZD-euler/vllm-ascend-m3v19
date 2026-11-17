# SPDX-License-Identifier: Apache-2.0
"""A/B early query gathering through real sparse H2D and native attention.

Both arms retain one query all-gather and one output all-to-all per layer,
with the same packet format and merged query/bridge copy reductions.
"""

import argparse
import json
import os
import statistics
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from benchmark_sfa_copy_reduction import AttentionPipeline, measure_case
from benchmark_sfa_target_preparation import ROOT, make_inputs


def tuning_pipeline(args, inputs, sources, group, candidate):
    return AttentionPipeline(
        args,
        inputs,
        sources,
        group,
        direct=True,
        early_gather=candidate,
    )


def measure_variant(args, control):
    result = measure_case(args, control, pipeline_factory=tuning_pipeline)
    # Preserve both arms' samples. One sample is a whole captured multi-step
    # group, divided by steps; this is not an individual-token p95 statistic.
    all_samples = result["per_rank_samples"]
    slowest = {
        name: [max(rank[name][i] for rank in all_samples) for i in range(len(all_samples[0][name]))]
        for name in ("baseline", "direct")
    }
    result.update(
        variant=args.variant,
        collective_calls_per_layer={"query_all_gather": 1, "output_all_to_all": 1},
        packet_bytes_per_head=1056,
        slowest_rank_mean={name: statistics.mean(values) for name, values in slowest.items()},
        median_reduction_percent=100
        * (1 - result["slowest_rank_median"]["direct"] / result["slowest_rank_median"]["baseline"]),
    )
    return result


def format_case_summary(case):
    medians, means = case["slowest_rank_median"], case["slowest_rank_mean"]
    samples = len(case["per_rank_samples"][0]["baseline"])
    return (
        f"RESULT {case['variant']} {case['layout']} batch={case['requests']} padding={case['padding']} "
        f"DCP={case['dcp_size']} layers={case['layers']} steps={case['steps']} "
        f"samples={samples} parity={case['parity']} "
        f"median_us={medians['baseline']:.3f}->{medians['direct']:.3f} "
        f"mean_us={means['baseline']:.3f}->{means['direct']:.3f} "
        f"reduction={case['median_reduction_percent']:+.2f}%"
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmcache-ascend-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["early"],
        choices=("early",),
    )
    parser.add_argument("--layouts", nargs="+", choices=("uniform", "bounded", "mixed"), default=["uniform"])
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--heads", type=int, default=64, help="global attention heads")
    parser.add_argument("--layers", type=int, default=2, help="one planner producer plus shared-plan consumers")
    parser.add_argument("--overlap-retrieval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.9)
    parser.add_argument("--history", type=int, default=16384)
    parser.add_argument("--live-tokens", type=int, default=128)
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--json", type=Path, default=Path("sfa-dcp-tuning.json"))
    args = parser.parse_args()
    args.dcp_size, args.rank, args.device = (int(os.environ[name]) for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if (
        args.dcp_size not in (2, 4, 8)
        or args.heads % args.dcp_size
        or args.heads % 8
        or min(*args.requests, args.heads, args.layers, args.steps, args.trials, args.iterations, args.history) < 1
        or args.warmup_steps < 0
    ):
        parser.error("use DCP=2/4/8, aligned divisible heads and positive sizes")
    for layout in args.layouts:
        try:
            make_inputs(1, 1, args.history, args.live_tokens, args.overlap, layout, args.padding)
        except ValueError as exc:
            parser.error(str(exc))
    if not (args.lmcache_ascend_dir / "benchmark/v1/kv_transfer/load_benchmark_utils.py").is_file():
        parser.error("--lmcache-ascend-dir must be the companion LMCache-Ascend checkout")
    sys.path.insert(0, str(ROOT))
    import torch_npu

    from vllm_ascend import envs
    from vllm_ascend.utils import enable_custom_op

    torch.npu.set_device(args.device)
    if not enable_custom_op():
        raise RuntimeError("vLLM-Ascend native operators are unavailable")
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    control = dist.new_group(backend="gloo", timeout=timedelta(seconds=120))
    result = {
        "scope": (
            "projected queries/raw Top-K -> resident planner + real H2D -> query exchange -> native SFA -> output merge"
        ),
        "excludes": "projection/indexer compute, MoE, sampling and server scheduling",
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "HCCL_OP_EXPANSION_MODE": os.getenv("HCCL_OP_EXPANSION_MODE"),
        "dcp_hccl_host": bool(envs.VLLM_ASCEND_DCP_HCCL_HOST),
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cases": [],
    }
    try:
        for layout in args.layouts:
            for requests in args.requests:
                for variant in args.variants:
                    if not args.rank:
                        print(f"Starting {variant} {layout} batch={requests} DCP={args.dcp_size}", flush=True)
                    case_args = SimpleNamespace(
                        **{
                            **vars(args),
                            "variant": variant,
                            "layout": layout,
                            "requests": requests,
                            "shared_plan": True,
                            "heads": args.heads // args.dcp_size,
                            "profile_dir": args.profile_dir / variant if args.profile_dir else None,
                        }
                    )
                    case = measure_variant(case_args, control)
                    if not args.rank:
                        result["cases"].append(case)
                        args.json.parent.mkdir(parents=True, exist_ok=True)
                        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
                        print(format_case_summary(case), flush=True)
        if not args.rank:
            print(f"Saved full results and per-rank samples to {args.json}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
