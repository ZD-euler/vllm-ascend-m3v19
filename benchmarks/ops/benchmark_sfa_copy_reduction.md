# SFA copy reduction benchmark

Compare the parent commit's bridge path with direct shared-plan consumption.
Both variants use the current mask-free planner. Native attention and the DCP
all-to-all packet format are unchanged.

The measured interval covers a complete shared-index layer group:

1. Raw Top-K publication, union, finalize and membership update.
2. Query/payload bridging where required by the selected variant.
3. Real sparse K/RoPE H2D from registered CPU memory.
4. Query all-gather and unpack, native attention, pack, all-to-all and combine.
5. Shared-plan consumers, including their real H2D and attention. With overlap
   enabled, prefetch starts after the previous layer's attention and joins before
   the consumer, using the production `RetrievalEdge` implementation.

It excludes indexer/projection computation, output projection, MoE and server
scheduling. This is a device pipeline benchmark, not a full-model TPOT estimate.
Reset, input generation, host readbacks, warmup and Gloo barriers are outside
timing. Each sample replays a graph containing multiple changing Top-K steps;
it does not repeatedly time a single all-hit resident state.

## Run

Stop other workloads on the selected NPUs. Run from this checkout with the same
Python, vLLM, LMCache and LMCache-Ascend installations as the server. Existing
native extensions are sufficient; this branch changes Python/Triton only.

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export HCCL_BUFFSIZE=256

python3 -m torch.distributed.run \
  --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  benchmarks/ops/benchmark_sfa_copy_reduction.py \
  --lmcache-ascend-dir /workspace/qzy/LMCache-Ascend \
  --requests 1 2 4 8 --heads 64 --layers 2 \
  --steps 16 --warmup-steps 4 --overlap 0.9 \
  --json sfa-copy-reduction-dcp8.json
```

Keep the server's other HCCL/network environment settings. `--heads` is the
global attention head count; use the model's actual value. DCP size comes from
torchrun's world size. `--layers 1` measures a producer alone; `--layers 2` also
measures one shared consumer. Adjust to the model's actual sharing pattern.

For DCP=1, use `--nproc-per-node=1`, select one idle NPU and pass
`--local-heads 8` (or the model's TP-local head count). This tests the ordinary
attention/query-copy path; it does not simulate TP model computation.

Useful additional runs:

- `--no-overlap-retrieval`: serial H2D control.
- `--layouts bounded mixed --padding 1`: mixed Q1/Q2 and padded graph rows.
- `--resident`: zero H2D misses; isolates overhead when history is resident.
- `--profile-dir profiles/sfa-copy-reduction`: export per-rank device traces.
- `--requests 1 --steps 4 --warmup-steps 1 --trials 2 --iterations 2`: short check.

## Validation and results

Before timing, the benchmark checks the cold-to-warm trajectory, planner state,
both layers' retrieved token contents and attention output parity. It then
checks three graph replays against the eager result. Ranks agree on validation
failures before entering another collective.

`slowest_rank_median` reports microseconds per step for the **whole layer group**.
It takes the slowest rank for each sample, then the median. `per_rank_samples`
exposes imbalance. Baseline/direct order alternates between trials. Both real
graphs stay alive during the comparison; a new communicator is created for
each case, and no synthetic startup probe is run.

The production optimization is automatic for full SFA graphs with an active
shared resident plan. DCP keeps projected queries internal until packet
construction; DCP=1 retains its contiguous query bridge. Both consume planner
payloads directly. Legacy staged execution and diagnostic bridges retain their
existing path. The full operator declares shared tensor mutations explicitly,
and uses the same compiler policy that prevents resident workspace clones.
