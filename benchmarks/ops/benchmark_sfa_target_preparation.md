# SFA target preparation benchmark

This compares the previous causal-mask path with target mask elision on one NPU.
Both execute the same serving resident planner and bridge, then the real
LMCache-Ascend `sfa_copy_i32_s64` registered-CPU transfer. It requires the existing
vLLM-Ascend and LMCache-Ascend native extensions; this change needs no native rebuild.

The timed interval contains raw IndexCache publication, mask/copy where required,
bounded lane packing/restoration, the shared-plan copy when enabled, union,
finalize, membership update, query/metadata bridge copies and completed sparse
H2D. It excludes the indexer's computation, attention, collectives and model-side
CPU scheduling. It measures one producer layer, not a complete decoder step.
Shared-plan consumers already reuse the producer's plan.

## Run

Use an otherwise idle NPU. There is no torchrun or HCCL communicator in this test.
The companion packages must be installed from their matching checkouts.

```bash
cd /workspace/sqh/vllm-ascend
python3 benchmarks/ops/benchmark_sfa_target_preparation.py \
  --lmcache-ascend-dir /workspace/qzy/LMCache-Ascend \
  --requests 1 2 4 8 --steps 32 --warmup-steps 4 --overlap 0.9 \
  --dcp-size 8 --rank 0 --layouts uniform bounded mixed \
  --json sfa-target-preparation.json
```

`--shared-plan` is enabled by default, matching shared resident planning in serving.
Use `--no-shared-plan` for an independent producer. Use `--resident` to measure a
zero-frontier request: no historical KV is copied, but the transfer kernel still
executes. Use `--padding 1 --layouts bounded mixed` to include an idle request lane.
The uniform layout has two query rows per request. Mixed alternates one and two;
all layouts retain the same 4096-slot scratch reservation.

`--overlap` controls the fraction of each query row's global Top-K IDs retained
between consecutive steps (rounded to an integer count). The two query rows have
independent selections; overlap is not an assumed rank-local resident hit rate.
The reported per-step misses and copied bytes come from the native planner.

For a short run with traces:

```bash
python3 benchmarks/ops/benchmark_sfa_target_preparation.py \
  --lmcache-ascend-dir /workspace/qzy/LMCache-Ascend \
  --requests 1 --steps 16 --trials 3 --iterations 3 \
  --profile-dir sfa-preparation-traces --json sfa-preparation-b1.json
```

The `old` trace includes `_mask_topk`; `new` omits it. Both must contain the resident
kernels, query bridge and real `sfa_copy_i32_s64`. The shared/bounded cases also
avoid the redundant preliminary Top-K copy. The generic draft/native path keeps
masking, so its mask can still appear in a complete serving profile.

## Correctness and measurement

- Inputs are precomputed causal INT32 Top-K, identical for old/new. They exercise
  historical and live IDs; source generation is outside timing.
- Old/new own separate persistent state, planner workspaces, bridge and KV buffers.
  Registered CPU sources are immutable and retained until both graphs are reset.
- Before timing, every step compares defined planner outputs and resident state,
  and verifies the actual copied K/PE values against the registered CPU source.
- Captured replay is compared with that eager trajectory. Each timed sample resets
  state and replays the same warmup prefix outside its event interval. This avoids
  timing repeated identical inputs that accidentally turn into an all-hit workload.
- Samples alternate old/new order. Times are NPU-event elapsed microseconds for
  the full captured sequence divided by measured steps. The copy completes before
  the end event. Profiling and verification readbacks are outside timing.
- `copied_bytes` is for the entire measured sequence, for one DCP rank. Old/new
  must have identical misses, slots and copied values before results are printed.

The benchmark intentionally uses synthetic indexer outputs to isolate the requested
interval. Qualify the actual indexer backend and graph replay separately:

```bash
export LMCACHE_ASCEND_SOURCE_DIR=/workspace/qzy/LMCache-Ascend
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest --noconftest -o addopts= -q \
  tests/e2e/singlecard/test_sfa_target_preparation.py
```

These tests cover local BF16/C8 indexers, changing Q1/Q2 and short/long lengths,
padding and an idle batch, then the complete preparation/transfer chain. They are
test-only; no probe or verification scan is added to worker startup or graph replay.
Serving throughput still requires a full-model comparison on the target system.
