# DCP packet, combine tiling and early gather

These candidates preserve one query all-gather and one output all-to-all per
attention layer. Native attention and the mathematical weighted merge are
unchanged. All serving controls default to the existing behavior.

| Candidate | Change |
| --- | --- |
| `raw` | INT32 packet containing 256 raw output words, exact FP32 max/sum words and six padding words |
| `tile128` | Four 128-feature combine tiles per output head |
| `tile256` | Two 256-feature combine tiles per output head |
| `early` | Submit query gathering before resident planning/retrieval; join before unpack/SFA |
| `raw_tile128` | Raw packet plus 128-feature combine tiles |
| `combined` | Raw packet, 128-feature combine tiles and early gathering |

Both packet formats use 1,056 bytes per head. The raw representation removes
numerical byte encoding/decoding without rounding either FP32 statistic. The
tiled combine preserves rank accumulation order and normalization before
multiplication, including all-empty rows and large finite values.

Early submission keeps the packet, receive buffer and work handle in a local
context until the communication stream rejoins. No per-layer global pending
state, temporary probe, or extra collective is introduced. In serving, submit
occurs immediately after query projection/RoPE, before indexer postprocessing.
The ordinary full-attention call owns cleanup, including preparation failures.

## Benchmark

Use idle NPUs, with the normal native operators already built. This change
requires no C++ rebuild. The benchmark uses the companion LMCache-Ascend sparse
transfer implementation and registered host memory, as the copy-reduction
benchmark did. Set `--lmcache-ascend-dir` to its actual checkout.

Start with one candidate and batch:

```bash
cd /workspace/sqh/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
HCCL_OP_EXPANSION_MODE=AIV TASK_QUEUE_ENABLE=1 HCCL_BUFFSIZE=256 HCCL_EXEC_TIMEOUT=60 \
python3 -m torch.distributed.run \
  --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  benchmarks/ops/benchmark_sfa_dcp_tuning.py \
  --lmcache-ascend-dir /workspace/qzy/LMCache-Ascend \
  --requests 1 --variants raw --steps 2 --warmup-steps 1 \
  --trials 2 --iterations 2 --json dcp-tuning-smoke.json
```

For the full comparison replace the final three lines with:

```bash
  --lmcache-ascend-dir /workspace/qzy/LMCache-Ascend \
  --requests 1 2 4 8 --variants raw tile128 early combined \
  --steps 8 --warmup-steps 2 --trials 10 --iterations 10 \
  --json dcp-tuning.json
```

Every candidate gets a paired baseline using the current optimized bridge
path. The JSON `direct` arm means the named candidate here; `baseline` has all
three changes disabled. Positive `median_reduction_percent` means improvement.
The benchmark passes explicit options and does not depend on the three serving
knobs below. `VLLM_ASCEND_DCP_HCCL_HOST` still controls the actual communicator;
unset it for an AIV comparison.

Useful additional cases:

- `--variants tile256 raw_tile128` isolates tiling size and packet/tiling interaction.
- `--layouts bounded mixed --padding 1` exercises irregular and padded rows.
- `--layers 1` isolates a planner-producing layer; `--layers 4` adds more consumers.
- `--no-overlap-retrieval` separates query overlap from existing next-layer retrieval overlap.
- `--resident` removes miss retrieval traffic; `--overlap 0.5` increases turnover.
- `--profile-dir profiles/dcp-tuning` exports separate variant/arm/rank traces.

Timing includes query packing, all-gather and unpack, the real resident planner,
real sparse H2D, native counted SFA, output packing/all-to-all and weighted
combine. It excludes projection/indexer compute, output projection, MoE,
sampling and server scheduling. Query and raw Top-K inputs are precomputed;
early gathering overlaps actual planner/H2D work, with no synthetic delay.
Thus it does not measure the extra overlap available during a real indexer.
Consumer queries start gathering only after the previous retrieval edge joins,
matching production's dependency order.

Samples report microseconds per step for the entire layer group (two layers by
default), including communication completion. They divide graph time by steps,
not layers, and use the slowest rank for each paired sample. Trials alternate
arm order. Reset, warmup, validation and Gloo barriers are outside timing.
Eager and repeated graph replay must preserve attention outputs, planner state,
miss counts and retrieved data before timing starts. All real graphs remain
alive during the comparison; their communicator is not reused after cleanup.

## Serving controls

Set independently before restarting workers, after qualifying on the deployment
runtime. Flags affect DCP>1 only:

```bash
export VLLM_ASCEND_SFA_DCP_RAW_PACKET=1        # default 0
export VLLM_ASCEND_SFA_DCP_COMBINE_TILE=128    # default 512; valid 128/256/512
export VLLM_ASCEND_SFA_DCP_EARLY_GATHER=1      # default 0
```

Raw packets and tiling apply to both target and draft DCP attention. Early
gathering applies to full target graphs with active shared resident plans;
staged islands, eager fallbacks, diagnostics and MTP retain late submission.
The packet layout is process-group-wide: use identical settings on every rank.

## Qualification

Host tests cover DCP2/4/8, FP16/BF16 bits, all three tile sizes, empty shards,
large maxima/sums, one-collective routing and asynchronous source lifetime.
They do not compile Triton or run HCCL. The NPU test changes queries and counts
between replays and checks both baseline parity and a dense CPU reference:

```bash
HCCL_OP_EXPANSION_MODE=AIV TASK_QUEUE_ENABLE=1 HCCL_EXEC_TIMEOUT=60 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m torch.distributed.run \
  --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  --module pytest --noconftest -o addopts= -q -s \
  tests/e2e/multicard/test_sfa_dcp_tuning_npu.py
```

The single-NPU `test_dcp_exchange_pack_combine_capture` cases in
`tests/e2e/singlecard/test_sfa_dcp_merge_npu.py` additionally cover extreme
statistics and compilation of each packet/tile combination without HCCL.
Full-server qualification and throughput measurements remain separate from
this attention benchmark; no speedup is assumed from fewer encoding operations.
