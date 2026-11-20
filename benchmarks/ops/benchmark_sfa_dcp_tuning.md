# Early DCP query gathering

The `early` candidate preserves one query all-gather and one output all-to-all
per attention layer. Native attention, the 1,056-byte FP16/BF16 packet and the
original weighted combine are unchanged. The unsuccessful raw-packet and
combine-tiling experiments have been removed.

Early submission keeps the packet, receive buffer and work handle in a local
context until the communication stream rejoins. No per-layer global pending
state, temporary probe, or extra collective is introduced. In serving, submit
occurs immediately after query projection/RoPE, before indexer postprocessing.
The ordinary full-attention call owns cleanup, including preparation failures.
Prefetched shared-index consumers use late gathering, because their load joins
before projection and their plan is already available. Consumers without an
incoming prefetch edge retain early gathering to overlap their serial H2D load.

## Benchmark

Use idle NPUs, with the normal native operators already built. This change
requires no C++ rebuild. The benchmark uses the companion LMCache-Ascend sparse
transfer implementation and registered host memory, as the copy-reduction
benchmark did. Set `--lmcache-ascend-dir` to its actual checkout.

Start with one batch:

```bash
cd /workspace/sqh/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
HCCL_OP_EXPANSION_MODE=AIV TASK_QUEUE_ENABLE=1 HCCL_BUFFSIZE=256 HCCL_EXEC_TIMEOUT=60 \
python3 -m torch.distributed.run \
  --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  benchmarks/ops/benchmark_sfa_dcp_tuning.py \
  --lmcache-ascend-dir /workspace/qzy/LMCache-Ascend \
  --requests 1 --variants early --steps 2 --warmup-steps 1 \
  --trials 2 --iterations 2 --json dcp-tuning-smoke.json
```

For the full comparison replace the final three lines with:

```bash
  --lmcache-ascend-dir /workspace/qzy/LMCache-Ascend \
  --requests 1 2 4 8 --variants early --layers 4 \
  --steps 16 --warmup-steps 4 --trials 10 --iterations 10 \
  --json dcp-tuning.json
```

The paired baseline uses the same optimized bridge path and late gathering.
The JSON `direct` arm means early gathering; positive
`median_reduction_percent` means improvement. The optional `--variants early`
argument is retained for existing commands; `early` is the only valid value.
Console output contains one short `RESULT` line per case; full per-rank timing
samples and miss counts are saved only in the JSON file. Use `rg '^RESULT'`
on the console log to extract the summaries. The JSON file is updated after
every completed case, so existing results can be summarized without rerunning.
The benchmark selects gathering timing explicitly and does not depend on the
serving knob below. `VLLM_ASCEND_DCP_HCCL_HOST` still controls the communicator;
unset it for an AIV comparison.

Useful additional cases:

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
Prefetched consumers use the original late gather, matching production's
dependency order. Use `--layers 4` for a model with one indexer in each four-layer
group; the default remains two layers for compatibility with earlier results.

Samples report microseconds per step for the entire layer group (two layers by
default), including communication completion. They divide graph time by steps,
not layers, and use the slowest rank for each paired sample. Trials alternate
arm order. Reset, warmup, validation and Gloo barriers are outside timing.
Eager and repeated graph replay must preserve attention outputs, planner state,
miss counts and retrieved data before timing starts. All real graphs remain
alive during the comparison; their communicator is not reused after cleanup.

## Serving control

Set before restarting workers. This affects DCP>1 only and defaults to disabled:

```bash
export VLLM_ASCEND_SFA_DCP_EARLY_GATHER=1
```

Remove the retired settings from older launch scripts:

```bash
unset VLLM_ASCEND_SFA_DCP_RAW_PACKET VLLM_ASCEND_SFA_DCP_COMBINE_TILE
```

Early gathering applies to full target graphs with active shared resident plans;
staged islands, eager fallbacks, diagnostics and MTP retain late submission.
Use identical settings on every rank.

## Further optimization boundary

The normal path joins the gather only before unpacking its result, and failure
cleanup does not issue a second device wait after a successful join. Moving
submission earlier would require rescheduling query projection relative to KV
preparation; the existing order releases fresh KV temporaries before projecting
queries. A faster schedule is not established by the current measurements.

The separate [query-first consumer benchmark](benchmark_sfa_dcp_query_first.md)
includes projections and current-KV writes to measure this reorder. Its opt-in
`VLLM_ASCEND_SFA_DCP_QUERY_FIRST` setting extends early gathering to prefetched
consumers; the benchmark on this page keeps the original KV-first schedule.

Early gathering has no explicit AIV budget. The current-layer sparse H2D load
uses its automatic launch width; the existing 12-block limit applies only to
next-layer prefetch and does not reserve disjoint physical cores. Profile the
whole attention path and individual kernel durations before adding core limits
or changing submission order. Keep logical resident sharding and Top-K fixed
when comparing resource allocation.

## Qualification

Host tests cover DCP2/4/8, FP16/BF16 bits, empty shards, large maxima/sums,
one-collective routing and asynchronous source lifetime.
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
statistics and compilation of the original pack/combine kernels without HCCL.
Full-server qualification and throughput measurements remain separate from
this attention benchmark.
