# Query-first DCP consumers

This opt-in candidate extends early query gathering to shared-index consumer
layers. After the shared QKV projection it runs Q normalization, Q projection
and query RoPE, submits the existing asynchronous query gather, joins any
incoming retrieval, and then runs current-KV normalization/RoPE/cache writes.
The gather is joined before unpacking and native attention. No attention kernel,
packet layout, Top-K, resident capacity or collective count changes.

Q never reads the KV cache. Incoming retrieval must still finish before cache
writes; dummy warmup cache zeroing retains an earlier wait for that reason.
Failure cleanup consumes an outstanding incoming edge and query gather once.
Producer layers, alternate context parallelism, DCP=1, MTP, diagnostics and
non-full-graph paths retain their previous schedules.

The possible gains are overlap of Q computation with unfinished prefetch and
overlap of gathering with current-KV work. AIV contention can erase either gain.
Query tensors also remain live across KV preparation. Neither a speedup nor a
production memory result is established by host tests.

## Whole-group benchmark

Run on idle NPUs with the existing native operators built. No C++ rebuild is
needed. The baseline already has early gathering on the producer (and on
consumers with serial H2D). The `direct` JSON arm additionally uses Q-first
consumers. Thus this comparison measures the incremental change.

```bash
cd /workspace/sqh/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
HCCL_OP_EXPANSION_MODE=AIV TASK_QUEUE_ENABLE=1 HCCL_BUFFSIZE=256 HCCL_EXEC_TIMEOUT=60 \
python3 -m torch.distributed.run \
  --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  benchmarks/ops/benchmark_sfa_dcp_query_first.py \
  --lmcache-ascend-dir /workspace/qzy/LMCache-Ascend \
  --requests 1 --layers 4 --steps 2 --warmup-steps 1 \
  --trials 2 --iterations 2 --json query-first-smoke.json
```

Then use `--requests 1 2 4 8 --steps 16 --warmup-steps 4 --trials 10
--iterations 10 --json query-first.json`. Each case prints one short result;
full per-rank samples remain in JSON. Positive reduction means faster. Reported
microseconds cover the complete four-layer group per step, including completed
communication, rather than a single kernel. Both arms' captured graphs stay
alive until the case finishes; their communicator is never reused afterward.

Timing includes:

- Shared QKV projection, Q RMSNorm/projection/BMM and query RoPE.
- The production fused KV RMSNorm/RoPE/cache-write operation, with DCP-owned
  current-token slots and padding masked to `-1`.
- The actual resident planner, registered CPU sources and sparse H2D, including
  the real next-layer retrieval stream and its existing 12-block launch width.
- Query packing/gather/unpack, native counted SFA, output packing/all-to-all
  and weighted combine.
- Local value/output projection and a residual, feeding the next layer. This
  gives retrieval real post-attention work to overlap before the next consumer.

Projection weights are deterministic synthetic BF16, with different query
weights on each rank, not the model's quantized weights. Set `--hidden-size`,
`--q-rank`, `--qk-nope`, `--value-dim` and global
`--heads` to the model dimensions. Defaults are 7168/1536/128/128/64. Actual
indexer computation, TP output reduction, MoE, sampling and scheduling are
excluded. These exclusions and the lack of intervening MoE can change incoming
prefetch slack; full-server throughput/profile remains the final check.

Top-K includes a current token so KV writes affect attention. Before timing,
the benchmark compares exact eager and repeated graph outputs, planner state,
misses, H2D payloads, current KV and local projected outputs. Verification,
reset, warmup and Gloo barriers are outside the timing interval. It retains the
fused KV operation's returned tensors for verification in both arms, adding no
verification kernel to the timed interval. This modest extra lifetime is not
intended as a production memory measurement.

Additional controls:

- `--layers 1`: no consumers; the two schedules should have no systematic gap.
- `--layouts mixed --padding 1`: varying query counts and padded graph rows.
- `--no-overlap-retrieval`: measure current-KV overlap without incoming prefetch.
- `--overlap 0.5`: increase retrieval turnover and expose AIV contention.
- `--resident`: suppress historical H2D traffic.
- `--profile-dir profiles/query-first`: inspect gather/KV/H2D contention in
  separate arm/rank traces. Compare whole-group time as well as kernel time.

## Serving and qualification

Keep both settings identical on all ranks and restart workers:

```bash
export VLLM_ASCEND_SFA_DCP_EARLY_GATHER=1
export VLLM_ASCEND_SFA_DCP_QUERY_FIRST=1
```

`QUERY_FIRST` defaults to `0` and requires `EARLY_GATHER=1`. It affects only
full target DCP graphs with active shared plans, a shared-index consumer and no
local indexer/alternate CP path. Disable `QUERY_FIRST` to recover the previous
schedule. No disposable startup probe or CPU synchronization is added.

The NPU regression uses the same whole-group runner with small projections,
mixed/padded rows, real H2D, prefetch on/off and repeated replay:

```bash
LMCACHE_ASCEND_DIR=/workspace/qzy/LMCache-Ascend \
HCCL_OP_EXPANSION_MODE=AIV TASK_QUEUE_ENABLE=1 HCCL_EXEC_TIMEOUT=60 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m torch.distributed.run \
  --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  --module pytest --noconftest -o addopts= -q -s \
  tests/e2e/multicard/test_sfa_dcp_query_first_npu.py
```
