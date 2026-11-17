# DCP prefetch after packing

This candidate moves an existing next-layer sparse H2D transfer out of the
current layer's `_pack` interval. It changes only the full-target DCP path with
an outgoing shared-retrieval edge. No native operator rebuild is needed.

```text
Current: SFA -> launch next H2D -> pack -> all-to-all -> combine
New:     SFA -> pack -> submit async all-to-all -> launch next H2D
                    -> wait for all-to-all -> combine
```

The copy stream's existing start event is recorded after packing and before
joining all-to-all. Thus H2D depends on packing but not on communication
completion. Using synchronous all-to-all before recording that event would
serialize the copy behind communication. The explicit work wait orders
`_combine` and protects both packet buffers, including transfer-launch failure.
TorchNPU's [WorkHCCL implementation](https://github.com/Ascend/pytorch/blob/v2.9.0/torch_npu/csrc/distributed/ProcessGroupHCCL.cpp)
provides the communication-to-current-stream dependency.

The consumer still joins the existing H2D completion event before cache access.
This also preserves Q-first consumers' deferred cache wait. Pack/combine math,
FP32 statistics, packets, collective count, source lifetime and the 12-block
prefetch launch width are unchanged. There is no additional event, probe or
explicit CPU/device synchronization in our per-layer implementation.

Eligible edges include producer-to-consumer and consumer-to-consumer transfers
within a shared-index group. In a qualified four-layer group, the first three
layers can launch the following layer's H2D. The last layer has no such edge.
The first layer's own H2D must still complete before its attention; the next
indexer group's selection is not available for prefetch. DCP=1, ordinary
fallbacks and layers without an outgoing edge keep their previous path.

HCCL AIV and H2D can still contend with each other. A long transfer can extend
past communication and overlap combine/projection, or increase the next
consumer's wait. This removes pack/prefetch overlap; it does not establish a
speedup or guarantee completion within the all-to-all interval.

## Serving

Set identically on every rank before restarting workers:

```bash
export VLLM_ASCEND_SFA_SHARED_RETRIEVAL_OVERLAP=1
export VLLM_ASCEND_SFA_DCP_PREFETCH_AFTER_PACK=1
```

The new flag defaults to `0`. It requires DCP>1 and full target graphs with
active shared resident plans and prepared retrieval edges. It works with Q-first
and early gathering enabled or disabled; keep those settings fixed in A/B runs.

## Whole-group benchmark

Use idle NPUs with the normal native libraries. The benchmark reuses the
[query-first whole-group runner](benchmark_sfa_dcp_query_first.md): real current
KV, resident planning, registered-host H2D, native attention, communication,
merge and local output projections. Both arms use identical Q scheduling;
`baseline` prefetches before pack, while `direct` prefetches alongside async
all-to-all after pack. The serving flag does not select benchmark arms.

```bash
cd /workspace/sqh/vllm-ascend
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset VLLM_ASCEND_DCP_HCCL_HOST
HCCL_OP_EXPANSION_MODE=AIV TASK_QUEUE_ENABLE=1 HCCL_BUFFSIZE=256 HCCL_EXEC_TIMEOUT=60 \
python3 -m torch.distributed.run \
  --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  benchmarks/ops/benchmark_sfa_dcp_prefetch.py \
  --lmcache-ascend-dir /workspace/qzy/LMCache-Ascend \
  --requests 1 --layers 4 --steps 2 --warmup-steps 1 \
  --trials 2 --iterations 2 --json prefetch-smoke.json
```

Use `--requests 1 2 4 8 --steps 16 --warmup-steps 4 --trials 10 --iterations 10`
for longer measurements. One compact `RESULT` line is printed per case; JSON
contains the full per-rank samples. Time is per step for the whole layer group,
including the consumers' completion waits. Positive reduction means faster.

- `--no-query-first` holds consumers on the old KV-first schedule in both arms.
- `--layers 1` or `--no-overlap-retrieval` is a negative control: there is no
  outgoing transfer to reschedule, so both arms use identical paths.
- `--layouts mixed --padding 1` checks irregular/padded rows.
- `--overlap 0.5` increases misses; `--resident` removes historical H2D traffic.
- `--profile-dir profiles/dcp-prefetch` exports separate arm/rank traces.

Check whether `_pack` gets shorter, whether all-to-all/combine get longer, and
whether the next consumer waits longer. The whole-group result decides the
tradeoff. Projection weights are synthetic BF16; actual indexer computation,
TP output reduction, MoE and server scheduling are excluded. Full-server
measurement remains necessary.

The NPU regression covers real H2D, current-KV writes, repeated graphs,
mixed/padded rows, retrieval on/off, and Q-first on/off:

```bash
LMCACHE_ASCEND_DIR=/workspace/qzy/LMCache-Ascend \
HCCL_OP_EXPANSION_MODE=AIV TASK_QUEUE_ENABLE=1 HCCL_EXEC_TIMEOUT=60 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m torch.distributed.run \
  --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29571 \
  --nnodes=1 --node-rank=0 --nproc-per-node=8 \
  --module pytest --noconftest -o addopts= -q -s \
  tests/e2e/multicard/test_sfa_dcp_query_first_npu.py -k prefetch
```
