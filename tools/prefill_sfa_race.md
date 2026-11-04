# SFA / bank handoff numerical reproduction

`prefill_sfa_race.py` exercises the installed **real NPU SFA operator**, the
LMCache-Ascend raw H2D/D2H copy operator, and the connector's production
per-request load waits and final-reader handoffs. It does not load model
weights, start services, or alter production code.

The default workload mirrors the failing three-request prefill tail:

- 499712 history tokens per request; new token counts 378, 385, 383.
- 79 layer calls, 4 rotating banks, top-k 2048, 16 query heads.
- BF16 latent width 512 and RoPE width 64; reverse physical request layout.
- Next layer entry submits the preceding layer's save and future bank load.

Run inside the usual P container with its CANN environment, on an **idle NPU**:

```bash
cd /workspace/sqh/vllm-ascend
TASK_QUEUE_ENABLE=1 ASCEND_LAUNCH_BLOCKING=0 \
  /usr/local/python3.11.14/bin/python3 tools/prefill_sfa_race.py \
  --device 0 --repeat 1 2>&1 | tee prefill-sfa-race.log
```

Device 0 is an example; select an idle device. The default explicit buffers
need about 8 GiB device memory, plus operator/runtime workspace. A diagnostic
rerun needs approximately another 3 GiB. Immutable pinned CPU sources need
about 1.6 GiB. All layer inputs are views into shifted immutable sources,
not 79 complete copies of the history. Sizes are printed before allocation.
No service environment or debug flags are changed by this tool.

## What is checked

Layer, request, token and component determine each value. A real NPU matmul
produces zero queries, giving an independent attention oracle: the mean of
selected BF16 values. Top-k indices are fixed, causal, unique, and spread
across history; each query also selects its own newly computed KV row.
Attention uses an absolute-plus-relative numerical tolerance. Raw saved KV
uses exact equality because it is a copy of explicitly constructed values.

The script first runs a synchronized control and stops if its oracle fails.
It then automatically runs ordinary events, `p_sfa`, and `p_use`, with no host
payload reads until each complete run finishes. It compares **every output
element**, every saved new-KV element, and eight saved history rows per request.
The normal asynchronous run adds no input-observation kernels.

If ordinary events produce a mismatch, an automatic diagnostic rerun records:

| Stage | Data checked |
| --- | --- |
| `after_h2d_history` | All history rows selected by SFA, on the DMA stream after H2D |
| `before_sfa_history` | All selected history and all new KV rows, on the compute stream |
| `after_sfa_history` | The same KV rows after SFA |
| `query` | All query elements |
| `sfa_output` | All attention output elements |
| `save_new_kv` | All new KV elements after D2H |
| `save_history_probe` | Eight history rows after D2H |

The diagnostic rerun adds device copies but no CPU waits. If the mismatch
disappears in that rerun, its observations cannot explain the failed earlier
run; the report keeps both results. First errors include layer, request,
logical token, tensor index, expected value and actual value.

Terminal output is compact; full results are saved to `prefill-sfa-race.json`.
Exit status 0 means this fixture did not reproduce a mismatch, 1 means at
least one asynchronous case failed, and 2 means the synchronized control
failed. The final status explicitly says `not_reproduced` instead of claiming
that the serving bug is fixed.

## Scope

This isolates SFA and bank handoffs. The top-k is synthetic: it does **not**
exercise the GLM indexer, quantized projections, FFN, graph capture/replay,
TP collectives, LMCache allocation/eviction, or P-to-D transport. Matching
results constrain this reproduction only. They do not certify those paths.
The CPU tests validate the fixture, oracle and injected load/save/output
damage; hardware runs are required to test actual NPU execution ordering.
