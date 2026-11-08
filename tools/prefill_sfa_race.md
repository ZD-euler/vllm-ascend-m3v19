# SFA / bank handoff numerical reproduction

## CPU preflight before handing off an NPU command

Run the complete CPU regression suite before asking for a device run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python3 -m pytest -q \
  --confcutdir=tests/standalone tests/standalone/test_prefill_sfa_race.py
```

The suite invokes the actual command parser and main loop with both focused
commands below, including 64 heads, 12 layers, four banks and two repeats.
It checks data construction, the production connector's bank methods, every
result, saved KV, JSON reports and exit codes. Injected errors also exercise
serial-control failure, selected-mode failure and the event audit rerun.
CPU operator substitutes evaluate zero-query attention from the actual bank
contents and the actual one-hot projection weights; separate tests compare
them with dense CPU operations. Shared views keep the repeated CPU values
compact while preserving the full logical tensor shapes. Real NPU kernels,
runtime initialization and asynchronous device execution are device checks.

## Focused run after `p_use` passes and `p_vup` fails

Run this bounded operator check in the serving container on an idle NPU.
It uses the installed SFA operator, the serving `_v_up_proj` method, and
the production bank event handoff. The 64 heads match GLM-5.3 with DSA-CP.
Both V-up branches are exercised: 384 local tokens use the custom transpose
matmul, and 1536 local tokens use `torch.bmm`. Four banks and 12 layer calls
exercise repeated bank reuse. Model weights and service startup are absent.
The short-history generator supports `history=8192` with `topk=2048` by
widening the two sampled residue bands while retaining distinct causal
indices and a detectable layer fingerprint across bank reuse.

From the updated `vllm-ascend` checkout, with its usual CANN environment:

```bash
export TASK_QUEUE_ENABLE=1
export ASCEND_LAUNCH_BLOCKING=0

for case in small large; do
  if [ "$case" = small ]; then
    queries="128 127 129"
  else
    queries="512 511 513"
  fi
  timeout --signal=INT --kill-after=10s 150s python3 tools/prefill_sfa_race.py \
    --device 0 --history 8192 --layers 12 --banks 4 --heads 64 \
    --queries $queries --repeat 2 \
    --modes p_use p_vup before_vup p_vup_hold \
    --output "sfa-vup-${case}.json" > "sfa-vup-${case}.log" 2>&1
  status=$?
  echo "$case exit=$status"
done
```

Use a fresh output directory for each run. Each case has a 150-second budget
plus a 10-second termination grace period. `exit=124` marks a time budget
expiration, while exit 1 reports numerical differences and exit 2 reports
invalid command-line arguments or a failed serial oracle. Inspect the log
to distinguish these cases. Partial JSON is
saved after each completed mode.

The extra names below are this tool's `--modes` values:

| Mode | Intervention |
| --- | --- |
| `p_use` | Wait for the SFA completion event before handing it to bank DMA. |
| `p_vup` | Hand off the event, submit V-up, release the original SFA output, then wait for the compute stream. |
| `before_vup` | Hand off the event and submit any final-layer save, then wait for the same SFA event immediately before V-up. |
| `p_vup_hold` | Use exactly the `p_vup` waits, retaining the current layer's SFA output through that wait. Release it before starting the next layer. |

When the fixture reproduces the serving pattern (`p_use` passes and `p_vup`
fails), `before_vup` passing focuses investigation on projection submission
and input lifetime; `before_vup` failing focuses it on bank handoff and final
save submission. `p_vup_hold` passing provides a direct lifetime intervention
for the original SFA output, with the same synchronization as `p_vup`.
Compare modes within the same token shape. If all cases pass, retain the
serving workload's allocation, CP communication and input context in the next
diagnostic. The operator fixture is then a regression test for its covered
shapes, rather than the serving reproducer.

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
