# Prefill corruption fix and diagnostic cleanup

The serving fix uses the native SFA V-up projection backport. Its branch log,
`SFA_VUP`, remains enabled once per process. The projection implementation and
its compatibility limits are unchanged by this cleanup.

## Retired diagnostics

- KVFENCE modes (`p_load`, `p_sfa`, `p_use`, `p_vup`, reuse/submit/save and decode
  probes), their environment controls, event traces, and payload sampling.
- KV hashes, byte features, CPU/NPU content fingerprints, audits, and tensor
  snapshots, including target-layer and MTP draft captures.
- Prefill overlap, cold-load, startup, sampling, and serving timing wrappers.
- Diagnostic top-k transfer limits, sparse-transfer watchdogs, policy prints,
  and the stage-3 path that intentionally skipped loading scratch KV.
- Tests and launchers specific to retired runtime diagnostics. Standalone
  numerical probes and offline readers for previously saved reports remain.

Ordinary operational warnings, errors, and upstream logging remain. DMA
completion waits previously wrapped in timing code execute at the same point.

## Earlier attempts and disposition

| Change | Disposition | Reason |
| --- | --- | --- |
| Group-0 bank event delayed to the next layer (`a226d01db`) | Already reverted by `12b72ca73` | The serving comparison still showed corruption. |
| Group-1 bank event delayed to the next layer (`748f9fe8c`) | Already reverted by `12b72ca73` | Restore the original event boundary for both groups. |
| L0C initialization experiment (`64a7b224d`) | Already reverted by `56a1a80f4` | Retire the earlier experimental change. |
| Default host wait before P bank reuse (`2689771`, LMCache-Ascend) | Already reverted by `4fa23c1` | Keep the normal event-based bank handoff. |
| Cross-bank source joins | Already removed by `03cf1d6`, LMCache-Ascend | Preserve the current bank-local dependency model. |
| Runtime diagnostic fences and tracing | Removed in this cleanup | The native V-up fix is retained; diagnostic synchronization and readbacks are retired. |
| QKV waits for its bank (`12038c21d`) | Kept | QKV writes must follow completion of that bank's preceding transfer. |
| ACLNN workspace lifetime (`e0bed7b3a`) | Kept | Retain workspace ownership through consumer submission. |
| Legacy V-up first-allocation fence (`3e64d22a2`) | Kept | Protect the still-callable legacy custom operator from direct tiling writes into reused storage. |
| Native SFA V-up (`1d712d835`) and startup compatibility (`5c4897c31`) | Kept | This is the final serving path, with operator capability/shape fallback. |
| `SFA_VUP` branch log (`45cfe2a24`) | Kept | Identify the selected projection branch in actual serving. |
| Existing first sparse-load synchronization (`67d4c5718`, `90d6c5c73`, July 2026) | Kept | Preserve this earlier default behavior; it is separate from the prefill diagnostic fences. |

Other retained correctness changes cover all-request bank joins, D2H completion
before CPU publication/reuse, failed-transfer owner retention, fixed bank
mapping, P/D isolation, and allocation/admission failure handling.

## Why V-up was changed

The legacy operator allocated a tiling tensor from the device allocator and
wrote it with a direct host-to-device copy. Reused storage could still have a
queued reader on the compute stream. The cold-allocation probe observed exactly
72 changed bytes containing the expected tiling signature; synchronized,
retained-owner, and warm-allocation controls were clean. The native SFA V-up
backport avoids that legacy tiling initialization path.

## Local validation

On Windows with CPU PyTorch, the standalone suites passed:

- vllm-ascend: 1,400 passed, 14 skipped.
- LMCache: 354 passed.
- LMCache-Ascend: 545 passed, 6 skipped. Tests requiring installed vLLM/NPU
  package integration were excluded from this standalone run.
- SFA graph preparation, retrieval overlap, full graph, and async MTP:
  198 passed, 3 skipped.
- Sampling masks: 37 passed.

Validation covers bank rotation, DMA ordering, source-owner retention,
publication and error paths, native V-up CPU references, and graph dispatch.
The `_v_up_proj` AST matches the pre-cleanup version. Changed Python files pass
parsing and undefined-name/unused-import checks. The full `bash format.sh ci`
run stopped while installing actionlint: downloads from `proxy.golang.org`
timed out. NPU serving validation belongs to the target Ascend environment.

This cleanup changes Python code, tests, and tools. Pull the matching `dev-lmy`
updates for vllm-ascend, LMCache, and LMCache-Ascend, then restart P/D workers.
It introduces no C++ build change.
