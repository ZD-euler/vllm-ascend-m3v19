# Independent DP execution for full SFA

This experimental mode removes the cross-DP **host metadata/preparation vote**
from live target and draft forwards. Each DP selects its own existing graph bucket
or compatible native MC2 path. TP-local checks, EP dispatch/combine, startup capture
coordination and the EngineCore finish check every 32 steps remain.

The TP route check uses a dedicated Gloo process group, created once during
activation and destroyed by the existing worker distributed teardown. It does
not share collective sequencing with LMCache's background TP broadcasts and
readiness reductions. No additional per-step collective is introduced.

The proposal uses cached CPU storage views, retaining the same packet checks and
target/draft sequence. Forward contexts reuse the existing CPU token maximum.
Regular MC2 omits the optional token mask because its dispatcher does not consume
it; other communication methods retain their mask behavior. These optimizations
need no additional configuration or native rebuild. CPU timings do not establish
NPU serving speedups; qualify the installed stack after restarting the decoders.

It is default off. It does not establish that a slower DP can no longer delay
peers inside expert communication, or guarantee a throughput improvement.

## Deployment

Install matching modified `vllm` and `vllm-ascend` Python sources on every decoder
host. These changes do not require rebuilding the C++ extensions or changing the
installed Mooncake library. Restart the whole decoder DP/EP cohort together.

```bash
export VLLM_ASCEND_SFA_INDEPENDENT_DP=1
export VLLM_ASCEND_SFA_SINGLE_AGREEMENT=0
# Optional guardian health lease; default 30 seconds, valid range [2, 86400].
# This does not replace the existing model-execution/startup timeout settings.
export VLLM_ASCEND_SFA_GUARD_TIMEOUT_S=30
```

Keep the existing full-SFA, KV, graph and MTP configuration. The flag must match
on all DP peers. Startup logs `[SFA independent DP] active` only after every TP
worker installs the common qualified mode. Static ineligibility chooses the
established protocol for the whole cohort; mixed flags/contracts fail startup.

The scope is Linux MP execution, one host per TP group, fixed EP membership,
PP=PCP=1, no DBO, regular MC2 V2, and either no speculation or the existing
one-token MTP schedule. DCP=1 supports A2/A3. Decoder DCP additionally supports
the existing A2 two-group layout with local DCP=TP in {2, 4, 8}, 128-token
interleave, and full SFA graphs. Generic context parallelism, external-launcher
DP, different draft TP subgroups and fused/hierarchical MC2 remain outside this
mode. DP must exceed one; setting this flag with DP=1 provides no optimization.

For decoder DCP, restart with the two flags above and the existing DCP/graph/cache
settings. No new knob or native rebuild is needed. With independent DP requested,
the recompute scheduler bounds recovery to the existing MC2 query-token capacity;
this budget is **not divided by DCP size**. Coordinated DCP retains its original
recovery budget. Bounded recovery can require more prefill/recovery steps, so
compare recovery latency as well as steady decoding. Worker qualification checks
the actual DCP/TP rank ordering, selected decoder backend, MC2 capacity, DCP
communication/overlap settings and relevant source fingerprints before activation.
All DCP query gathers/result exchanges and the TP-only route check remain; no
additional per-layer host checks, device copies or synchronization are introduced.

CANN 8.5.1 remains the target stack: unequal-token dispatch/combine is documented
in [CANN 8.5](https://www.hiascend.com/document/detail/en/canncommercial/850/API/aolapi/context/ops-transformer/aclnnMoeDistributeDispatchV2.md),
and the implementation retains the existing default algorithm
arguments. It does not copy newer A3 algorithm names or require a CANN upgrade.
Installed torch-npu, CANN/driver versions and native binaries still need to be
recorded and tested together.

Startup compares the loaded dispatcher, Python sources and content fingerprints
of mapped torch-npu, HCCL, runtime and custom-op libraries. File device/inode checks
reject a replaced mapping. If OverlayFS reports different `fstat` identities,
a temporary private mapping verifies the opened file's backing device/inode
before hashing; no file bytes are written. Unavailable identity selects the common
legacy path; differing identities fail activation. Hashing runs once after capture and reads
the native files on each worker, so it adds startup I/O. It does not hash weights
or run during decoding, and it does not prove driver or NPU-kernel compatibility.

The guardian TCP endpoint uses vLLM's existing host-IP selection. On multi-NIC
hosts, keep `VLLM_HOST_IP` reachable from every decoder host. No per-step worker
traffic uses that endpoint.

## Ownership and failure

One CPU-only guardian per executor supervises the registered TP workers. It does
not load torch/NPU libraries, own model/cache data, or send per-step worker messages.
The executor supplies fixed-rate control and observes existing RPC deadlines.
Workers send only registration and fatal-error notifications.
Only the executor's current outstanding-operation set decides RPC expiry. The
guardian enforces its health lease; it does not infer a timeout from an older
heartbeat after the executor may already have consumed the successful response.

Faults terminate the cohort; they do not trigger an independent-rank restart or
switch a live rank back to CPU agreement. Worker/process death is not proof of DMA
completion. Existing LMCache owners retain uncertain leases and quiesce on parent
loss; the guardian does not delete slabs or fabricate successful retirement.

Epoch markers live under `${TMPDIR:-/tmp}/vllm-dp-guardian/`. Successful common
fallback or completed group drain removes its marker. A failed epoch leaves it
in place. Before removing a failed marker and restarting, establish that the
original resource owners have drained, or perform the necessary device/transport
fencing. The marker is a restart barrier for this deployment, not a machine-wide
device fencing service.

## Qualification

Desktop CPU tests cover the actual control protocol, subprocess/TCP lifecycle,
selected runner methods and failure ordering. They do not qualify CANN graph
replay, DMA, full-server correctness or performance.

Before NPU qualification, run the CPU Gloo overlap regression on Linux. It uses
four local CPU processes to exercise two independent TP groups alongside cache
broadcasts and readiness reductions:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest --noconftest -q \
  tests/ut/compilation/test_sfa_independent_collectives.py
```

On both test hosts, run the installed-library probe on otherwise available NPUs:

```bash
# Use a reachable rendezvous address and a different node rank (0 or 1) per host.
torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29617 --module pytest \
  --noconftest -q tests/e2e/multicard/test_sfa_independent_mc2.py
```

Preserve the deployment's HCCL interface configuration. This probe compares
unequal local inputs and mixed native/captured dispatch/combine sequences with
identity expert computation, including communication quantization. The
`decoder-dcp` case adds real query gathering, counted sparse attention and output
exchange before each MC2 pair on four separate TP4/DCP4 replicas. It compares
against dense attention, includes empty selections and retains graphs across
replays. Its printed three-layer pipeline latency includes launch and device
completion with unequal peer workloads; it is a diagnostic, not a serving
throughput estimate or an independent-versus-coordinated A/B measurement.
It does not exercise LMCache ownership, retrieval or the full server schedule.

To select only the DCP case, replace the pytest path in that command with:

```text
tests/e2e/multicard/test_sfa_independent_mc2.py::test_unequal_mc2_inputs_with_independent_native_and_graph_routes[decoder-dcp]
```

Then qualify the production serving topology: TP4/DP4 with DCP=1 or 4, or
TP8/DCP8/DP2, using identical trace, graph-size list, cache state, model and MTP
settings. Include unequal/idle DPs, Q1/Q2, native recovery, cold
loading, two-group C8 KV, local/Mooncake placement, checkpoint preemption/resumption,
GC disabled and failures during preparation/replay/draft/retirement. Verify all
participants stop on failure and uncertain ownership remains retained. Compare
outputs, finish reasons, MTP acceptance, TTFT, TPOT, throughput, per-rank gaps and
peak/retired memory together. Keep the mode off until that qualification passes.
