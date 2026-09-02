# ADR 0004: Fail-closed performance evidence gate

## Status

Accepted.

## Context

FENIX distinguishes trace characterization from performance evidence. Trace mode
intentionally disables CUDA graphs and adds Python-side instrumentation, so its
latency is not a performance endpoint. Separately, the pinned runtime may JIT
compile Triton kernels during the first requests for a shape. A successful HTTP
request therefore does not by itself establish a publishable timing sample.

`configs/campaign.json` predeclares warmup requests, measured requests,
concurrency, and repetition policy. Before this ADR, the client benchmark could
measure requests but did not enforce that warmup and measured phases were
separate or that the measured server-log interval was free of JIT/runtime
contamination.

## Decision

Performance collection uses `python -m scripts.performance_evidence`.

The command is fail-closed and requires:

1. a clean FENIX Git worktree;
2. a server log containing the startup-complete marker;
3. an unambiguous `FENIX_TRACE=0` launch;
4. at least one successful warmup request;
5. successful measured requests with complete TTFT/E2E timing and TPOT when
   at least two completion tokens are reported;
6. a valid byte-bounded server-log window covering only the measured phase;
7. no declared contamination marker in that measured log window.

Warmup and measured request records are stored separately. The measured server
log window is preserved as its own artifact. The evidence manifest records the
FENIX commit, locked runtime/model revisions from `configs/runtime_lane.json`,
runtime image names visible in the launch log, workload parameters, log byte
offsets, contamination findings, and SHA-256 hashes of the generated artifacts.

Only a run whose manifest contains:

```json
{
  "evidence_kind": "local_measured",
  "performance_eligible": true
}
```

may enter a quantitative FENIX performance claim. A completed but contaminated
run is retained as `diagnostic_measurement` and the command returns a non-zero
gate code.

Direct use of `scripts.bench_openai` remains valid for trace work and diagnostic
smokes. It does not independently promote performance evidence.

## Contamination markers

The initial fail-closed marker set includes:

- Triton JIT compilation during inference;
- CUDA/PyTorch/allocator OOM;
- Python traceback;
- runtime error;
- the 60-second shared-memory broadcast stall.

The set is intentionally explicit and version-controlled. New known runtime
confounders must be added with a regression test.

## Consequences

The campaign cannot accidentally promote the first cold request after model
startup. Trace-mode servers cannot be used for performance promotion. Evidence
eligibility is reconstructable from stored artifacts instead of depending on
operator memory.

This gate does not by itself prove host-wide isolation from every unrelated CPU
or storage workload. Endpoint-isolation checks remain a separate campaign
precondition and must be recorded when the capacity-tradeoff experiment is
promoted.
