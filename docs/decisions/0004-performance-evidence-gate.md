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
3. an unambiguous `FENIX_TRACE=0` launch and exactly one identifiable FENIX
   runtime image tag;
4. a supported workload profile declared in `configs/campaign.json`;
5. a prompt whose rendered chat-template token count is verified through the
   running server's `/tokenize` endpoint and exactly equals the predeclared
   input-token count;
6. an explicit 1-based repetition index within the campaign repetition policy;
7. successful warmup requests whose observed prompt and completion token counts
   exactly match the campaign contract;
8. successful measured requests whose observed prompt/completion counts and
   TTFT/E2E/TPOT timing exactly match the campaign endpoint;
9. a valid byte-bounded server-log window covering only the measured phase;
10. no declared contamination marker in that measured log window; and
11. a fresh output artifact set so previous evidence cannot be overwritten.

Warmup and measured request records are stored separately. The exact generated
prompt and the measured server-log window are preserved as artifacts. The
evidence manifest records the FENIX commit, locked runtime/model revisions from
`configs/runtime_lane.json`, runtime image name, workload profile, preflight and
observed token counts, repetition index, log byte offsets, contamination
findings, and SHA-256 hashes of the generated artifacts.

`python -m scripts.workload_contract` may be used independently to materialize
and inspect the deterministic token-exact prompt. Performance promotion does
not trust a locally estimated token count: it reuses the same contract code and
asks the live runtime to tokenize the rendered chat request.

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
