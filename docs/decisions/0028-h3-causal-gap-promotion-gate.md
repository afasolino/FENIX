# ADR 0028: Causal H3 gap-promotion gate

## Context

The frozen H3 physical campaign remains at
`e08bc8d32b85ef3fe929d350ca122159fcee4073`. The host's legacy cgroup-v1
configuration prevents the frozen v6 Linux page-cache collector from enforcing
its preregistered per-experiment 7-GiB memory/blkio envelope without privileged
system changes.

A stronger rootless future-aware cache oracle was therefore evaluated for gap
falsification. When it was additionally granted arbitrary perfect expert
prefetch, the lower H3 bound reached zero for one sensitivity, so arbitrary
expert-prefetch robustness was not supportable.

The separately preregistered causal runtime experiment then measured the exact
host-visible router-ID to native `quant_method.apply` interval in the pinned Qwen
runtime. Across 6144 router events and all 48 layers in both prefill and decode,
the maximum observed native interval was 6514 ns. The fastest lower-95%-CI QD32
one-expert transfer from the complete frozen fio campaign was
2335891.381345927 ns. Thus the maximum exact-ID lookahead was only
0.002788657063431893 of the fastest measured one-expert transfer time. The
causal gate passed with no failures.

A source audit of the same pinned runtime revision resolves the ordering inside
the native dispatch. With `VLLM_WNA16_DYNAMIC_LRU=1`,
`CompressedTensorsWNA16MoEMethod.apply()` updates the LRU miss map, gathers each
missing expert tensor row into the hot cache, and only then invokes
`cache.kernel.apply(...)`. The native path therefore does not split resident and
missing experts into independent compute dispatches. The causal conclusion is
accordingly precise: there is negligible exact-ID lookahead before dispatch, and
required cold-expert miss staging remains on the blocking native path inside
that dispatch before expert computation. A redesigned split/speculative
scheduler is a different architecture and remains outside this claim.

A subsequent `session/useful_object_lru/full` canary retained perfect PLE
prefetch and the future-aware cache upper bound but serialized unavoidable
expert storage on that blocking miss-staging path. Its global gap lower bound
was 0.4045828073257465, above the preregistered 0.10 H3 threshold.

## Decision

Use a separate causal promotion gate; do not modify the original v6 campaign
gate or the frozen measurement artifacts.

The causal gate:

- regenerates the exact 84-row promotion matrix: seven required strata, four
  policies, and full/prefill/decode scopes at 7 GiB and QD32;
- reuses the frozen e08 residency, fio, generic LPDDR, and actual-H3 Ramulator
  artifacts by exact SHA provenance;
- requires the passed causal expert-prefetch artifact on every row;
- binds the causal interpretation to the pinned dynamic-LRU source ordering;
- retains perfect PLE prefetch;
- requires the future-aware causal Belady cache control for full `session` and
  `long_context_8k` rows;
- permits that cache control to discharge the Linux page-cache requirement only
  for a memory-gap conclusion, because it is deliberately stronger than a
  realizable replacement policy under the same pinned-runtime expert-visibility
  and blocking miss-staging constraints;
- validates placement invariance, long-context characterization, deployment
  capacity, and tool qualification against the frozen e08 measurement identity;
- permits a different committed analysis HEAD while preserving the frozen
  measurement identity explicitly in every decision.

The gate may emit `H3_PAPER_GATE_SUPPORTED` only if every required full and
independently phase-filtered row retains a lower penalty of at least the
predeclared 0.10 threshold and all frozen prerequisites pass.

## Non-permitted conclusion

This causal gate cannot promote `CONVENTIONAL_MEMORY_SUFFICIENT`. Such a claim
still requires a realizable measured conventional page-cache/scheduler path and,
for QD32 sufficiency, the separately required workload-derived scheduler
feasibility evidence.

The supported H3 scope, if reached, remains the conditional-memory service
layer. It is not an end-to-end inference-speedup claim, an energy-superiority
claim, or evidence of FeRAM superiority. Prediction, speculative expert
prefetch, GPUDirect Storage, split resident/miss execution, and redesigned
schedulers remain outside the measured pinned-runtime claim.
