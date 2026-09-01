# Evidence policy

FENIX uses explicit evidence classes so that literature reuse does not become
endpoint-incompatible numerical comparison.

## Evidence classes

### `peer_reviewed_measured`

Measured evidence published in a peer-reviewed venue.

Use for mechanism-level claims and, when the endpoint is compatible, numerical
comparison.

### `preprint_measured`

Measured evidence reported in a preprint or technical report.

Use for scoped mechanism-level claims with publication status stated
explicitly. Do not silently promote the result to peer-reviewed evidence.

### `upstream_implementation`

Behavior established by a mature upstream implementation, release, pull
request, or source inspection.

Use to justify the existence and semantics of an implementation path. Do not
treat implementation existence as an end-to-end performance result.

### `local_measured`

FENIX measurements collected on the target endpoint under versioned
configuration.

This is the only evidence class that may establish the FENIX motivation gate.

### `trace_projection`

A deterministic or calibrated model driven by a FENIX trace.

Use for cache sizing, working-set analysis, and screening informative
measurement points. It cannot establish the motivation gate.

## Claim rules

1. Every quantitative FENIX performance claim must identify model revision,
   runtime revision, hardware, workload, and repetition policy.
2. Cross-paper throughput and latency numbers are never combined into a FENIX
   speedup.
3. A literature result may remove the need to reproduce a mechanism only when
   the FENIX claim does not depend on endpoint-specific magnitude.
4. PLE locality and expert locality are model/workload dependent and therefore
   remain local measurements.
5. The PLE-versus-expert host-memory tradeoff is the core local causal
   experiment and cannot be replaced by a literature result unless an
   endpoint-equivalent study becomes available.
6. Trace instrumentation runs are not used for throughput claims.
7. Projected results are labeled `trace_projection` in machine-readable output.

## Current evidence allocation

| Question | Evidence source | Local reproduction |
| --- | --- | --- |
| Can conditional memory be prefetched from host memory? | Engram | No |
| Can conditional memory use a lower memory tier? | CXL-Engram, TF-Engram | No, unless required by a later architecture claim |
| Does heterogeneous placement matter under limited GPU memory? | PowerInfer | No |
| Can MoE inference exploit CPU/GPU orchestration? | Fiddler, KTransformers | No |
| What PLE rows does Qwen3.8-Flash-Next access? | FENIX trace | Yes |
| What experts does Qwen3.8-Flash-Next access? | FENIX trace | Yes |
| Does PLE host-DRAM occupancy materially reduce useful expert residency? | FENIX capacity tradeoff | Yes |
