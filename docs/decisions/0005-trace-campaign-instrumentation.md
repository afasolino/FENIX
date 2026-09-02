# ADR 0005: Campaign-owned trace instrumentation

## Status

Accepted.

## Context

The FENIX motivation depends on Qwen3.8-Flash-Next-specific PLE and routed-expert
locality. The runtime already emits physical PLE row batches and MoE routing,
residency, and transfer records when `FENIX_TRACE=1`, but ad-hoc benchmark
commands cannot guarantee exact workload identity, trace isolation, or
request-level attribution.

## Decision

A dedicated trace campaign runner executes only the matrix declared in
`configs/campaign.json`.

The workload profile is versioned as `trace_characterization_v1` and its
multi-domain natural-language material is stored in
`configs/trace_prompt_corpus_v1.json`. Every prompt is deterministic from the
campaign seed, input length, request ordinal, and versioned corpus, and is
independently validated against the live server tokenizer. Raw trace streams
remain append-only; each case records byte offsets and publishes only its own
complete JSONL window.

Concurrency one uses exact client-interval correlation for both PLE and MoE.
Higher concurrency is retained as aggregate service evidence until direct
scheduler request IDs are exported by the runtime.

A completed trace case is immutable and published atomically. The runner refuses
to overwrite an existing case. Cross-case verification requires a common FENIX
commit, campaign hash, runtime/model lane, runtime image tag and image ID, and a
common prompt set for cases sharing an input length.

## Consequences

Trace artifacts can support locality, working-set, residency, and transfer
claims. They cannot support throughput or latency claims and cannot by
themselves establish the FENIX motivation gate.

Capacity screening derives expert bytes from explicit transfer records and PLE
bytes from measured row width plus versioned model geometry whenever those
measurements are available. Manual byte-size overrides remain diagnostic and
are machine-labeled.
