# Decision 0029 — H3 placement semantic-control and prerequisite provenance amendment

## Status

Accepted for the causal H3 continuation path.

## Context

The frozen e08 H3 campaign did not materialize the placement-invariance prerequisite. During finalization, the v6 evaluator was also found to include the raw request UUID in the placement semantic digest, while the trace harness intentionally generates a fresh UUID for every independently executed request. The prompt digest additionally included endpoint/tokenize metadata. These are run-local correlation/provenance fields, not conditional-memory semantics, and would make independently executed resident/mmap controls fail even when the prompt, PLE-row, and routing sequences are identical.

The numerical H3 measurements, causal decision matrix, thresholds, and hostile conventional bounds are unaffected by this issue.

## Decision

Placement invariance is re-derived as an analysis-only semantic control with these rules:

- the two required physical PLE modes are exactly `resident` and `mmap`;
- each mode is bound to its captured server launch preamble through `FENIX_PLE_STORAGE_MODE`; mmap must additionally bind `FENIX_PLE_BANK_MANIFEST`;
- each case must be clean, trace-valid, concurrency-one evidence for `chat_en`, `session`, and `long_context_8k`;
- both placements must share the same source repository commit, runtime image, runtime lane, frozen corpus, and source manifest;
- prompt semantics are the ordered prompt bytes/token counts; endpoint/tokenize URLs are excluded;
- run-local request UUIDs are canonicalized to first-seen request ordinals;
- PLE comparison preserves request order, token position, PLE head, physical row, byte width, and phase;
- MoE comparison preserves request order, layer, phase, trace scope, and each atomic top-k selection, while runtime event batching and timestamps are excluded.

The frozen causal gate previously required every prerequisite artifact to have been emitted at e08. The causal continuation instead allows only the semantic prerequisites `placement_invariance` and `long_context_beyond_8k` to be emitted by the current analysis HEAD. `deployment_capacity_budget` and `tool_qualification` remain frozen-only at e08. The base causal gate is still run first; the provenance amendment may apply only when the base prerequisite failure is exactly the implementation-identity mismatch, meaning all prerequisite kind, SHA, contract, content, capacity, tool, and campaign-fingerprint checks already passed.

## Consequences

This amendment does not change any H3 numerical bound or directional threshold. It prevents run-local UUIDs, endpoint metadata, and event batching from masquerading as placement-dependent model semantics while preserving a fail-closed physical resident-versus-mmap control. A passing placement control remains a semantic-invariance prerequisite only; it is not a performance result and does not establish end-to-end speedup, energy superiority, or FeRAM superiority.
