# Decision 0012 — Primary H1 traces disable automatic prefix caching

## Status
Accepted for the intrinsic H1/H2 workload-robustness campaign.

## Observation
The first natural-workload robustness trace completed all seven strata with
vLLM V1's runtime-default prefix-cache behavior. H1 then failed closed on
session requests because the PLE/MoE execution stream contained fewer model
tokens than the client prompt-plus-decode count. This is expected when a later
session request reuses a cached prompt prefix: the cached prefix is not
re-executed through PLE or the routed experts.

## Decision
The primary H1 workload measures intrinsic conditional-state demand. Automatic
prefix caching is therefore disabled explicitly with
`--no-enable-prefix-caching`. The client prompt-plus-decode token count must
continue to equal the PLE model-token count, and every routed MoE layer must
continue to equal the PLE count.

The prefix-cache-enabled trace is retained as a served-system control. It is not
promoted as primary intrinsic H1 evidence.

## Rationale
Weakening the token-equivalence gate would mix model conditional-state locality
with serving-layer KV/prefix reuse. Those effects must be measured separately.
