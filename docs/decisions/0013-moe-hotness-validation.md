# Decision 0013 — Validate MoE hotness at the timescale at which it is used

## Status
Accepted.

## Motivation
Long-run union coverage is not a valid hot-expert metric for Qwen3.8-Flash-Next.
Each token selects 10 of 512 experts per routed layer. Under a uniform-routing
null, moderate token counts already drive expected union coverage close to all
512 experts.

## Primary metrics
Hotness is evaluated per request, phase, and rolling token window. The analysis
reports ranked concentration, normalized entropy, effective expert count, and
observed unique occupancy relative to an analytic uniform-routing null.

One model token's top-k expert set is the atomic event. No artificial ordering is
introduced among the ten experts selected by one token.

Request Expert Activation Matrices (48 x 512) are compared within domains,
across domains, between consecutive session turns, and between prefill/decode.

## Cache validation
The replay compares static LFU, random static placement, causal token-atomic LRU,
causal request-history LFU, and an offline token-atomic Belady oracle. One
complete layer-expert object is the transfer unit. These are transfer
projections, not latency/bandwidth/energy results.

## Setup-invariance control
The same frozen `code` stratum is rerun with 16, 32, and 64 configured hot
experts. Router traces must match exactly after timestamps and request IDs are
removed. This tests that the router-level trace observes routing rather than
expert-residency policy.

The control does not establish invariance to checkpoint precision or to a
different runtime implementation.
