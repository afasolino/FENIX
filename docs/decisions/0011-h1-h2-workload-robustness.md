# Decision 0011 — H1/H2 workload-robustness campaign

## Status

Accepted for the `study/h1-h2-edge-replay-v1` evidence branch after the first
complete router-level H1/H2 campaign.

## Problem

The original trace corpus is a useful deterministic systems-domain control, but
it is not representative enough to support a general workload-locality claim.
Its twenty seeds all concern memory systems, caching, offload, sparse routing,
or measurement, and exact-length prompts are produced by reusing a small bank
of similarly worded continuation sentences. This can increase lexical/n-gram
reuse and therefore make PLE locality look more favorable than a heterogeneous
natural workload.

The follow-on campaign must try to falsify H1/H2 under semantic, language,
session, and context variation without changing the already-qualified runtime
configuration during the primary experiment.

## Decision

Keep `trace_characterization_v1` and `edge_memory_replay_v1` unchanged as the
controlled homogeneous-domain reference. Add a separate
`h1_h2_workload_robustness_v1` path.

The primary suite contains seven exact-C=1 strata:

- natural English chat from WildChat-1M;
- knowledge/factual questions from MMLU-Pro;
- mathematical reasoning from MATH-500;
- native Python code prompts from HumanEval;
- multilingual knowledge questions from six MMMLU locales;
- chronological multi-turn WildChat sessions rendered as accumulated history;
- natural LongBench-v2 contexts fitted into a 6.5–7.0 Ki-token band.

The selected source material is frozen under the ignored `external/workloads/`
tree. Every Hugging Face revision is resolved to an immutable repository SHA
before iteration. The frozen corpus and source manifest are hashed into every
trace case.

Representative native prompts are not padded to synthetic exact token lengths.
If a native independent prompt exceeds the qualified input bound, trace
collection fails closed rather than silently truncating it. Session prompts may
drop the oldest dialogue blocks because preserving recent state is part of the
declared session workload. LongBench-v2 may truncate only the natural context
prefix while preserving the original question and answer choices; no filler is
added.

Natural generation is allowed to terminate before the declared maximum output
length. H1 coverage remains strict: every successful request must expose all 48
MoE layers, each layer must have the same routed-token count as PLE, and PLE
model-token observations must match the autoregressive request accounting.

## H1 analysis

Report per stratum and by prefill/decode phase:

- PLE accesses, unique rows, fraction of the address space, and request-order
  working-set growth;
- routed-expert unique `(layer, expert)` objects, per-layer working set,
  normalized selection entropy, and Top-16/32/64/128/256 concentration;
- pairwise PLE and expert-set Jaccard overlap across strata;
- pairwise Top-128-per-layer expert-set overlap and Jensen-Shannon divergence;
- multilingual subgroup sizes;
- consecutive-turn PLE/expert overlap for real sessions.

## H2 analysis

Use the same 4/7/8/12/16/19-GiB conditional-state cache capacities and the
measured 2,534,400-byte complete expert object. H2 remains a logical
capacity/traffic projection only.

Evaluate three policy roles:

1. `static_frequency_demand_fill`: selected from training requests and evaluated
   on unseen requests with first-touch compulsory misses;
2. `adaptive_request_epoch_lfu`: update frequency-derived placement only after
   each complete request, preserving already resident objects that remain
   selected and demand-filling newly selected objects on a future access;
3. `oracle_frequency_demand_fill`: same-evaluation-trace static upper bound used
   only to quantify the deployable-policy gap.

Report in-domain even/odd request splits, leave-one-domain-out generalization,
structural session/long-context holdouts, and mixed online streams under
blocked, round-robin, and deterministic-random domain orderings.

An exact shared-cache LRU is deliberately not claimed. The trace contains
batched PLE and MoE events and does not establish one unique fine-grained global
ordering between every PLE row and routed expert reference. Request-epoch
adaptation is therefore the strongest ordering-safe online policy in this
campaign.

## Context-length boundary

The primary campaign keeps the qualified `max_model_len=8192`. Contexts at
32K, 64K, and 128K are predeclared follow-on feasibility points but require a
separate runtime-capacity requalification before measurement. They must not be
mixed into the primary H1/H2 evidence by simply changing the server launch.

## Evidence boundary

A6000 traces establish H1 workload behavior. H2 consumes those measured streams
and establishes only conditional-state cache capacity and logical lower-tier
traffic. Neither H1 nor H2 establishes LPDDR/UFS/NVMe latency, transaction
amplification, bandwidth, queueing, or energy. Those remain H3.
