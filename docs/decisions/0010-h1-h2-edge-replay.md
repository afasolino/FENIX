# Decision 0010: H1/H2 use complete exact-C1 routing traces and edge-capacity replay

## Status

Accepted for the `study/h1-h2-edge-replay-v1` mirror study.

## Scientific boundary

The RTX A6000 is the real-Qwen execution oracle. H1 is measured directly from exact concurrency-one Qwen3.8 traces. H2 replays those measured conditional-state demands through small volatile-cache capacities. Neither H1 nor H2 treats the A6000 as an edge SoC.

H3 is explicitly excluded from this study. No LPDDR, UFS, NVMe, latency, bandwidth, queueing, power, or energy parameter is introduced here.

## H1 coverage requirement

Historical FENIX MoE instrumentation emitted selection records from the dynamic GPU-LRU path. Large prefill batches could bypass that path, so decode could be observed while prefill routing was absent.

The H1/H2 trace instrumentation therefore emits one routed-expert selection batch for every WNA16 MoE `apply()` invocation:

- large/fallback paths emit a selection-only record at the common path;
- the existing dynamic-LRU path emits the selection together with cache telemetry.

Exact H1 analysis is fail-closed. For every request:

1. all configured transformer layers must be present;
2. the number of routed token-equivalents at every layer must equal the number of PLE token positions observed for the same request.

A decode-only MoE stream therefore cannot pass H1.

The previous exact expert stack-distance analysis is intentionally deferred when full prefill selection batches are present. Expanding tens of millions of expert references into the old global Fenwick implementation would create an unnecessary multi-GiB analysis transient. H1 instead records a streaming expert-selection reuse-gap histogram together with exact per-layer frequencies and working-set sizes. PLE retains the existing exact row-locality analysis.

## H1 outputs

For each 128-, 1024-, and 4096-input-token exact C=1 case, H1 reports:

- PLE row accesses, unique rows, bytes and locality;
- routed expert selections and unique `(layer, expert)` objects;
- per-layer unique expert counts;
- top-16/32/64/128/256 expert concentration;
- streaming expert reuse-gap distribution;
- logical PLE, expert and combined conditional-state bytes per model token;
- request-level PLE/MoE coverage proof.

Evidence kind remains `local_measured_trace_analysis`.

## H2 capacity scope

The capacities

`4 / 7 / 8 / 12 / 16 / 19 GiB`

are interpreted as **conditional-state cache capacity**. They are not asserted to be the complete DRAM capacity available to all dense weights, activations, runtime state and OS allocations on a phone.

This makes the initial H2 result an optimistic but explicit capacity/traffic experiment. It can falsify the usefulness of small conditional caches, but cannot establish end-to-end edge feasibility by itself.

## H2 policy

The 1024-token exact C=1 case trains a deterministic static hot-set policy. The policy chooses complete layer-expert objects and physical PLE rows to maximize avoided lower-tier bytes under each cache budget.

Objects are demand-filled: the first access is always a compulsory lower-tier miss. Subsequent accesses hit only if the object belongs to the selected hot set.

The optimization is exact for the two fixed object sizes by enumerating the number of expert objects and filling the remaining capacity with the highest-value repeated PLE rows.

The 1024-token result is marked `training_in_sample`. The 128- and 4096-token cases are marked `cross_workload_holdout`; these holdouts are the scientifically useful generalization check.

Evidence kind is `trace_projection`.

## H2 outputs

For every budget and case:

- selected expert-object count and selected PLE-row count;
- cache bytes used;
- PLE and expert hit/miss counts;
- lower-tier bytes by object class;
- combined lower-tier bytes per model token;
- traffic reduction relative to no volatile conditional cache.

These are capacity and traffic results only.

## H3 boundary

H2 output must not be converted directly into latency, throughput or energy. H3 starts only when the same measured access demand is combined with independently grounded LPDDR and UFS/NVMe service models.
