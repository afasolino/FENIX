# Decision 0014 — H3 conventional hierarchy v2

## Status

Accepted for execution. This supersedes earlier provisional H3 implementations.

## Question

H3 tests whether a strong conventional hierarchy adds a material **conditional-memory service** penalty relative to the same endpoint with all conditional state already resident in LPDDR. H3 does not infer end-to-end token latency from component-service time.

## Trace semantics

The authoritative source is the no-prefix C=1 H1 robustness trace. Every stratum starts with a cold finite-LPDDR cache. Collection order never becomes cache state. Source timestamps are used only to preserve measured event ordering; they are never interpreted as edge-platform timing.

PLE and MoE traces are revalidated against client autoregressive token counts before H3 promotion. Replay is streaming and does not expand the millions of expert-token observations into another unbounded JSON artifact. Events sharing the same source timestamp are classified against pre-epoch cache state, preventing arbitrary tie order from creating hits.

## LPDDR baseline

The LPDDR baseline uses the pinned Ramulator2 LPDDR5 latency-throughput infrastructure rather than a hand-written analytical controller. One x16 LPDDR5-6400 channel is characterized with the upstream streaming frontend and then scaled over 16 independent logical channels to match a 204.8-GB/s edge-class public envelope. Four deliberately conventional-favourable sensitivity points are retained: pinned Ramulator with two ranks per x16 channel and `nCS=1`, `nCS=2`, and `nCS=4`, plus the 204.8-GB/s theoretical platform peak. Refresh remains enabled in all Ramulator points. The `nCS` sweep is required because the pinned upstream profile labels its default rank-switch penalty as a guesstimate; it is therefore not collapsed into one physically certain value. The saturation stream follows the predeclared H3 v2 implementation contract (`stream_cls=8`, no warmup) and is separately checked to offer 25.6 GB/s, above the modeled 12.8-GB/s x16 peak.

## Storage baseline

Pinned fio is the primary lower-tier tool. It replays real direct-I/O requests against the actual filesystem/block device resolved by `findmnt` and `lsblk`. Five deterministic contiguous PLE windows, five expert windows and five mixed windows are measured at QD1 and QD32. Class-level service coefficients carry a bootstrap interval; mixed windows quantify device interaction relative to independent class projections.

MQSim is not a primary H3 tool. If used later, it is only generic NVMe/NAND sensitivity and cannot be labelled UFS or a named-device reproduction.

## Linux page-cache control

A conventional buffered/mmap control is mandatory for session and long-context strata. It runs inside cgroup-v2 `MemoryMax` pressure, with swap disabled. The control replays the complete logical stratum; unique conditional-state footprint is used only to qualify that the full stream exceeds the memory envelope by at least 1.25×. The run must also demonstrate substantial actual `memory.current` pressure. The memory cap is a process-plus-page-cache envelope, not an exact page-cache capacity.

## Same-endpoint bound

For identical logical conditional-state reads, the all-LPDDR oracle pays only LPDDR read service. The finite hierarchy adds lower-tier service and LPDDR fill writes. H3 reports a perfect-overlap lower bound and a full-serialization upper bound. A point is:

- conventional-memory sufficient when the entire penalty interval is <=5%;
- H3 memory-gap supported when the entire interval is >=10%;
- otherwise inconclusive and escalated to integrated timing.

These are predeclared engineering-relevance thresholds, not literature constants.

## Promotion boundary

Even a supported memory-service gap is preliminary until placement invariance, >8K context scaling, a defensible deployment memory budget and the Linux page-cache control all pass. H3 cannot establish FeRAM energy superiority; LPDDR PHY and storage absolute energy remain outside the primary evidence chain.

## Kernel-cache participation in the decision

The buffered/mmap control is part of the hostile conventional baseline rather than a procedural checkbox. Its cgroup-v2 `io.stat` lower-tier read bytes are measured after a best-effort cold-cache eviction with `POSIX_FADV_DONTNEED`. For the same logical request stream, those bytes are normalized with the fastest measured direct-storage service coefficient and compared with the explicit trace-LRU hierarchy. The decision takes the best qualified conventional candidate. Required `session` and `long_context_8k` decisions must include both buffered and mmap controls.
