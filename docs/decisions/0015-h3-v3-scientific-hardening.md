# Decision 0015 — H3 v3 scientific hardening

## Status

Accepted as the execution contract for H3. This decision narrows and strengthens Decision 0014; it does not change the H3 scientific question or the authoritative H1/H2 source trace.

## Motivation

A review of H3 v2 found that the trace/capacity layer was rigorous, while the conventional timing envelope could still favor a conclusion through modeling choices that were not independently bounded. In particular, saturated streaming bandwidth alone did not characterize PLE-like random 160-byte demand or multi-megabyte expert bursts; the page-cache upper bound reused the fastest storage class; QD32 was configured but not required to be achieved; fio sampling mixed spatial and run-to-run variability; and directional project conclusions did not share identical promotion prerequisites.

H3 is a falsification stage. Every correction below therefore strengthens the conventional baseline or widens uncertainty rather than making the memory-gap hypothesis easier to support.

## LPDDR access-pattern envelope

The pinned, unmodified Ramulator2 LPDDR5 model remains authoritative. One two-rank `LPDDR5_8Gb_x16 / LPDDR5_6400` channel is calibrated at `nCS={1,2,4}` under three access regimes:

1. saturated streaming;
2. PLE-like random 160-byte objects, emitted as five contiguous 32-byte transactions per object;
3. expert-like 2,534,400-byte contiguous bursts at 4-KiB-aligned random bases.

Each regime is measured as read-only, write-only, and deterministic 50/50 read/write traffic. The flat-address regimes use Ramulator's `LoadStoreTrace` and public `RoBaRaCoCh` mapper. FENIX wraps the upstream interface and does not modify Ramulator. The measured one-channel rates are scaled over 16 explicitly independent logical x16 channels; the 204.8-GB/s theoretical peak is retained as a deliberately optimistic sensitivity.

The all-LPDDR and finite-hierarchy service calculations retain every access-regime/nCS point. A conclusion must survive the full envelope.

## Shared LPDDR interface

Logical reads and lower-tier fills share the LPDDR data interface. The lower service bound therefore includes

```text
(read useful bytes + fill bytes) / measured same-regime mixed R/W bandwidth
```

in addition to the independent read, storage, and fill terms. This is still an envelope rather than an integrated queueing model; threshold-crossing intervals remain escalated to integrated timing.

## Strong conventional residency policies

The primary object/VM-granularity LRU policies remain. H3 adds size-aware online LFU variants at both granularities. LFU learns only after an epoch has been classified, preserving equal-timestamp atomicity.

Every replay also reports:

- an offline static-frequency demand-fill bound at storage granularity with compulsory first touch;
- a deterministic-PLE perfect-prefetch sensitivity that hides PLE lower-tier latency while retaining its storage bytes and LPDDR fill traffic.

The campaign gate evaluates the best predeclared conventional policy for each workload. The offline and prefetch cases are explicitly optimistic sensitivities, not measured production policies.

## Storage measurement and uncertainty

The five deterministic contiguous windows per class are retained. Every window/QD point is repeated three times by default. Run order is deterministically randomized. fio's `iodepth_level` histogram is recorded and QD32 is rejected unless at least 75% of the measured time is at or above half of the configured queue depth.

Uncertainty is hierarchical: spatial windows are resampled, then physical repetitions inside the selected windows are resampled. The final storage service projection is resampled directly from the empirical PLE, expert, and mixed-interaction distributions; marginal confidence-interval endpoints are not multiplied as though they formed a joint interval.

## Linux page-cache control

The complete logical stratum is still replayed. Before execution, `POSIX_FADV_DONTNEED` is followed by deterministic `mincore` sampling of trace-derived pages. A run is rejected when the initially resident sampled fraction exceeds the contract threshold. This is a qualification of the sampled working set, not proof that every page in the file is absent.

For decision normalization, observed backing-device bytes use the fastest class low coefficient for the lower bound and the slowest class high coefficient for the upper bound. The measured fio wall time is retained as a separate sensitivity.

## Provenance and build reproducibility

Every decision input must contain the same H3 contract SHA and the same clean FENIX execution `HEAD`, tree, and authoritative H1/H2 ancestor. Cross-commit artifact mixing fails closed.

Tool qualification records the exact Ramulator/fio Git commits, fio binary SHA256, resolved Ramulator Python environment, Python/pip versions, CMake/compiler versions, and built Ramulator extension SHA256 values.

## Phase reporting

Full-stratum replay remains the primary endpoint. Full residency artifacts also contain prefill and decode counters, and the H3 decision evaluates each phase with the same qualified LPDDR/storage calibration. Page-cache and offline-frequency controls remain full-stratum constructs and are not silently projected into one phase.

## Symmetric campaign gate

Placement invariance, >8K context scaling, deployment-memory accounting, and page-cache qualification are prerequisites for both directional project conclusions. Therefore neither `H3_MEMORY_GAP_SUPPORTED` nor `CONVENTIONAL_MEMORY_SUFFICIENT` becomes a paper/project-stopping conclusion before those prerequisites pass.

## Claim boundary

H3 remains a conditional-memory-service study. It does not establish end-to-end inference speedup, FeRAM superiority, LPDDR PHY energy, or absolute storage energy.
