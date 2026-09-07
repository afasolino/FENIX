# Decision 0017 — H3 v5 statistical pairing, collision-free oracle, and derived promotion evidence

## Status

Accepted.

## Context

The v4 audit identified promotion-level risks: class-specific fio windows were
not genuinely paired; population-limited sampling could create pseudo-replicates;
the actual-H3 LPDDR adapter could alias addresses by modulo folding; several
promotion prerequisites were evidence-bound but not fully derived; QD32 device
capability did not establish workload-realizable QD32; resident and storage
footprints were conflated in the offline oracle; and phase promotion could rely
on full-run calibration.

## Decision

H3 v5 uses non-overlapping blocks of complete source requests. PLE, expert and
mixed samples for one window are exact filters of the same block, and promotion
requires at least three genuine common windows. The bootstrap resamples a common
block and then a physical repetition.

Modulo folding is forbidden in actual-H3 Ramulator replay. The bounded sampled
working set is embedded injectively into a representative channel; repeated
logical objects preserve identity and distinct objects cannot overlap. The
counterfactual is a capacity-independent LPDDR service oracle, while the finite
hierarchy remains constrained by the 32-GiB primary platform.

Placement invariance, >8K coverage and deployment capacity are derived from
executable evaluators and SHA-bound evidence. QD32 remains an intentionally
strong gap-falsification point, but conventional-sufficiency promotion requires
a measured workload-derived scheduler artifact demonstrating realizable QD32.

The final promotion matrix requires exactly one full, prefill and decode QD32
decision for every required stratum/policy point and one common campaign
fingerprint across contract, source trace, storage device/binding, fio and
Ramulator identities, and H3 implementation provenance.

The offline static oracle separates resident footprint from lower-tier transfer
footprint and optimizes measured storage plus LPDDR-fill service cost.

## Claim boundary

H3 remains a conditional-memory-service comparison. Its thresholds are
predeclared engineering thresholds, not end-to-end inference claims. A positive
H3 result motivates integrated timing/H4 and does not establish FeRAM
superiority.
