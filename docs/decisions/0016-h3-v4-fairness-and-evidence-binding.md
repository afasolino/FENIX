# Decision 0016: H3 v4 aligns concurrency and binds paper promotion to evidence

## Status
Accepted before H3 paper-level promotion.

## Decision
H3 v4 removes QD1 from the promotion verdict. QD1 and QD8 remain diagnostic; the
paper-level hostile conventional baseline uses QD32 because the LPDDR oracle is
calibrated as a high-MLP/saturated envelope. This avoids comparing a saturated
all-LPDDR oracle against deliberately serialized storage.

The synthetic LPDDR regimes remain calibration envelopes, but representative H3
residency replays additionally emit a compact logical-access trace. A separate
Ramulator calibration command derives policy/stratum/capacity-specific sampled
flat-address read, fill-write and mixed traces from that logical sequence.
Promotion requires this actual-trace calibration.

Offline static frequency placement is no longer optimized for saved bytes in the
residency stage. Residency emits class frequency histograms; after fio calibration
the decision stage solves the same finite-capacity placement using measured
class service cost.

Storage uncertainty is projected by a common-window hierarchical bootstrap: one
spatial seed is sampled first, then one replicate for PLE, expert and mixed
interaction within that seed. This preserves the experiment hierarchy and the
available pairing.

Editable prerequisite booleans are removed from the paper gate. The prerequisite
manifest contains SHA256-bound evidence artifacts. Capacity evidence is checked
against the exact decision capacity; placement-invariance and >8K evidence must
be explicit passed artifacts from the same H3 contract and implementation commit.
Page-cache qualification is derived directly from the decision matrix. Tool
qualification is also a bound prerequisite: the archived resolved Ramulator
Python lock, built-extension hashes, compiler/CMake identity and exact fio binary
are checked before directional promotion. The resolved lock includes the pinned
Ramulator project's declared runtime dependencies and is replayed without build
isolation.

Finally, perfect-prefetch sensitivity is extended from deterministic PLE-only to
an intentionally optimistic PLE+expert latency-hiding bound. It does not claim a
real predictor; it is a hostile conventional upper envelope.
