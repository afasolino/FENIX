# Decision 0018 — H3 v6 address-layout sensitivity and scaling evidence

## Status
Accepted for H3 v6.

## Context
The v5 audit left two material uncertainties: collision-free sample-local LPDDR
placement still imposed one artificial physical layout, and the long-context
prerequisite established coverage rather than a multi-length characterization.
QD32 sufficiency evidence also still trusted aggregate scheduler fields.

## Decision
H3 v6 keeps the same conditional-memory-service endpoint and thresholds. Actual-H3
Ramulator calibration must include both dense first-touch and deterministic hashed
bank-spread collision-free layouts at every nCS point. The decision uses the full
calibration envelope; neither mapping is selected post hoc. This is explicitly an
address-layout sensitivity, not native physical-address preservation.

Placement invariance is order-sensitive while ignoring timing-only differences.
Long-context evidence requires distinct trace-valid 8K, 16K, and 32K bands and
reports PLE/MoE conditional-state metrics at each point. The prerequisite establishes
multi-length characterization only; monotonic service or end-to-end scaling requires
separate evidence.

QD32 can still falsify a memory gap as an optimistic conventional envelope. It can
prove conventional sufficiency only when raw workload-derived scheduler events show
that the measured outstanding-I/O opportunity is high enough and storage completions
fit inside object-specific available prefetch slack for every required stratum and
both prefill/decode phases. The scheduler measurement is bound to the campaign trace
manifest and storage device, and FENIX derives all fractions from raw events.

The campaign directional bound itself is phase-aware. Gap support must survive all
full/prefill/decode policy points. Conventional sufficiency requires one fixed online
policy per stratum whose worst upper bound across those three scopes remains <=5%.

## Consequences
The H3 paper gate is more conservative and may block more often. This is deliberate.
No FeRAM/CNM promotion is allowed merely because one favorable LPDDR layout, one
long-context point, or a hand-authored scheduler aggregate passes.
