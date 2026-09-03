# Decision 0006: Predeclare layered capacity A/B geometry

## Status

Accepted for the motivation campaign before measured capacity A/B execution.

## Decision

The capacity-tradeoff projection and the measured runtime must use the same
whole-expert cache geometry: each transformer layer owns an independent expert
cache. Managed host capacity is converted to complete `(layer, expert)` slots,
capped at the finite model population (`num_hidden_layers * num_experts`), and
distributed as evenly as possible across layers. Any remainder is assigned to
lower layer IDs deterministically.

The 64, 96 and 112 GiB managed-capacity points are all measured. Trace
projection may characterize them but may not remove a predeclared point from
the measured campaign. Their roles are fixed before A/B performance data:

- 64 GiB: strong memory-pressure treatment;
- 96 GiB: partial memory-pressure treatment;
- 112 GiB: saturation control, not independently motivation-eligible.

Placement order is also predeclared and counterbalanced across budgets. Each
budget/placement server instance executes the same number of measured
repetitions under the existing performance workload contract.

## Rationale

A single global expert LRU can allocate unused capacity from one layer to
another even though expert tensors are layer-specific. It can therefore
underestimate misses and can report impossible capacities above the model's
finite expert population. A layered model removes both artifacts and provides
a direct contract for the bounded host cache to be implemented in the runtime.

Selecting only budgets that appear favorable after trace characterization
would weaken the falsification design. Measuring all three points preserves a
strong treatment, a partial treatment and a saturation control. The control is
particularly important because any performance change after expert capacity
has saturated cannot be attributed to increased expert residency.

## Evidence boundary

Trace-derived capacity projections and expert rankings remain
`can_establish_motivation = false`. Only the later clean, repeated, measured
A/B performance evidence can pass or falsify the motivation gate.
