# Measurement protocol

## Scope

The protocol follows ADR 0001. FENIX does not reproduce established
conditional-memory or heterogeneous-MoE systems solely for another endpoint.
Local measurements target the Qwen3.8-Flash-Next-specific resource coupling.

## Run classes

Performance runs use `FENIX_TRACE=0`.

Trace runs use `FENIX_TRACE=1` and are never used for throughput claims because
instrumentation can perturb execution.

## Runtime qualification

Runtime qualification establishes one mature, reproducible model-serving path
on the target system. It is intentionally small: the purpose is to validate the
endpoint and obtain a reference point, not to benchmark the serving ecosystem.

## Trace characterization

PLE traces must contain enough information to reconstruct the physical row
stream and independently verify it against `analysis/ple_address.py`.

Expert traces must identify at least the layer and selected expert IDs. Runtime
cache residency and transfer events are recorded when available, but locality
analysis must remain possible from the selection stream alone.

Exact request-level PLE correlation is first established at concurrency one.
Higher-concurrency runs characterize aggregate service overlap until serving
scheduler request IDs are exported directly.

## Capacity tradeoff

The two placements are:

- `ple_in_host_dram`: PLE capacity is charged against the host-memory budget;
- `ple_externalized`: the same capacity is available for expert residency.

The comparison requires identical model revision, runtime revision, hardware,
workload, and quality settings.

Trace projection may screen host-memory budgets. Projected results cannot
establish the motivation.

At least the configured number of independent measured repetitions is required
for a promoted comparison point.

## Metrics

Performance metrics:

- TTFT;
- TPOT;
- decode tokens/s;
- prefill throughput when the runtime exposes an unambiguous measurement;
- request-level p50/p95/p99.

Resource metrics:

- GPU memory occupancy;
- host-memory occupancy;
- PLE service time and exposed stall when observable;
- CPU-GPU transfer bytes from explicit transfer records;
- cold-expert storage traffic from explicit runtime or block-device accounting.

GPU utilization is never used as a byte-traffic estimator.

## Motivation verdict

`analysis/evaluate_motivation.py` is fail-closed:

- missing measured data -> `INCONCLUSIVE`;
- incompatible endpoint metadata -> `INCONCLUSIVE`;
- insufficient repetitions -> `INCONCLUSIVE`;
- at least one informative budget passing all predeclared gates -> `SUPPORTED`;
- sufficiently measured budgets with no passing point -> `FALSIFIED`.
