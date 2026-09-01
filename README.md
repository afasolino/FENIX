# FENIX

FENIX is an evidence-first investigation of a host-centric inference architecture
for Qwen3.8-Flash-Next.

The scientific question is deliberately narrow:

> Does moving the PLE table out of conventional host DRAM materially improve
> host-centric MoE inference because the freed DRAM can hold more routed
> experts, reducing cold-expert storage traffic and exposed stalls?

FENIX does not reproduce established systems results merely to obtain another
measurement on a different machine. It reuses literature and mature upstream
implementation evidence where the underlying claim is already established, and
measures only the Qwen3.8-Flash-Next-specific coupling that the literature does
not answer.

The rationale and claim-provenance rules are recorded in:

- `docs/decisions/0001-literature-grounded-local-falsification.md`
- `docs/evidence_policy.md`

## Evidence architecture

FENIX separates evidence into five classes:

1. peer-reviewed measured evidence;
2. measured preprints and technical reports;
3. mature upstream implementation evidence;
4. FENIX local measurements;
5. FENIX trace-derived projections.

Only compatible local measurements may establish the final FENIX motivation
claim. Trace projections may select informative measurement points, but cannot
promote the hardware architecture by themselves.

Endpoint-incompatible throughput or latency values from prior work are never
placed on a FENIX performance curve.

## Local experiment scope

The local campaign contains four experiments:

1. **Runtime qualification** — establish one working Qwen3.8-Flash-Next path on
   the target system and obtain a reference latency/throughput point.
2. **PLE locality characterization** — measure the exact PLE row stream needed
   for cache sizing and external-memory design.
3. **Expert locality characterization** — measure the exact routed-expert stream
   needed to quantify host-memory residency pressure.
4. **Capacity tradeoff experiment** — compare the same endpoint with PLE
   capacity charged to host DRAM versus the counterfactual in which that
   capacity is available to experts.

The capacity tradeoff is the motivation gate. If it is not materially positive,
FENIX does not proceed to a specialized SRAM/ferroelectric-memory architecture.

## Repository structure

- `configs/` — versioned campaign configuration.
- `docs/` — methodology, evidence policy, and architecture decisions.
- `instrumentation/` — runtime instrumentation only.
- `analysis/` — deterministic post-processing and statistical evaluation.
- `literature/` — primary-source ledger and comparison metadata.
- `results/` — small processed outputs; large raw data is ignored.
- `traces/` — trace schemas and processed trace outputs.
- `profiles/` — profiler outputs; large profiles are ignored.
- `external/` — ignored third-party runtimes and model assets.

## Primary runtime candidate

The currently pinned candidate lane is:

- model: `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`
- model revision: `ef554143369a706525336f6b42a09094835dc077`
- runtime: `DominikBucko/qwen38-flash-next-2x3090`
- runtime revision: `7b5f0465db90fc49d6324904f48ad995ebdcb62f`
- base image:
  `vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`

This is a candidate execution lane, not a scientific assumption. It must be
qualified on the target system before performance claims are made.

## Campaign order

1. Capture preflight evidence.
2. Freeze the literature/source ledger and experiment configuration.
3. Fetch and source-qualify the pinned runtime lane without model weights.
4. Acquire the pinned model only after the runtime-lane gate passes.
5. Qualify one mature Qwen3.8-Flash-Next runtime path on the target host.
6. Collect exact PLE and expert traces.
7. Verify PLE addresses independently of the serving runtime.
8. Run locality analysis.
9. Use the trace-derived capacity model only to screen informative host-memory
   budgets.
10. Measure the capacity tradeoff on the same model/runtime/hardware/workload.
11. Run `analysis/evaluate_motivation.py`.
12. Proceed to hardware architecture only if the predeclared gate passes.

At repository initialization, no same-endpoint capacity-tradeoff measurements
exist. The verdict is therefore `INCONCLUSIVE`.

## Development environment

FENIX requires Python 3.11 or newer.

`requirements-analysis.txt` contains the supported dependency ranges.
`requirements-analysis.lock.txt` records the exact analysis environment used
for the initial 2026-09-01 campaign setup. Use the lock file when reproducing
that environment exactly.

Before committing a campaign change, run:

```bash
/home/giando/work/FENIX/.venv/bin/python scripts/check_repository.py
```

Analysis modules that import other FENIX analysis code should be invoked from
the repository root with `python -m analysis.<module>`.

## Runtime-lane gate

Runtime source and model acquisition are intentionally separate.

Both direct-script and module invocation are supported. The documented form is:

```bash
python -m scripts.fetch_runtime
python -m scripts.qualify_runtime
```

The equivalent direct-script forms are regression-tested as well:

```bash
python scripts/fetch_runtime.py
python scripts/qualify_runtime.py
```

The source-only gate writes an ignored report to
`results/raw/runtime_qualification/report.json`. A large model download is
blocked unless that report reaches `READY_FOR_MODEL_FETCH`.

The TP=1 launcher forces `--distributed-executor-backend mp` for the pinned
preview image because that image predates the upstream PLE uniprocess-executor
initialization fix. This is a documented compatibility condition, not a local
vLLM modification. See ADR 0002.
