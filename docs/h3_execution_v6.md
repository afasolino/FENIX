# H3 conventional hierarchy v6 — execution protocol

## Authority and claim boundary

```text
repository: afasolino/FENIX
base: 8b8694647840e5af73fd4e1fb1275d0b15c19056
authoritative trace: results/raw/h1_h2_workload_robustness_noprefix/20260904T162349Z
contract: configs/h3/contract_v6.json
scope: conditional-memory service only
```

The 5%/10% thresholds are predeclared engineering-relevance thresholds, not
literature constants. H3 does not convert service penalties into end-to-end
inference slowdown and does not establish FeRAM/CNM superiority.

All command examples are safe for an interactive shell: no shell-wide fail-fast
settings and no explicit shell-termination commands are used. Inspect each
command's output before moving to the next stage.

## 1. Build and qualify exact mature tools

```bash
.venv/bin/python -m scripts.h3_campaign bootstrap-tools --jobs "$(nproc)"

H3_ROOT="results/raw/h3_conventional_hierarchy_v6/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$H3_ROOT"

.venv/bin/python -m scripts.h3_campaign qualify-tools \
  --out "$H3_ROOT/tool-qualification.json"
```

The exact Ramulator2/fio revisions are pinned in
`configs/h3/upstream_tools.lock.json`. Tool qualification records Git HEAD/tree,
fio binary SHA256, Ramulator extension SHA256s, Python/pip, compiler and CMake,
and archives the resolved Ramulator Python lock.

## 2. Promote the exact no-prefix H1/H2 trace

```bash
TRACE_ROOT="results/raw/h1_h2_workload_robustness_noprefix/20260904T162349Z"

.venv/bin/python -m scripts.h3_campaign manifest \
  --trace-root "$TRACE_ROOT" \
  --execution-verification configs/h1_h2_trace_execution_v1.json \
  --out "$H3_ROOT/trace-manifest.json"
```

## 3. Generic high-MLP LPDDR envelope

```bash
.runtime/h3-tools/ramulator-venv/bin/python \
  -m scripts.h3_campaign ramulator \
  --out "$H3_ROOT/lpddr-generic.json"
```

This is a throughput/high-MLP envelope, not a dependency-limited latency model.

## 4. Structural replay

```bash
POLICIES="useful_object_lru vm_granularity_lru useful_object_lfu vm_granularity_lfu"
STRATA="chat_en knowledge math code multilingual session long_context_8k"

for S in $STRATA; do
  for B in 4 7 8 12 16 19; do
    for P in $POLICIES; do
      OUT="$H3_ROOT/replay/$S/${B}GiB/$P"
      if .venv/bin/python -m scripts.h3_campaign replay \
        --manifest "$H3_ROOT/trace-manifest.json" \
        --stratum "$S" --capacity-gib "$B" --policy "$P" \
        --out-dir "$OUT"; then
        echo "OK replay $S $B GiB $P"
      else
        echo "FAILED replay $S $B GiB $P; inspect the error before continuing."
        break
      fi
    done
  done
done
```

Miss records include request IDs. Equal-timestamp accesses remain atomic.

## 5. Derive the deployment capacity

```bash
cp configs/h3/capacity_budget_template_v6.json \
  "$H3_ROOT/capacity-budget-input.json"
```

Populate every reservation with a non-negative measured GiB value and a SHA-bound
evidence artifact. The primary physical capacity is fixed by the contract at
32 GiB.

```bash
.venv/bin/python -m scripts.h3_campaign capacity-budget \
  --input "$H3_ROOT/capacity-budget-input.json" \
  --out "$H3_ROOT/capacity-budget.json"
```

Set `PRIMARY_BUDGET` only to a value listed in
`candidate_budgets_that_fit_gib`.

```bash
PRIMARY_BUDGET=<4|7|8|12|16|19>
```

## 6. Validate physical backing and storage binding

```bash
PLE="external/storage/qwen38-ple/ple.bin"
PLE_MANIFEST="external/storage/qwen38-ple/ple.manifest.json"
EXPERT="external/storage/h3/expert-surrogate.bin"

mkdir -p external/storage/h3

.venv/bin/python -m scripts.h3_campaign prepare-backing \
  --ple "$PLE" --ple-manifest "$PLE_MANIFEST" \
  --expert "$EXPERT" --create-expert \
  --out "$H3_ROOT/backing-validation.json"

.venv/bin/python -m scripts.h3_campaign probe-storage \
  --ple "$PLE" --expert "$EXPERT" \
  --out "$H3_ROOT/storage-binding.json"
```

The PLE backing is verified against the checkpoint-exact 51,200,245,760-byte
SHA256 contract. The primary fio result requires PLE and expert backing on the
same resolved block device.

## 7. Full-run fio calibration from paired request blocks

For every stratum/policy, v6 selects non-overlapping blocks of complete source
requests. PLE, expert and mixed iologs for one window are exact filters of that
same request block. Duplicate population-limited windows are rejected. At least
three genuine common windows are required for promotion.

```bash
for P in $POLICIES; do
  for S in $STRATA; do
    RDIR="$H3_ROOT/replay/$S/${PRIMARY_BUDGET}GiB/$P"
    SDIR="$H3_ROOT/storage/full/$S/${PRIMARY_BUDGET}GiB/$P"
    mkdir -p "$SDIR"

    .venv/bin/python -m scripts.h3_campaign samples \
      --misses "$RDIR/misses.jsonl.gz" \
      --ple "$PLE" --expert "$EXPERT" \
      --storage-binding "$H3_ROOT/storage-binding.json" \
      --out-dir "$SDIR/samples"

    .venv/bin/python -m scripts.h3_campaign fio-matrix \
      --samples "$SDIR/samples/samples.json" \
      --out-dir "$SDIR/fio-runs"

    .venv/bin/python -m scripts.h3_campaign fio-summary \
      --samples "$SDIR/samples/samples.json" \
      --runs-dir "$SDIR/fio-runs" \
      --out "$SDIR/fio-summary.json"
  done
done
```

QD1/QD8 are diagnostic. QD32 is an intentionally strong conventional
gap-falsification point. Each fio summary reports `paired_window_design.statistical_strength`: three to four independent blocks are `minimum_only`, five to seven are `preferred`, and eight or more are `target`. Bootstrap draws never increase the independent-block count.

## 8. Full-run actual-H3 Ramulator calibration

```bash
for P in $POLICIES; do
  for S in $STRATA; do
    RDIR="$H3_ROOT/replay/$S/${PRIMARY_BUDGET}GiB/$P"

    .runtime/h3-tools/ramulator-venv/bin/python \
      -m scripts.h3_campaign ramulator-actual \
      --residency "$RDIR/summary.json" \
      --out "$RDIR/lpddr-actual.json"
  done
done
```

The sampled logical working set is mapped injectively into the representative
channel. Modulo address folding is forbidden. V6 runs both `dense_first_touch`
and `hashed_bank_spread` layouts at every nCS point; every mapping enters the
decision envelope and none is selected post hoc. Calibration fails if a
collision-free embedding cannot be constructed. This is a capacity-independent
LPDDR service counterfactual and an address-layout sensitivity, not a claim that
the full conditional state fits inside 32 GiB or that native physical addresses
are preserved.

## 9. Independent prefill and decode calibration

Paper promotion requires separate phase-filtered residency, fio, and actual-H3
Ramulator artifacts for every stratum/policy.

```bash
for PHASE in prefill decode; do
  for P in $POLICIES; do
    for S in $STRATA; do
      RDIR="$H3_ROOT/replay/$S/${PRIMARY_BUDGET}GiB/$P/$PHASE"

      .venv/bin/python -m scripts.h3_campaign replay \
        --manifest "$H3_ROOT/trace-manifest.json" \
        --stratum "$S" --capacity-gib "$PRIMARY_BUDGET" \
        --policy "$P" --phase "$PHASE" \
        --out-dir "$RDIR"

      SDIR="$H3_ROOT/storage/$PHASE/$S/${PRIMARY_BUDGET}GiB/$P"
      mkdir -p "$SDIR"

      .venv/bin/python -m scripts.h3_campaign samples \
        --misses "$RDIR/misses.jsonl.gz" \
        --ple "$PLE" --expert "$EXPERT" \
        --storage-binding "$H3_ROOT/storage-binding.json" \
        --out-dir "$SDIR/samples"

      .venv/bin/python -m scripts.h3_campaign fio-matrix \
        --samples "$SDIR/samples/samples.json" \
        --out-dir "$SDIR/fio-runs"

      .venv/bin/python -m scripts.h3_campaign fio-summary \
        --samples "$SDIR/samples/samples.json" \
        --runs-dir "$SDIR/fio-runs" \
        --out "$SDIR/fio-summary.json"

      .runtime/h3-tools/ramulator-venv/bin/python \
        -m scripts.h3_campaign ramulator-actual \
        --residency "$RDIR/summary.json" \
        --out "$RDIR/lpddr-actual.json"
    done
  done
done
```

A phase with too little evidence to form the contract's minimum number of
genuinely paired request blocks fails closed.

## 10. Page-cache hostile controls

Required for full-run `session` and `long_context_8k` decisions.

```bash
for S in session long_context_8k; do
  PCDIR="$H3_ROOT/pagecache/$S/${PRIMARY_BUDGET}GiB"

  .venv/bin/python -m scripts.h3_campaign pagecache-window \
    --manifest "$H3_ROOT/trace-manifest.json" \
    --stratum "$S" --capacity-gib "$PRIMARY_BUDGET" \
    --ple "$PLE" --expert "$EXPERT" \
    --out-dir "$PCDIR"

  for MODE in buffered mmap; do
    .venv/bin/python -m scripts.h3_campaign pagecache-scope \
      --window "$PCDIR/window.json" --mode "$MODE" \
      --memory-gib "$PRIMARY_BUDGET" \
      --storage-binding "$H3_ROOT/storage-binding.json" \
      --out "$PCDIR/$MODE.json"
  done
done
```

## 11. Generate full and phase decisions

```bash
for P in $POLICIES; do
  for S in $STRATA; do
    RDIR="$H3_ROOT/replay/$S/${PRIMARY_BUDGET}GiB/$P"
    SDIR="$H3_ROOT/storage/full/$S/${PRIMARY_BUDGET}GiB/$P"
    DDIR="$H3_ROOT/decisions/full/$P/$S"
    mkdir -p "$DDIR"

    PC_ARGS=()
    if [ "$S" = "session" ] || [ "$S" = "long_context_8k" ]; then
      PCDIR="$H3_ROOT/pagecache/$S/${PRIMARY_BUDGET}GiB"
      PC_ARGS=(--pagecache "$PCDIR/buffered.json" --pagecache "$PCDIR/mmap.json")
    fi

    for QD in 1 8 32; do
      .venv/bin/python -m scripts.h3_campaign decide \
        --residency "$RDIR/summary.json" \
        --lpddr "$H3_ROOT/lpddr-generic.json" \
        --actual-lpddr "$RDIR/lpddr-actual.json" \
        --fio "$SDIR/fio-summary.json" --qd "$QD" \
        "${PC_ARGS[@]}" \
        --out "$DDIR/qd${QD}.json"
    done
  done
done

for PHASE in prefill decode; do
  for P in $POLICIES; do
    for S in $STRATA; do
      RDIR="$H3_ROOT/replay/$S/${PRIMARY_BUDGET}GiB/$P/$PHASE"
      SDIR="$H3_ROOT/storage/$PHASE/$S/${PRIMARY_BUDGET}GiB/$P"
      DDIR="$H3_ROOT/decisions/$PHASE/$P/$S"
      mkdir -p "$DDIR"

      for QD in 1 8 32; do
        .venv/bin/python -m scripts.h3_campaign decide \
          --residency "$RDIR/summary.json" \
          --lpddr "$H3_ROOT/lpddr-generic.json" \
          --actual-lpddr "$RDIR/lpddr-actual.json" \
          --fio "$SDIR/fio-summary.json" --qd "$QD" \
          --out "$DDIR/qd${QD}.json"
      done
    done
  done
done
```

The service-time optimized offline bound uses resident footprint for capacity and
measured lower-tier transfer footprint/cost for service savings. Impossible
perfect-prefetch bounds can falsify a gap but cannot prove conventional
sufficiency.

## 12. Derive placement invariance

Create an input from `configs/h3/placement_invariance_template_v6.json` and
supply at least two physical-placement controls for the required strata.

```bash
.venv/bin/python -m scripts.h3_campaign placement-invariance-evidence \
  --input "$H3_ROOT/placement-invariance-input.json" \
  --out "$H3_ROOT/placement-invariance.json"
```

The evaluator compares the ordered prompt/PLE/routing semantic sequence. Timing
differences remain provenance but do not cause semantic failure; reordering the
same PLE rows or selected experts is a semantic failure because it can change
cache locality.

## 13. Derive multi-length long-context evidence

Create the input from `configs/h3/long_context_scaling_template_v6.json` with
three distinct trace-valid case directories covering the 8K, 16K, and 32K
context bands.

```bash
.venv/bin/python -m scripts.h3_campaign long-context-evidence \
  --input "$H3_ROOT/long-context-input.json" \
  --out "$H3_ROOT/long-context-beyond-8k.json"
```

The artifact reports PLE-row and MoE-expert conditional-state metrics at every
context point. Passing this prerequisite establishes trace-valid multi-length
characterization only. It does not by itself establish monotonic memory-service
penalty or end-to-end inference scaling.

## 14. QD32 scheduler evidence for conventional sufficiency

QD32 can always be used as an optimistic gap-falsification point. It may prove
`CONVENTIONAL_MEMORY_SUFFICIENT` only if a separate workload-derived scheduler
measurement demonstrates that QD32 is realizable.

Start from `configs/h3/storage_scheduler_measurement_template_v6.json`. Populate
one raw event for every measured lower-tier object service with `request_id`,
`object_id`, `stratum`, `phase`, `observed_outstanding_io`,
`available_prefetch_slack_ns`, and `storage_completion_latency_ns`. Bind the
measurement to the exact `trace_manifest_sha256` and `storage_binding_sha256`.
All seven required strata must contain both prefill and decode evidence; each
stratum×phase cell requires at least 32 events from at least four requests.
Do not calculate the promotion-driving fractions yourself; FENIX derives them
from the event rows. Then derive evidence:

```bash
.venv/bin/python -m scripts.h3_campaign storage-concurrency-evidence \
  --measurement "$H3_ROOT/storage-scheduler-measurement.json" \
  --out "$H3_ROOT/storage-concurrency.json"
```

## 15. Bind promotion prerequisites

```bash
.venv/bin/python -m scripts.h3_campaign bind-prerequisites \
  --placement-invariance "$H3_ROOT/placement-invariance.json" \
  --long-context "$H3_ROOT/long-context-beyond-8k.json" \
  --capacity-budget "$H3_ROOT/capacity-budget.json" \
  --tool-qualification "$H3_ROOT/tool-qualification.json" \
  --out "$H3_ROOT/prerequisites.json"
```

If the preliminary campaign result is conventional sufficiency, include the
derived QD32 scheduler evidence:

```bash
.venv/bin/python -m scripts.h3_campaign bind-prerequisites \
  --placement-invariance "$H3_ROOT/placement-invariance.json" \
  --long-context "$H3_ROOT/long-context-beyond-8k.json" \
  --capacity-budget "$H3_ROOT/capacity-budget.json" \
  --tool-qualification "$H3_ROOT/tool-qualification.json" \
  --storage-concurrency "$H3_ROOT/storage-concurrency.json" \
  --out "$H3_ROOT/prerequisites.json"
```

## 16. Paper gate

Exactly one QD32 full, prefill, and decode decision is required for every
required `(stratum, policy)` point. The final directional bound is phase-aware:
a memory gap must survive every policy and every full/prefill/decode scope;
conventional sufficiency requires one fixed policy per stratum whose worst
full/prefill/decode upper bound remains within 5%.

```bash
ARGS=()

for P in $POLICIES; do
  for S in $STRATA; do
    ARGS+=(--decision "$H3_ROOT/decisions/full/$P/$S/qd32.json")
    ARGS+=(--decision "$H3_ROOT/decisions/prefill/$P/$S/qd32.json")
    ARGS+=(--decision "$H3_ROOT/decisions/decode/$P/$S/qd32.json")
  done
done

.venv/bin/python -m scripts.h3_campaign campaign-gate \
  "${ARGS[@]}" \
  --prerequisites "$H3_ROOT/prerequisites.json" \
  --out "$H3_ROOT/campaign-gate.json"
```

Promotion rejects duplicate/missing rows, mixed capacities, mixed
source/storage/tool fingerprints, incomplete actual-H3 calibration, invalid
prerequisites, or missing page-cache controls.

## Interpretation

* `H3_PAPER_GATE_SUPPORTED`: the >=10% conditional-memory-service gap survives
  all required conventional baselines and falsification sensitivities. Proceed
  to integrated timing/H4; do not call it end-to-end speedup.
* `H3_CONVENTIONAL_SUFFICIENT_STOP_H4`: the <=5% conventional result is
  realizable and independently demonstrates QD32 scheduler feasibility.
* Any other result remains inconclusive or blocked.
