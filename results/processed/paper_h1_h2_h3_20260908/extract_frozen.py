from pathlib import Path
from collections import Counter
import csv
import hashlib
import json

ROOT = Path.cwd()
OUT = ROOT / "results/processed/paper_h1_h2_h3_20260908"
TRACE = ROOT / "results/raw/h1_h2_workload_robustness_noprefix/20260904T162349Z"
H3 = ROOT / "results/raw/h3_conventional_hierarchy_v6/20260907T072015Z"
FROZEN = H3 / "final-e08bc8d"
ORACLE = H3 / "oracle-analysis-8b157b68"
CAUSAL = H3 / "causal-analysis-3faa1cb6"

def load(path):
    return json.loads(path.read_text())

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def concentration_value(rows, k):
    for row in rows:
        if int(row["topk_experts_per_layer"]) == int(k):
            return row["mean_selection_fraction"]
    return None

# ------------------------------------------------------------------
# H1 exact natural-workload global union and frequency concentration.
# Streaming only: no raw data are copied or modified.
# ------------------------------------------------------------------

campaign = load(ROOT / "configs/campaign.json")
model = campaign["model"]
layers = int(model["num_hidden_layers"])
experts_per_layer = int(model["num_experts"])
addressable_rows = int(model["ple_addressable_rows"])

ple_union = set()
ple_row_bytes = set()
expert_counts = [Counter() for _ in range(layers)]

case_dirs = sorted(p for p in TRACE.glob("s-*-r01") if p.is_dir())

for case in case_dirs:
    ple_path = case / "ple_normalized.jsonl"
    with ple_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ple_union.add(int(row["physical_row_id"]))
            if row.get("bytes") is not None:
                ple_row_bytes.add(int(row["bytes"]))

    moe_path = case / "moe_normalized.jsonl"
    with moe_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_layer = row["layer"]
            if isinstance(raw_layer, int):
                layer = raw_layer
            else:
                text = str(raw_layer)
                digits = "".join(ch if ch.isdigit() else " " for ch in text).split()
                layer = int(digits[-1])
            for expert in row.get("selected_expert_ids", []):
                expert_counts[layer][int(expert)] += 1

if len(ple_row_bytes) != 1:
    raise RuntimeError(f"unexpected PLE row widths: {sorted(ple_row_bytes)}")

row_bytes = next(iter(ple_row_bytes))
expert_union = sum(len(c) for c in expert_counts)
expert_total = layers * experts_per_layer

global_concentration = {}
for k in (16, 32, 64, 128, 256):
    fractions = []
    for counts in expert_counts:
        total = sum(counts.values())
        if total:
            fractions.append(
                sum(v for _, v in counts.most_common(k)) / total
            )
    global_concentration[str(k)] = sum(fractions) / len(fractions)

h1_global = {
    "ple_union_rows": len(ple_union),
    "ple_row_bytes": row_bytes,
    "ple_union_bytes": len(ple_union) * row_bytes,
    "ple_union_mib": len(ple_union) * row_bytes / 1024**2,
    "ple_fraction_of_address_space": len(ple_union) / addressable_rows,
    "ple_addressable_rows": addressable_rows,
    "moe_union_layer_experts": expert_union,
    "moe_total_layer_experts": expert_total,
    "moe_union_fraction": expert_union / expert_total,
    "mean_selection_concentration": global_concentration,
}

# Per-stratum H1 rows from the frozen robustness evaluator.
h1_rows = []
h1_dir = OUT / "h1_h2/h1"

for path in sorted(h1_dir.glob("*.json")):
    if path.name == "cross_stratum.json":
        continue
    obj = load(path)
    conc = obj["experts"]["concentration"]
    h1_rows.append({
        "scope": obj["stratum"],
        "request_count": obj["request_count"],
        "model_token_observations": obj["model_token_observations"],
        "ple_unique_rows": obj["ple"]["unique_rows"],
        "ple_unique_mib": obj["ple"]["unique_bytes"] / 1024**2,
        "ple_fraction_of_address_space":
            obj["ple"]["unique_row_fraction_of_table"],
        "moe_unique_layer_experts":
            obj["experts"]["unique_layer_objects"],
        "moe_union_fraction":
            obj["experts"]["unique_fraction_of_all_layer_experts"],
        "top64_selection_fraction": concentration_value(conc, 64),
        "top128_selection_fraction": concentration_value(conc, 128),
        "top256_selection_fraction": concentration_value(conc, 256),
    })

h1_rows.insert(0, {
    "scope": "global_natural_workload_union",
    "request_count": "",
    "model_token_observations": "",
    "ple_unique_rows": h1_global["ple_union_rows"],
    "ple_unique_mib": h1_global["ple_union_mib"],
    "ple_fraction_of_address_space":
        h1_global["ple_fraction_of_address_space"],
    "moe_unique_layer_experts":
        h1_global["moe_union_layer_experts"],
    "moe_union_fraction": h1_global["moe_union_fraction"],
    "top64_selection_fraction":
        h1_global["mean_selection_concentration"]["64"],
    "top128_selection_fraction":
        h1_global["mean_selection_concentration"]["128"],
    "top256_selection_fraction":
        h1_global["mean_selection_concentration"]["256"],
})

write_csv(
    OUT / "h1_locality.csv",
    h1_rows,
    [
        "scope",
        "request_count",
        "model_token_observations",
        "ple_unique_rows",
        "ple_unique_mib",
        "ple_fraction_of_address_space",
        "moe_unique_layer_experts",
        "moe_union_fraction",
        "top64_selection_fraction",
        "top128_selection_fraction",
        "top256_selection_fraction",
    ],
)

# ------------------------------------------------------------------
# H1 token/window hotness.
# ------------------------------------------------------------------

hotness = load(OUT / "moe_hotness_validation.json")
window_rows = []

for stratum, node in sorted(hotness["strata"].items()):
    for row in node["rolling_windows"]:
        out = {
            "stratum": stratum,
            "phase": row["phase"],
            "window_tokens": row["window_tokens"],
            "uniform_expected_unique_experts":
                row["uniform_expected_unique_experts"],
            "observed_unique_mean":
                row["observed_unique_experts"]["mean"],
            "observed_unique_median":
                row["observed_unique_experts"]["median"],
            "observed_unique_min":
                row["observed_unique_experts"]["min"],
            "observed_unique_max":
                row["observed_unique_experts"]["max"],
            "observed_over_uniform_mean":
                row["observed_over_uniform_occupancy"]["mean"],
            "effective_expert_count_mean":
                row["effective_expert_count"]["mean"],
        }
        for k, summary in row["concentration"].items():
            out[f"top{k}_concentration_mean"] = summary["mean"]
        window_rows.append(out)

all_window_fields = [
    "stratum",
    "phase",
    "window_tokens",
    "uniform_expected_unique_experts",
    "observed_unique_mean",
    "observed_unique_median",
    "observed_unique_min",
    "observed_unique_max",
    "observed_over_uniform_mean",
    "effective_expert_count_mean",
]
for k in sorted(
    {
        key
        for row in window_rows
        for key in row
        if key.startswith("top")
    }
):
    all_window_fields.append(k)

write_csv(
    OUT / "h1_windowed_hotness.csv",
    window_rows,
    all_window_fields,
)

# ------------------------------------------------------------------
# H2 exact robustness curves.
# ------------------------------------------------------------------

h2 = load(OUT / "h1_h2/h2/robustness_replay.json")
homogeneous = load(
    ROOT /
    "results/processed/h1_h2_edge_v2/20260904T113013Z/summary.json"
)

homogeneous_by_budget = {
    float(row["budget_gib"]):
        row["mean_conditional_lower_tier_bytes_reduction_fraction"]
    for row in homogeneous["h2"]["holdout_summary"]
}

budgets = sorted(
    {float(row["budget_gib"]) for row in h2["in_domain"]}
)

h2_rows = []

for budget in budgets:
    same = [
        row["static_frequency"][
            "conditional_lower_tier_bytes_reduction_fraction"
        ]
        for row in h2["in_domain"]
        if float(row["budget_gib"]) == budget
    ]

    cross = [
        row["static_frequency"][
            "conditional_lower_tier_bytes_reduction_fraction"
        ]
        for row in h2["leave_one_domain_out"]
        if float(row["budget_gib"]) == budget
    ]

    structural = {
        row["held_out_stratum"]:
            row["static_frequency"][
                "conditional_lower_tier_bytes_reduction_fraction"
            ]
        for row in h2["structural_holdouts"]
        if float(row["budget_gib"]) == budget
    }

    online = [
        row["conditional_lower_tier_bytes_reduction_fraction"]
        for row in h2["mixed_online"]
        if float(row["budget_gib"]) == budget
    ]

    h2_rows.append({
        "budget_gib": budget,
        "homogeneous_holdout":
            homogeneous_by_budget.get(budget),
        "same_domain_mean": sum(same) / len(same),
        "same_domain_min": min(same),
        "same_domain_max": max(same),
        "cross_domain_mean": sum(cross) / len(cross),
        "cross_domain_min": min(cross),
        "cross_domain_max": max(cross),
        "session":
            structural.get("session"),
        "long_context_8k":
            structural.get("long_context_8k"),
        "mixed_online_mean":
            sum(online) / len(online),
        "mixed_online_min":
            min(online),
        "mixed_online_max":
            max(online),
    })

write_csv(
    OUT / "h2_traffic_reduction.csv",
    h2_rows,
    [
        "budget_gib",
        "homogeneous_holdout",
        "same_domain_mean",
        "same_domain_min",
        "same_domain_max",
        "cross_domain_mean",
        "cross_domain_min",
        "cross_domain_max",
        "session",
        "long_context_8k",
        "mixed_online_mean",
        "mixed_online_min",
        "mixed_online_max",
    ],
)

# ------------------------------------------------------------------
# H3 exact causality and canary.
# ------------------------------------------------------------------

causality_path = CAUSAL / "expert-prefetch-causality.json"
canary_path = (
    CAUSAL /
    "decisions/session-useful_object_lru-full-q32.json"
)
oracle_decision_path = (
    ORACLE /
    "decisions/session-useful_object_lru-full-q32.json"
)

causality = load(causality_path)
canary = load(canary_path)
oracle_decision = load(oracle_decision_path)

ratio = causality["slack_over_fastest_one_expert_service"]

h3_causality = {
    "records_used": causality["records_used"],
    "required_layers": causality["required_layers"],
    "derived_pass": causality["derived_pass"],
    "prefill_events":
        causality["phase_stats"]["prefill"]["events"],
    "decode_events":
        causality["phase_stats"]["decode"]["events"],
    "prefill_median_slack_ns":
        causality["phase_stats"]["prefill"]["median_slack_ns"],
    "prefill_p95_slack_ns":
        causality["phase_stats"]["prefill"]["p95_slack_ns"],
    "prefill_max_slack_ns":
        causality["phase_stats"]["prefill"]["maximum_slack_ns"],
    "decode_median_slack_ns":
        causality["phase_stats"]["decode"]["median_slack_ns"],
    "decode_p95_slack_ns":
        causality["phase_stats"]["decode"]["p95_slack_ns"],
    "decode_max_slack_ns":
        causality["phase_stats"]["decode"]["maximum_slack_ns"],
    "maximum_native_prefetch_slack_ns":
        causality["maximum_native_prefetch_slack_ns"],
    "fastest_one_expert_service_ns_lower":
        causality[
            "fastest_measured_one_expert_transfer_ns_lower"
        ],
    "slack_over_service": ratio,
    "service_over_slack": 1.0 / ratio,
}

bounds = canary["global_penalty_fraction_bounds"]
threshold = canary["thresholds"][
    "memory_gap_min_penalty_fraction"
]

h3_gap = {
    "stratum": canary.get("stratum"),
    "policy": canary.get("policy"),
    "phase_filter": canary.get("phase_filter"),
    "queue_depth":
        canary.get("queue_depth")
        or canary.get("storage_queue_depth")
        or 32,
    "verdict": canary["verdict"],
    "global_penalty_lower": bounds[0],
    "global_penalty_upper": bounds[1],
    "memory_gap_threshold": threshold,
    "causal_expert_prefetch_evidence_used":
        canary.get("causal_expert_prefetch_evidence_used"),
    "ple_perfect_prefetch_sensitivity_retained":
        canary.get(
            "ple_perfect_prefetch_sensitivity_retained"
        ),
    "causal_belady_pagecache_used":
        canary.get("causal_belady_pagecache_used"),
    "oracle_prefetch_global_penalty_bounds":
        oracle_decision.get(
            "global_penalty_fraction_bounds"
        ),
}

def limiting_fields(obj, prefix=""):
    rows = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if "limit" in key.lower():
                rows[path] = value
            if isinstance(value, (dict, list)):
                rows.update(limiting_fields(value, path))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if isinstance(value, (dict, list)):
                rows.update(
                    limiting_fields(value, f"{prefix}[{i}]")
                )
    return rows

h3_gap["limiting_fields"] = limiting_fields(canary)

write_csv(
    OUT / "h3_causal_prefetch.csv",
    [h3_causality],
    list(h3_causality),
)

write_csv(
    OUT / "h3_memory_service_gap.csv",
    [{
        "stratum": h3_gap["stratum"],
        "policy": h3_gap["policy"],
        "phase_filter": h3_gap["phase_filter"],
        "queue_depth": h3_gap["queue_depth"],
        "verdict": h3_gap["verdict"],
        "global_penalty_lower":
            h3_gap["global_penalty_lower"],
        "global_penalty_upper":
            h3_gap["global_penalty_upper"],
        "memory_gap_threshold":
            h3_gap["memory_gap_threshold"],
        "causal_expert_prefetch_evidence_used":
            h3_gap["causal_expert_prefetch_evidence_used"],
        "ple_perfect_prefetch_sensitivity_retained":
            h3_gap[
                "ple_perfect_prefetch_sensitivity_retained"
            ],
        "causal_belady_pagecache_used":
            h3_gap["causal_belady_pagecache_used"],
    }],
    [
        "stratum",
        "policy",
        "phase_filter",
        "queue_depth",
        "verdict",
        "global_penalty_lower",
        "global_penalty_upper",
        "memory_gap_threshold",
        "causal_expert_prefetch_evidence_used",
        "ple_perfect_prefetch_sensitivity_retained",
        "causal_belady_pagecache_used",
    ],
)

# ------------------------------------------------------------------
# Compact evidence inventory: result-level artifacts only.
# Avoid 11k fio run meta files and bulk traces.
# ------------------------------------------------------------------

principal = []

principal += sorted((OUT / "h1_h2/h1").glob("*.json"))
principal += [
    OUT / "h1_h2/h2/robustness_replay.json",
    OUT / "h1_h2/summary.json",
    OUT / "moe_hotness_validation.json",
]

for p in sorted(FROZEN.rglob("fio-summary.json")):
    principal.append(p)

for p in sorted((FROZEN / "lpddr-actual").rglob("*.json")):
    try:
        if load(p).get("artifact_kind") == \
                "fenix_h3_lpddr_actual_trace_calibration":
            principal.append(p)
    except Exception:
        pass

principal += [
    FROZEN / "lpddr-generic.json",
    FROZEN / "capacity-budget.json",
    FROZEN / "storage-binding.json",
    FROZEN / "tool-qualification.json",
    ORACLE / "pagecache-oracle/session.json",
    oracle_decision_path,
    causality_path,
    canary_path,
]

seen = set()
inventory = []

for path in principal:
    if not path.is_file():
        continue
    path = path.resolve()
    if path in seen:
        continue
    seen.add(path)

    obj = load(path)

    repo_head = None
    for node in (
        obj.get("execution_repository"),
        obj.get("measurement_execution_repository"),
        (obj.get("provenance") or {}).get(
            "execution_repository"
        ),
        obj.get("source"),
    ):
        if isinstance(node, dict):
            repo_head = (
                node.get("head")
                or node.get("repository_commit")
                or repo_head
            )

    if repo_head is None:
        repo_head = (
            obj.get("frozen_measurement_head")
            or obj.get("source_repository_commit")
        )

    phase = obj.get("phase_filter")
    if phase is None and isinstance(obj.get("phase"), str):
        phase = obj.get("phase")

    inventory.append({
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "artifact_kind": obj.get("artifact_kind"),
        "evidence_kind": obj.get("evidence_kind"),
        "repository_head": repo_head,
        "hypothesis": obj.get("hypothesis"),
        "stratum": obj.get("stratum"),
        "policy": obj.get("policy"),
        "phase": phase,
        "capacity_gib":
            obj.get("capacity_gib")
            or obj.get("budget_gib"),
        "queue_depth":
            obj.get("queue_depth")
            or obj.get("storage_queue_depth"),
        "verdict": obj.get("verdict"),
        "global_penalty_fraction_bounds":
            obj.get("global_penalty_fraction_bounds"),
        "derived_pass": obj.get("derived_pass"),
    })

with (OUT / "evidence_inventory.jsonl").open("w") as f:
    for row in inventory:
        f.write(json.dumps(row, sort_keys=True) + "\n")

# ------------------------------------------------------------------
# Frozen consolidated result.
# ------------------------------------------------------------------

exact = {
    "schema_version": 1,
    "artifact_kind":
        "fenix_h1_h2_h3_publication_exact_results",
    "repository": "afasolino/FENIX",
    "final_code_head":
        "2dbf0b3663ddf993068acc1f8f5fdfc94a247ee4",
    "scientific_heads": {
        "h1_h2":
            "8b8694647840e5af73fd4e1fb1275d0b15c19056",
        "h3_measurement":
            "e08bc8d32b85ef3fe929d350ca122159fcee4073",
        "h3_causal_analysis":
            "3faa1cb6763beb092a4bb5de4639fac1cf09d97d",
    },
    "h1": h1_global,
    "h2": {
        "metric":
            "conditional_lower_tier_bytes_reduction_fraction",
        "rows": h2_rows,
        "claim_boundary":
            "capacity/logical-lower-tier-traffic only",
    },
    "h3": {
        "scope": "conditional_memory_service_layer",
        "causality": h3_causality,
        "canary": h3_gap,
        "fio_summary_count":
            len(list(FROZEN.rglob("fio-summary.json"))),
        "actual_lpddr_artifact_count":
            len([
                p for p in principal
                if p.is_file()
                and "lpddr-actual" in str(p)
                and load(p).get("artifact_kind")
                == "fenix_h3_lpddr_actual_trace_calibration"
            ]),
    },
    "inventory_artifact_count": len(inventory),
}

(OUT / "exact_results.json").write_text(
    json.dumps(exact, indent=2) + "\n"
)

print("\n=== EXACT FROZEN RESULTS ===")
print(json.dumps(exact, indent=2))
print("\n=== OUTPUT FILES ===")
for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(p)
