#!/usr/bin/env python3
"""Generate publication-facing FENIX H1/H2/H3 figures from frozen evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/processed/paper_h1_h2_h3_20260908"
FIG = DATA / "figures"

EXACT = DATA / "exact_results.json"
HOTNESS = DATA / "moe_hotness_validation.json"
H2CSV = DATA / "h2_traffic_reduction.csv"
CANARY = (
    ROOT
    / "results/raw/h3_conventional_hierarchy_v6/20260907T072015Z"
    / "causal-analysis-3faa1cb6"
    / "decisions/session-useful_object_lru-full-q32.json"
)

EXPECTED_HEAD = "2dbf0b3663ddf993068acc1f8f5fdfc94a247ee4"


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_figure(fig, stem: str):
    outputs = []
    for ext in ("pdf", "svg", "png"):
        path = FIG / f"{stem}.{ext}"
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def pct(value):
    return 100.0 * float(value)


exact = load_json(EXACT)
hotness = load_json(HOTNESS)
decision = load_json(CANARY)

head = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    text=True,
).strip()

if head != EXPECTED_HEAD:
    raise RuntimeError(
        f"repository HEAD drift: {head} != {EXPECTED_HEAD}"
    )

if exact["final_code_head"] != EXPECTED_HEAD:
    raise RuntimeError("exact-results code HEAD mismatch")

if exact["h3"]["fio_summary_count"] != 84:
    raise RuntimeError("expected exactly 84 fio summaries")

if exact["h3"]["actual_lpddr_artifact_count"] != 84:
    raise RuntimeError("expected exactly 84 actual-H3 LPDDR artifacts")

if exact["h3"]["canary"]["verdict"] != "H3_MEMORY_GAP_SUPPORTED":
    raise RuntimeError("frozen H3 verdict drift")

if (
    exact["h3"]["canary"]["global_penalty_lower"]
    < exact["h3"]["canary"]["memory_gap_threshold"]
):
    raise RuntimeError("H3 lower bound no longer clears threshold")

FIG.mkdir(parents=True, exist_ok=True)

generated = []

# ----------------------------------------------------------------------
# H1 request-local concentration
# ----------------------------------------------------------------------

topks = [16, 32, 64, 128, 256]

request_local = {}
for k in topks:
    values = []
    for row in hotness["request_metrics"]:
        node = row["concentration"].get(str(k), {})
        value = node.get("mean")
        if value is not None:
            values.append(float(value))
    request_local[str(k)] = mean(values)

global_concentration = {
    str(k): float(
        exact["h1"]["mean_selection_concentration"][str(k)]
    )
    for k in topks
}

# ----------------------------------------------------------------------
# Figure A: global support versus frequency/request-locality
# ----------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))

axes[0].bar(
    ["PLE rows", "MoE layer-experts"],
    [
        pct(exact["h1"]["ple_fraction_of_address_space"]),
        pct(exact["h1"]["moe_union_fraction"]),
    ],
)
axes[0].set_ylabel("Observed fraction of addressable objects (%)")
axes[0].set_ylim(0, 105)
axes[0].set_title("Global observed support")
axes[0].grid(axis="y", alpha=0.25)

for i, value in enumerate(
    [
        pct(exact["h1"]["ple_fraction_of_address_space"]),
        pct(exact["h1"]["moe_union_fraction"]),
    ]
):
    axes[0].text(
        i,
        value + 2,
        f"{value:.3f}%",
        ha="center",
        va="bottom",
    )

axes[1].plot(
    topks,
    [pct(global_concentration[str(k)]) for k in topks],
    marker="o",
    label="Global aggregate frequency",
)
axes[1].plot(
    topks,
    [pct(request_local[str(k)]) for k in topks],
    marker="s",
    label="Mean request-local frequency",
)
axes[1].set_xlabel("Top-K experts per layer")
axes[1].set_ylabel("Selection share (%)")
axes[1].set_title("MoE selection concentration")
axes[1].set_xticks(topks)
axes[1].set_ylim(0, 100)
axes[1].grid(alpha=0.25)
axes[1].legend()

fig.suptitle(
    "H1: PLE sparsity and MoE locality are distinct phenomena"
)
fig.tight_layout()
generated += save_figure(fig, "H1_PLE_vs_MoE_locality")

# ----------------------------------------------------------------------
# Figure B: rolling-window hotness against uniform-routing null
# ----------------------------------------------------------------------

rolling = defaultdict(lambda: {"weighted_sum": 0.0, "n": 0})

for stratum_node in hotness["strata"].values():
    for row in stratum_node["rolling_windows"]:
        summary = row["observed_over_uniform_occupancy"]
        n = int(summary.get("n") or 0)
        value = summary.get("mean")
        if n <= 0 or value is None:
            continue
        key = (str(row["phase"]), int(row["window_tokens"]))
        rolling[key]["weighted_sum"] += float(value) * n
        rolling[key]["n"] += n

rolling_mean = {}
for key, node in rolling.items():
    rolling_mean[key] = node["weighted_sum"] / node["n"]

fig, ax = plt.subplots(figsize=(5.7, 3.8))

for phase in ("prefill", "decode", "all"):
    points = sorted(
        (
            window,
            value,
        )
        for (row_phase, window), value in rolling_mean.items()
        if row_phase == phase
    )
    if not points:
        continue
    ax.plot(
        [x for x, _ in points],
        [y for _, y in points],
        marker="o",
        label=phase,
    )

ax.axhline(
    1.0,
    linestyle="--",
    linewidth=1.0,
    label="Uniform-routing expectation",
)
ax.set_xscale("log", base=2)
ax.set_xlabel("Token window")
ax.set_ylabel("Observed / uniform expected unique experts")
ax.set_title("H1: expert hotness persists at short horizons")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
generated += save_figure(fig, "H1_windowed_expert_hotness")

# ----------------------------------------------------------------------
# Figure C: H2 traffic reduction versus conditional-state capacity
# ----------------------------------------------------------------------

with H2CSV.open() as f:
    h2rows = list(csv.DictReader(f))

for row in h2rows:
    for key, value in list(row.items()):
        if key != "budget_gib":
            row[key] = float(value)
    row["budget_gib"] = float(row["budget_gib"])

budgets = [row["budget_gib"] for row in h2rows]

series = [
    ("Homogeneous holdout", "homogeneous_holdout"),
    ("Same-domain", "same_domain_mean"),
    ("Cross-domain", "cross_domain_mean"),
    ("Multi-turn session", "session"),
    ("Long context", "long_context_8k"),
]

fig, ax = plt.subplots(figsize=(6.0, 3.9))

for label, key in series:
    ax.plot(
        budgets,
        [pct(row[key]) for row in h2rows],
        marker="o",
        label=label,
    )

ax.axvline(
    7.0,
    linestyle=":",
    linewidth=1.2,
    label="Primary 7 GiB point",
)
ax.set_xlabel("Conditional-state cache capacity (GiB)")
ax.set_ylabel("Logical lower-tier traffic reduction (%)")
ax.set_title("H2: capacity reduces conditional-state lower-tier traffic")
ax.set_xticks(budgets)
ax.set_ylim(0, 100)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
generated += save_figure(fig, "H2_traffic_reduction_vs_capacity")

# ----------------------------------------------------------------------
# Figure D: causal exact-expert prefetch feasibility
# ----------------------------------------------------------------------

causal = exact["h3"]["causality"]

phase_stats = {
    "Prefill": {
        "median": causal["prefill_median_slack_ns"] / 1000.0,
        "p95": causal["prefill_p95_slack_ns"] / 1000.0,
        "max": causal["prefill_max_slack_ns"] / 1000.0,
    },
    "Decode": {
        "median": causal["decode_median_slack_ns"] / 1000.0,
        "p95": causal["decode_p95_slack_ns"] / 1000.0,
        "max": causal["decode_max_slack_ns"] / 1000.0,
    },
}

storage_us = causal["fastest_one_expert_service_ns_lower"] / 1000.0

fig, ax = plt.subplots(figsize=(5.6, 3.9))
x = [0, 1]

for stat in ("median", "p95", "max"):
    ax.plot(
        x,
        [
            phase_stats["Prefill"][stat],
            phase_stats["Decode"][stat],
        ],
        marker="o",
        label=f"Native slack {stat}",
    )

ax.axhline(
    storage_us,
    linestyle="--",
    linewidth=1.3,
    label="Fastest measured 1-expert storage service (lower)",
)

ax.set_xticks(x)
ax.set_xticklabels(["Prefill", "Decode"])
ax.set_yscale("log")
ax.set_ylabel("Time (µs, log scale)")
ax.set_title("H3: exact expert IDs arrive too late to hide cold service")
ax.grid(alpha=0.25)
ax.legend(fontsize=8)

ax.text(
    0.02,
    0.97,
    (
        "storage / maximum native slack = "
        f"{causal['service_over_slack']:.1f}×"
    ),
    transform=ax.transAxes,
    va="top",
)

fig.tight_layout()
generated += save_figure(fig, "H3_causal_prefetch")

# ----------------------------------------------------------------------
# Recover the exact sensitivity/candidate that determines H3 lower bound.
# decision.py defines the point lower endpoint as the minimum candidate
# service lower bound, then takes the minimum across LPDDR sensitivities.
# ----------------------------------------------------------------------

global_low = float(
    exact["h3"]["canary"]["global_penalty_lower"]
)

points = decision["sensitivity_points"]
minimum_point_low = min(
    float(p["gap_falsification_penalty_fraction_bounds"][0])
    for p in points
)

tol = max(1e-12, abs(global_low) * 1e-10)

limiting = []
for point in points:
    point_low = float(
        point["gap_falsification_penalty_fraction_bounds"][0]
    )
    if abs(point_low - minimum_point_low) > tol:
        continue

    candidate = min(
        point["conventional_candidates"],
        key=lambda row: float(row["service_ns_bounds"][0]),
    )

    limiting.append(
        {
            "sensitivity_id": point.get("sensitivity_id"),
            "regime": point.get("regime"),
            "point_lower_penalty_fraction": point_low,
            "candidate": candidate.get("baseline"),
            "candidate_promotion_role":
                candidate.get("promotion_role"),
            "candidate_service_ns_bounds":
                candidate.get("service_ns_bounds"),
        }
    )

if abs(minimum_point_low - global_low) > tol:
    raise RuntimeError(
        "recovered limiting H3 sensitivity does not reproduce global lower bound"
    )

# ----------------------------------------------------------------------
# Figure E: hostile perfect-prefetch oracle versus causalized result.
# Only lower endpoints are plotted because the H3 support criterion is
# explicitly a lower-bound test against the predeclared 10% threshold.
# ----------------------------------------------------------------------

oracle_bounds = exact["h3"]["canary"][
    "oracle_prefetch_global_penalty_bounds"
]
causal_bounds = [
    exact["h3"]["canary"]["global_penalty_lower"],
    exact["h3"]["canary"]["global_penalty_upper"],
]
threshold = exact["h3"]["canary"]["memory_gap_threshold"]

labels = [
    "Unbounded perfect-\nexpert-prefetch oracle",
    "Causalized pinned\nruntime",
]
lower_values = [
    pct(oracle_bounds[0]),
    pct(causal_bounds[0]),
]

fig, ax = plt.subplots(figsize=(5.6, 3.9))
bars = ax.bar(labels, lower_values)

ax.axhline(
    pct(threshold),
    linestyle="--",
    linewidth=1.3,
    label=f"Predeclared gap threshold ({pct(threshold):.0f}%)",
)

for bar, value in zip(bars, lower_values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.0,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
    )

ax.set_ylabel("Global lower-bound service penalty (%)")
ax.set_title("H3: realizable causality restores a material service gap")
ax.set_ylim(
    0,
    max(
        50.0,
        pct(causal_bounds[0]) * 1.35,
    ),
)
ax.grid(axis="y", alpha=0.25)
ax.legend()

ax.text(
    0.98,
    0.96,
    f"Upper envelope: {pct(causal_bounds[1]):.1f}%",
    transform=ax.transAxes,
    ha="right",
    va="top",
)

fig.tight_layout()
generated += save_figure(fig, "H3_memory_service_gap")

# ----------------------------------------------------------------------
# Publication summary and figure provenance
# ----------------------------------------------------------------------

h2_7 = next(
    row for row in h2rows
    if math.isclose(row["budget_gib"], 7.0)
)

rolling_serializable = []
for (phase, window), value in sorted(rolling_mean.items()):
    rolling_serializable.append(
        {
            "phase": phase,
            "window_tokens": window,
            "observed_over_uniform_mean": value,
            "samples": rolling[(phase, window)]["n"],
        }
    )

summary = {
    "schema_version": 1,
    "artifact_kind": "fenix_publication_summary_h1_h2_h3",
    "repository": "afasolino/FENIX",
    "final_code_head": head,
    "h1": {
        "global": exact["h1"],
        "request_local_mean_selection_concentration":
            request_local,
        "request_local_semantics":
            "equal-weight mean across requests of per-request mean per-layer selection concentration",
        "rolling_hotness": rolling_serializable,
        "interpretation": (
            "PLE global address support is sparse; MoE eventual layer-expert "
            "support is nearly complete, while frequency and short-window "
            "selection remain concentrated."
        ),
    },
    "h2_primary_7gib": h2_7,
    "h2_claim_boundary":
        "capacity/logical-lower-tier-traffic only",
    "h3": {
        "causality": causal,
        "canary": exact["h3"]["canary"],
        "limiting_points": limiting,
        "claim_boundary":
            "conditional_memory_service_layer for the pinned measured host-driven runtime",
    },
}

summary_path = DATA / "paper_summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

sources = [EXACT, HOTNESS, H2CSV, CANARY]
manifest = {
    "schema_version": 1,
    "artifact_kind": "fenix_publication_figure_manifest",
    "sources": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sources
    },
    "figures": {
        str(path.relative_to(ROOT)): sha256(path)
        for path in generated
    },
}

manifest_path = DATA / "figure_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

print("=== PUBLICATION SUMMARY ===")
print(json.dumps(summary, indent=2))

print("\n=== GENERATED FIGURES ===")
for path in generated:
    print(path.relative_to(ROOT))

print("\nmanifest:", manifest_path.relative_to(ROOT))
