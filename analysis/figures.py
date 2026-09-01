#!/usr/bin/env python3
"""Generate FENIX motivation and locality figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def plot_grouped(
    rows: Iterable[dict[str, object]],
    *,
    x_field: str,
    y_field: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    group_field: str | None = None,
) -> None:
    materialized = list(rows)
    groups = (
        sorted({str(row[group_field]) for row in materialized})
        if group_field
        else [None]
    )

    figure, axis = plt.subplots()

    for group in groups:
        selected = [
            row
            for row in materialized
            if (group_field is None or str(row[group_field]) == group)
            and row.get(x_field) is not None
            and row.get(y_field) is not None
        ]
        selected.sort(key=lambda row: float(row[x_field]))

        if selected:
            axis.plot(
                [float(row[x_field]) for row in selected],
                [float(row[y_field]) for row in selected],
                marker="o",
                label=group,
            )

    if group_field:
        axis.legend()

    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def load_rows(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text()).get("rows", [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ple-locality", type=Path)
    parser.add_argument("--capacity-tradeoff", type=Path)
    parser.add_argument("--measured-results", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.ple_locality:
        locality = json.loads(args.ple_locality.read_text())
        plot_grouped(
            locality["cache_curve"],
            x_field="capacity_gib",
            y_field="hit_rate",
            title="PLE cache hit rate versus cache capacity",
            x_label="Cache capacity (GiB)",
            y_label="Hit rate",
            output_path=args.out_dir / "ple_cache_hit_vs_capacity.png",
        )

    if args.capacity_tradeoff:
        rows = load_rows(args.capacity_tradeoff)
        plot_grouped(
            rows,
            x_field="host_budget_gib",
            y_field="expert_capacity",
            title="Expert residency capacity versus host-memory budget",
            x_label="Host-memory budget (GiB)",
            y_label="Resident expert capacity",
            output_path=args.out_dir / "expert_capacity_vs_host_memory.png",
            group_field="placement",
        )
        plot_grouped(
            rows,
            x_field="host_budget_gib",
            y_field="expert_storage_bytes_per_selection",
            title="Projected cold-expert traffic versus host-memory budget",
            x_label="Host-memory budget (GiB)",
            y_label="Expert storage bytes per selection",
            output_path=args.out_dir / "expert_storage_traffic_vs_host_memory.png",
            group_field="placement",
        )

    if args.measured_results and args.measured_results.exists():
        rows = load_rows(args.measured_results)
        plot_grouped(
            rows,
            x_field="host_budget_gib",
            y_field="tokens_s",
            title="Measured decode throughput versus host-memory budget",
            x_label="Host-memory budget (GiB)",
            y_label="Decode tokens/s",
            output_path=args.out_dir / "decode_throughput_vs_host_memory.png",
            group_field="placement",
        )


if __name__ == "__main__":
    main()
