#!/usr/bin/env python3
"""Summarize per-step KL curves and pointwise clipping tail diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def metric_rows(run_dir: Path):
    rows = []
    for path in sorted((run_dir / "metrics").glob("round_*.json")):
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            payload = [payload]
        rows.extend(row for row in payload if "completed_updates" in row)
    return sorted(rows, key=lambda row: (int(row["completed_updates"]), int(row.get("update", 0))))


def write_csv(path: Path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(args):
    run_dir = args.run_dir.expanduser().resolve()
    output = (args.output_dir or run_dir / "analysis" / "kl_diagnostics").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = metric_rows(run_dir)
    if not metrics:
        raise ValueError(f"no per-step metrics found under {run_dir / 'metrics'}")

    curves = []
    tails = []
    aggregate = {}
    for metric in metrics:
        step = int(metric["completed_updates"])
        curves.append({
            "step": step,
            "optimization_direction": metric.get("kl_direction", "forward"),
            "optimization_kl": metric.get("optimization_kl", metric.get("forward_kl")),
            "forward_kl": metric.get("forward_kl"),
            "reverse_kl": metric.get("reverse_kl"),
            "grad_norm": metric.get("grad_norm"),
            "reasoning_tokens": metric.get("reasoning_tokens"),
        })
        for direction, detail in metric.get("pointwise_kl_statistics", {}).items():
            for threshold, tail in detail.get("thresholds", {}).items():
                row = {
                    "step": step, "direction": direction, "threshold": float(threshold),
                    "exceed_count": int(tail["exceed_count"]),
                    "affected_position_count": int(tail["affected_position_count"]),
                    "position_count": int(detail["position_count"]),
                    "vocabulary_entry_count": int(detail["vocabulary_entry_count"]),
                    "positive_contribution_count": int(detail["positive_contribution_count"]),
                    "positive_contribution_sum": float(detail["positive_contribution_sum"]),
                    "removed_positive_mass": float(tail["removed_positive_mass"]),
                    "fraction_of_all_vocabulary_entries": float(tail["fraction_of_all_vocabulary_entries"]),
                    "fraction_of_positions_affected": float(tail["fraction_of_positions_affected"]),
                    "fraction_of_positive_contributions": float(tail["fraction_of_positive_contributions"]),
                    "fraction_of_positive_mass_removed": float(tail["fraction_of_positive_mass_removed"]),
                    "implied_clipped_position_mean": float(tail["implied_clipped_position_mean"]),
                }
                tails.append(row)
                key = (direction, float(threshold))
                sums = aggregate.setdefault(key, {
                    "exceed_count": 0, "affected_positions": 0, "positions": 0,
                    "entries": 0, "positive_count": 0,
                    "positive_mass": 0.0, "removed_mass": 0.0,
                })
                sums["exceed_count"] += row["exceed_count"]
                sums["affected_positions"] += row["affected_position_count"]
                sums["positions"] += row["position_count"]
                sums["entries"] += row["vocabulary_entry_count"]
                sums["positive_count"] += row["positive_contribution_count"]
                sums["positive_mass"] += row["positive_contribution_sum"]
                sums["removed_mass"] += row["removed_positive_mass"]

    write_csv(output / "kl_by_step.csv", curves, list(curves[0]))
    if tails:
        write_csv(output / "pointwise_clip_by_step.csv", tails, list(tails[0]))

    aggregate_rows = []
    for (direction, threshold), sums in sorted(aggregate.items()):
        hit_fraction = sums["exceed_count"] / max(sums["entries"], 1)
        position_fraction = sums["affected_positions"] / max(sums["positions"], 1)
        positive_hit_fraction = sums["exceed_count"] / max(sums["positive_count"], 1)
        removed_fraction = sums["removed_mass"] / max(sums["positive_mass"], 1e-300)
        eligible = (hit_fraction <= args.max_entry_hit_fraction
                    and removed_fraction <= args.max_removed_positive_mass_fraction)
        aggregate_rows.append({
            "direction": direction, "threshold": threshold,
            "exceed_count": sums["exceed_count"],
            "affected_position_count": sums["affected_positions"],
            "fraction_of_positions_affected": position_fraction,
            "fraction_of_all_vocabulary_entries": hit_fraction,
            "fraction_of_positive_contributions": positive_hit_fraction,
            "removed_positive_mass": sums["removed_mass"],
            "fraction_of_positive_mass_removed": removed_fraction,
            "meets_heuristic": eligible,
        })
    if aggregate_rows:
        write_csv(output / "pointwise_clip_aggregate.csv", aggregate_rows, list(aggregate_rows[0]))

    recommendations = {}
    for direction in ("forward", "reverse"):
        candidates = [row for row in aggregate_rows
                      if row["direction"] == direction and row["meets_heuristic"]]
        recommendations[direction] = min(
            (row["threshold"] for row in candidates), default=None)
    summary = {
        "run_dir": str(run_dir), "steps": [int(curves[0]["step"]), int(curves[-1]["step"])],
        "diagnostic_steps": len({row["step"] for row in tails}),
        "heuristic": {
            "max_fraction_of_all_entries_clipped": args.max_entry_hit_fraction,
            "max_fraction_of_positive_mass_removed": args.max_removed_positive_mass_fraction,
            "rule": "smallest measured threshold satisfying both limits",
        },
        "recommended_measured_threshold": recommendations,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(11, 6))
        steps = [row["step"] for row in curves]
        for key, label in (("forward_kl", "Forward KL"), ("reverse_kl", "Reverse KL")):
            points = [(step, row[key]) for step, row in zip(steps, curves) if row[key] is not None]
            if points:
                axis.plot([point[0] for point in points], [point[1] for point in points],
                          marker="o", markersize=3, linewidth=1.5, label=label)
        axis.set_xlabel("Completed optimizer step")
        axis.set_ylabel("Configured-clipped sequence-mean KL")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / "kl_curve.png", dpi=180)
        plt.close(fig)

        if aggregate_rows:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
            for direction in ("forward", "reverse"):
                selected = [row for row in aggregate_rows if row["direction"] == direction]
                axes[0].plot([row["threshold"] for row in selected],
                             [row["fraction_of_all_vocabulary_entries"] for row in selected],
                             marker="o", label=direction)
                axes[1].plot([row["threshold"] for row in selected],
                             [row["fraction_of_positive_mass_removed"] for row in selected],
                             marker="o", label=direction)
            for axis in axes:
                axis.set_xscale("log")
                axis.set_yscale("log")
                axis.grid(alpha=0.25)
                axis.legend()
                axis.set_xlabel("Pointwise clip threshold")
            axes[0].set_ylabel("Fraction of all vocabulary entries clipped")
            axes[1].set_ylabel("Fraction of positive contribution mass removed")
            fig.tight_layout()
            fig.savefig(output / "pointwise_clip_distribution.png", dpi=180)
            plt.close(fig)
    except ImportError:
        pass

    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-entry-hit-fraction", type=float, default=1e-4)
    parser.add_argument("--max-removed-positive-mass-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.max_entry_hit_fraction <= 1:
        parser.error("--max-entry-hit-fraction must lie in [0,1]")
    if not 0 <= args.max_removed_positive_mass_fraction <= 1:
        parser.error("--max-removed-positive-mass-fraction must lie in [0,1]")
    analyze(args)


if __name__ == "__main__":
    main()
