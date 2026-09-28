#!/usr/bin/env python3
"""Plot recorded P5 development curves and the heldout success matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="v2")
    args = parser.parse_args()
    target = ROOT / "docs/verification" / ("p5-results-" + args.revision)
    summary = json.loads((target / "summary.json").read_text())
    curves = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, task in zip(axes, ("static", "dynamic"), strict=True):
        for budget in summary["budgets"]:
            if budget["task"] != task:
                continue
            reports = sorted((ROOT / "experiments" / budget["run_id"] / "eval").glob("step-*.json"))
            points = []
            for path in reports:
                report = json.loads(path.read_text())
                row = dict(
                    task=task,
                    sensor=budget["sensor"],
                    method=budget["method"],
                    step=report["step"],
                    success_rate=report["success_rate"],
                    collision_rate=report["collision_rate"],
                    out_of_bounds_rate=report["out_of_bounds"] / report["num_trials"],
                    timeout_rate=report["timeout"] / report["num_trials"],
                    return_mean=report["return_mean"],
                    constrained_time_s=report["constrained_time_mean_s"],
                    source=str(path.relative_to(ROOT)),
                )
                curves.append(row)
                points.append(row)
            if points:
                label = budget["sensor"] + "-" + budget["method"].upper()
                axis.plot(
                    [x["step"] / 1e6 for x in points],
                    [x["success_rate"] for x in points],
                    marker="o",
                    markersize=3,
                    label=label,
                )
        axis.set(
            title=task.capitalize(),
            xlabel="Training interactions (million)",
            xlim=(0, 8.388608),
            ylim=(-0.025, 1.025),
        )
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend(fontsize=8)
    axes[0].set_ylabel("Development macro success rate")
    fig.suptitle("Recorded development checkpoints — seed 0")
    fig.tight_layout()
    fig.savefig(target / "development-curves.png", dpi=180)
    fig.savefig(target / "development-curves.pdf")
    plt.close(fig)
    if curves:
        with (target / "development-curves.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, curves[0].keys())
            writer.writeheader()
            writer.writerows(curves)

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    metrics = (
        ("collision_rate", "Collision rate"),
        ("out_of_bounds_rate", "Out-of-bounds rate"),
        ("return_mean", "Mean undiscounted return"),
    )
    for column, task in enumerate(("static", "dynamic")):
        for row, (metric, label) in enumerate(metrics):
            axis = axes[row, column]
            for sensor, method in (
                ("depth", "ppo"),
                ("depth", "dva"),
                ("lidar", "ppo"),
                ("lidar", "dva"),
            ):
                points = [
                    x
                    for x in curves
                    if (x["task"], x["sensor"], x["method"]) == (task, sensor, method)
                ]
                if points:
                    axis.plot(
                        [x["step"] / 1e6 for x in points],
                        [x[metric] for x in points],
                        marker="o",
                        markersize=3,
                        label=sensor + "-" + method.upper(),
                    )
            axis.set_ylabel(label)
            axis.set_xlim(0, 8.388608)
            if metric.endswith("rate"):
                axis.set_ylim(-0.025, 1.025)
            axis.grid(alpha=0.25)
            if row == 0:
                axis.set_title(task.capitalize())
                if axis.lines:
                    axis.legend(fontsize=8)
            if row == 2:
                axis.set_xlabel("Training interactions (million)")
    fig.suptitle("Development outcomes and return — fixed checkpoint schedule")
    fig.tight_layout()
    fig.savefig(target / "development-diagnostics.png", dpi=180)
    fig.savefig(target / "development-diagnostics.pdf")
    plt.close(fig)

    heldout = [x for x in summary["cells"] if x["split"] == "heldout"]
    keys = list(dict.fromkeys((x["task"], x["sensor"], x["method"]) for x in heldout))
    difficulties = ("easy", "medium", "hard")
    matrix = np.full((len(keys), 3), np.nan)
    for cell in heldout:
        if cell["status"] == "completed":
            matrix[
                keys.index((cell["task"], cell["sensor"], cell["method"])),
                difficulties.index(cell["difficulty"]),
            ] = cell["success_rate"]
    fig, axis = plt.subplots(figsize=(7, 7))
    cmap = plt.colormaps["YlGnBu"].copy()
    cmap.set_bad("#eeeeee")
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for i in range(len(keys)):
        for j in range(3):
            value = matrix[i, j]
            axis.text(
                j,
                i,
                "pending" if np.isnan(value) else f"{value:.1%}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if not np.isnan(value) and value > 0.65 else "black",
            )
    axis.set_xticks(range(3), difficulties)
    axis.set_yticks(range(len(keys)), [f"{t} / {s}-{m.upper()}" for t, s, m in keys])
    axis.set_title(
        "Heldout success rate — 128 episodes per cell\n"
        + ("Complete matrix" if summary["matrix_complete"] else "Partial evidence")
    )
    fig.colorbar(image, ax=axis, label="Success rate", shrink=0.75)
    fig.tight_layout()
    fig.savefig(target / "heldout-matrix.png", dpi=180)
    fig.savefig(target / "heldout-matrix.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
