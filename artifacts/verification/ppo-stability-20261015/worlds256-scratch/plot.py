"""Plot observed acquisition and case retention without rerunning evaluations."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
for filename, label, color in (
    ("depth-fixed-scenes-parallel-progress.json", "64 worlds", "#4477AA"),
    ("depth-fixed-scenes-worlds256-scratch-progress.json", "256 worlds", "#CC6677"),
):
    checks = json.loads((HERE.parent / filename).read_text())["checks"]
    interactions, counts, retention = [], [], []
    previous = set()
    for row in checks:
        current = {tuple(case[:4]) for case in row["case_events"] if case[4] == "SUCCESS"}
        assert len(row["case_events"]) == 48 and len(current) == row["successes"]
        interactions.append(row["interactions"] / 1e6)
        counts.append(len(current))
        retention.append(
            100 * len(current & previous) / len(previous) if previous else float("nan")
        )
        previous = current
    axes[0].plot(interactions, counts, "o--", color=color, label=label, linewidth=1.3)
    axes[1].plot(interactions, retention, "o--", color=color, label=label, linewidth=1.3)

axes[0].set(title="Depth PPO seed 0: acquisition and retention", ylabel="Screening successes / 48")
axes[0].set_ylim(-2, 50)
axes[0].set_yticks([0, 8, 16, 24, 32, 40, 48])
axes[0].legend(loc="center right")
axes[1].set(
    xlabel="Training interactions (millions)", ylabel="Previous successful cases retained (%)"
)
axes[1].set_ylim(-4, 104)
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
fig.text(
    0.5,
    -0.16,
    "Fixed 48-case development screening; one seed. Missing retention means no prior successes.\n"
    "First full validation: 64 worlds 150/150 at 5.77M; 256 worlds 149/150 at 3.15M.\n"
    "These curves do not establish post-selection stability or final acceptance.",
    ha="center",
    fontsize=9,
)
fig.savefig(HERE / "acquisition.png", dpi=180, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="constrained")
for filename, label, color in (
    ("depth-fixed-scenes-parallel-stability-report.json", "64 worlds", "#4477AA"),
    ("worlds256-scratch-stability/report.json", "256 worlds", "#CC6677"),
):
    result = json.loads((HERE.parent / filename).read_text())
    checks = result["checks"]
    original = sum(case[4] == "SUCCESS" for case in result["baseline_cases"])
    assert result["status"] == "completed" and len(checks) == 7
    interactions = [0, *[row["additional_interactions"] / 1e6 for row in checks]]
    retention = [100, *[100 * row["retained_initial_successes"] / original for row in checks]]
    initial_counts = [
        sum(case[0] == scene and case[4] == "SUCCESS" for case in result["baseline_cases"])
        for scene in ("S01", "S02", "S03", "D01", "D02", "D03")
    ]
    worst = [min(initial_counts), *[min(row["per_scene_successes"].values()) for row in checks]]
    axes[0].plot(interactions, retention, "o--", color=color, label=label, linewidth=1.3)
    axes[1].plot(interactions, worst, "o--", color=color, label=label, linewidth=1.3)

axes[0].set(
    title="Depth PPO seed 0: unchanged continuation after first full pass",
    ylabel="Original successful cases retained (%)",
    ylim=(0, 104),
)
axes[0].axhline(95, color="black", linestyle=":", linewidth=1, label="Required: 95%")
axes[0].legend(loc="lower left")
axes[1].set(
    xlabel="Additional training interactions (millions)",
    ylabel="Worst primary scene: successes / 25",
    ylim=(-1, 26),
)
axes[1].axhline(23, color="black", linestyle=":", linewidth=1, label="Required: 23/25")
axes[1].legend(loc="lower left")
for axis in axes:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2)
fig.text(
    0.5,
    -0.12,
    "Both conditions must hold at every check. Both complete windows failed.\n"
    "Different first-pass checkpoints: 64 worlds at 5.77M; 256 worlds at 3.15M.\n"
    "One development seed; this is not an isolated causal comparison of batch size.",
    ha="center",
    fontsize=9,
)
fig.savefig(HERE / "stability.png", dpi=180, bbox_inches="tight")
plt.close(fig)
