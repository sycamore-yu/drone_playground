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
