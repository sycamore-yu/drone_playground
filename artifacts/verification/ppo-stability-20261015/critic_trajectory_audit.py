"""Recompute frozen-policy event and trajectory summaries from existing raw arrays."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np

from drone_playground.simulation.tasks import Event

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root = Path(__file__).resolve().parent / "depth-critic-calibration"
report = json.loads((root / "report.json").read_text())
initial = json.loads((root / "initial-state-audit.json").read_text())
assert initial["initial_state_sha256"] == report["initial_state_sha256"]
summary = {
    "scope": "Existing stochastic training-state trajectories, no new actions or fitting. "
    "State5 is illustrated because it had the largest observed paired mean-return decline; "
    "all eight states remain reported. This is not a unique-cause claim.",
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "models": [],
}
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
colors = ("#2563eb", "#d97706")
for model, color in zip(report["models"], colors, strict=True):
    path = root / f"{model['name']}.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == model["raw_sha256"]
    with np.load(path) as raw:
        assert np.allclose(
            raw["reward"].sum(axis=0, dtype=np.float64).reshape(8, 16),
            raw["returns"],
            atol=1e-10,
            rtol=0,
        )
        states = []
        for group in range(8):
            block = slice(group * 16, (group + 1) * 16)
            active = raw["active"][:, block]
            position = raw["position"][:, block]
            times = raw["time"][:, block]
            assert np.isfinite(position).all() and np.isfinite(times).all()
            events = [Event(int(event)).name for event in raw["event"][-1, block]]
            assert events == model["states"][group]["events"]
            start = np.array(initial["states"][group]["position"])
            failures = np.array([event == "COLLISION" for event in events])
            states.append(
                {
                    "state": group,
                    "events": dict(Counter(events)),
                    "flight_seconds": times.max(axis=0).tolist(),
                    "collision_median_seconds": float(np.median(times.max(axis=0)[failures]))
                    if failures.any()
                    else None,
                    "mean_displacement_at_2_seconds_m": (position[19] - start)
                    .mean(axis=0)
                    .tolist(),
                    "active_at_2_seconds": int(active[19].sum()),
                    "terminal_positions_m": position[-1].tolist(),
                }
            )
            if group == 5:
                assert active[:20].all(), "State5's two-second comparison requires active replicas"
                for replica in range(16):
                    mask = active[:, replica]
                    trace = np.vstack((start, position[mask, replica]))
                    axes[0].plot(trace[:, 0], trace[:, 1], color=color, alpha=0.3, lw=1)
                    axes[1].plot(
                        times[mask, replica],
                        position[mask, replica, 2],
                        color=color,
                        alpha=0.3,
                        lw=1,
                    )
                    if events[replica] != "SUCCESS":
                        axes[0].scatter(*trace[-1, :2], marker="x", color=color, s=18)
        summary["models"].append(
            {"name": model["name"], "raw_sha256": model["raw_sha256"], "states": states}
        )
    x = np.arange(8) + (-0.12 if model["name"] == "u300" else 0.12)
    mc = np.array([state["mean_return"] for state in model["states"]])
    ci = np.array([state["descriptive_bootstrap_95_interval"] for state in model["states"]])
    axes[2].errorbar(
        x, mc, yerr=np.stack((mc - ci[:, 0], ci[:, 1] - mc)), fmt="o", color=color, capsize=2
    )
    axes[2].scatter(x, [state["prediction"] for state in model["states"]], marker="x", color=color)
start, goal = (np.array(initial["states"][5][key]) for key in ("position", "goal"))
axes[0].scatter(*start[:2], c="black", marker="s", s=35)
axes[0].scatter(*goal[:2], c="black", marker="*", s=90)
axes[0].set(title="State 5: 16 trajectories per policy", xlabel="World x (m)", ylabel="World y (m)")
axes[1].axhline(0.5, color="black", ls=":", lw=1)
axes[1].set(xlim=(0, 10), title="State 5: first 10 seconds", xlabel="Time (s)", ylabel="Height (m)")
axes[2].set(
    title="All 8 states: MC return (o), critic (x)", xlabel="Training state", ylabel="Return"
)
axes[2].set_xticks(range(8))
axes[2].axvspan(4.65, 5.35, color="gray", alpha=0.1)
axes[0].legend(
    handles=[
        Line2D([0], [0], color=c, label=m["name"])
        for m, c in zip(report["models"], colors, strict=True)
    ],
    loc="upper left",
)
fig.savefig(root / "state5-regression.png", dpi=180)
plt.close(fig)
(root / "trajectory-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
for model in summary["models"]:
    print(
        model["name"],
        model["states"][5]["events"],
        "median collision seconds",
        model["states"][5]["collision_median_seconds"],
    )
