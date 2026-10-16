"""Measure terminal-truncated S03 paths without any new physical sampling."""

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[3]


def cylinder_distances(position, centers, sizes):
    """Use the same finite-cylinder signed distance as simulation.scene._distance."""
    local = position[..., None, :] - centers
    q = np.stack(
        (np.linalg.norm(local[..., :2], axis=-1) - sizes[:, 0], abs(local[..., 2]) - sizes[:, 1]),
        axis=-1,
    )
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(axis=-1), 0)


def main():
    """Validate event identity and write all-case measurements and the focused pair."""
    assert np.allclose(
        cylinder_distances(
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 4.0]]),
            np.zeros((1, 3)),
            np.array([[1.0, 3.0]]),
        ),
        [[-1], [1], [1]],
    )
    comparison = json.loads((OUTPUT / "comparison.json").read_text())
    assert comparison["status"] == "completed"
    assert all(r["case_events_match"] for r in comparison["runs"].values())
    bodies = ET.parse(ROOT / "assets/scenes/navigation/S03.xml").findall("worldbody/body")
    assert all(b.find("geom").get("type") == "cylinder" for b in bodies)
    centers = np.array([list(map(float, b.get("pos").split())) for b in bodies])
    sizes = np.array([list(map(float, b.find("geom").get("size").split())) for b in bodies])
    names = [b.get("name") for b in bodies]
    result = {
        "scope": comparison["scope"],
        "focused_world_index": 4,
        "runs": {},
        "limits": "Policy-step samples omit intervening physics ticks; native CSV minimum "
        "clearance and terminal events remain authoritative. Commands are applied "
        "roll/pitch/yaw/thrust, not normalized Actor actions.",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    initial = []
    event_names = {1: "COLLISION", 3: "OUT_OF_BOUNDS", 4: "SUCCESS"}
    for label, color in (("control", "tab:blue"), ("full_batch", "tab:orange")):
        directory = OUTPUT / label
        with (directory / "episodes.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        with np.load(directory / "trajectories.npz", allow_pickle=False) as archive:
            data = {k.removeprefix("S03__0000__"): archive[k] for k in archive.files}
        initial.append(data["position"][0])
        cases = []
        for world, row in enumerate(rows):
            assert int(row["world_index"]) == world
            end = np.flatnonzero(data["event"][:, world])[0] + 1
            t, p, v, cmd = (
                data[k][:end, world] for k in ("time", "position", "velocity", "command")
            )
            assert np.all(np.diff(t) > 0) and np.isfinite(p).all()
            assert event_names[int(data["event"][end - 1, world])] == row["event"]
            assert np.isclose(t[-1], float(row["flight_seconds"]), atol=1e-4)
            goal = np.asarray(comparison["runs"][label]["goal_metres"])
            distance = np.linalg.norm(p - goal, axis=-1)
            clearance = cylinder_distances(p, centers, sizes).min(axis=-1) - 0.07
            nearest = cylinder_distances(p[-1], centers, sizes).argmin()
            cross = np.flatnonzero(p[:, 0] >= 15.4)
            at_x = int(cross[0]) if cross.size else end - 1
            case = {
                "world_index": world,
                "event": row["event"],
                "flight_seconds": float(t[-1]),
                "final_position_metres": p[-1].tolist(),
                "final_goal_distance_metres": float(distance[-1]),
                "normalized_final_goal_progress": float(1 - distance[-1] / distance[0]),
                "height_range_metres": [float(p[:, 2].min()), float(p[:, 2].max())],
                "nearest_terminal_cylinder": names[nearest],
                "sampled_minimum_obstacle_clearance_metres": float(clearance.min()),
                "native_minimum_clearance_metres": float(row["min_clearance"]),
                "first_x_at_least_15_4": {
                    "time": float(t[at_x]),
                    "position": p[at_x].tolist(),
                    "velocity": v[at_x].tolist(),
                },
                "path_length_metres": float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum()),
            }
            cases.append(case)
            if world != 4:
                continue
            axes[0, 0].plot(p[:, 0], p[:, 1], color=color, label=label)
            axes[0, 0].scatter(*p[-1, :2], color=color, marker="x")
            axes[0, 1].plot(t, p[:, 2], color=color, label=label)
            axes[0, 2].plot(t, clearance, color=color, label=label)
            axes[1, 0].plot(t, distance, color=color, label=label)
            axes[1, 1].plot(t, v[:, 1], color=color, label=label)
            axes[1, 2].plot(t, cmd[:, 0], color=color, label=label)
        result["runs"][label] = cases
    assert np.array_equal(*initial)
    for center, size in zip(centers, sizes, strict=True):
        axes[0, 0].add_patch(Circle(center[:2], size[0] + 0.07, fc="0.88", ec="0.6", lw=0.5))
    axes[0, 0].set(
        xlim=(10, 19),
        ylim=(-3, 3),
        aspect="equal",
        xlabel="x (m)",
        ylabel="y (m)",
        title="World 4: first failed obstacle",
    )
    axes[0, 0].legend()
    for ax, title, unit in zip(
        axes.flat,
        [
            "",
            "Height",
            "Obstacle clearance",
            "Goal distance",
            "Lateral velocity",
            "Applied roll command",
        ],
        ["", "z (m)", "m", "m", "m/s", "rad"],
        strict=True,
    ):
        if ax is axes[0, 0]:
            continue
        ax.set(xlim=(0, 18), xlabel="Physical time (s)", ylabel=unit, title=title)
        ax.grid(alpha=0.2)
    axes[0, 1].axhline(6, color="black", ls="--", lw=0.8)
    axes[0, 2].axhline(0, color="black", ls="--", lw=0.8)
    fig.savefig(OUTPUT / "focused-paths.png", dpi=170)
    plt.close(fig)
    (OUTPUT / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                label: [r for r in cases if r["world_index"] == 4]
                for label, cases in result["runs"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
