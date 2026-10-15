"""Summarize recorded S03 paths and compare outcomes with cached screening evidence."""

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent


def main():
    """Write measured stopping locations, original-case consistency and a standalone plot."""
    cylinders = []
    for body in ET.parse(ROOT / "assets/scenes/navigation/S03.xml").findall("worldbody/body"):
        geom = body.find("geom")
        assert geom.attrib["type"] == "cylinder"
        center = list(map(float, body.attrib["pos"].split()))
        radius, half_height = map(float, geom.attrib["size"].split())
        assert center[2] == half_height == 3
        cylinders.append((body.attrib["name"], center[:2], radius))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    result = {
        "scope": "Development trajectories only; no training or formal acceptance",
        "runs": {},
    }
    sources = [
        ("altitude", 112, "depth-fixed-scenes-altitude-pd-progress.json"),
        ("parallel", 48, "depth-fixed-scenes-parallel-progress.json"),
    ]
    for index, (name, update, cache_name) in enumerate(sources):
        directory = OUTPUT / f"{name}_from_state"
        with (directory / "episodes.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        cached = json.loads((OUTPUT.parent / cache_name).read_text())
        original = next(c for c in cached["checks"] if c["update"] == update)
        events = {tuple(r[:4]): r[4] for r in original["case_events"] if r[0] == "S03"}
        actual = {
            tuple(
                r[k] for k in ("scene", "geometry_sha256", "initialization_seed", "world_index")
            ): r["event"]
            for r in rows
        }
        assert actual == events and len(rows) == 8
        with np.load(directory / "trajectories.npz", allow_pickle=False) as data:
            time = data["S03__0000__time"]
            position = data["S03__0000__position"]
            velocity = data["S03__0000__velocity"]
        assert position.shape == velocity.shape == (3001, 8, 3)
        assert np.isfinite(position).all() and np.isfinite(velocity).all()
        assert np.all(np.diff(time, axis=0) > 0) and np.allclose(time[-1], 300)
        cases = []
        for world, row in enumerate(rows):
            p = position[:, world]
            distance = np.linalg.norm(p - [98, 0, 3], axis=-1)
            tail = time[:, world] >= 270
            nearest = min(
                (float(np.linalg.norm(p[-1, :2] - xy) - radius - 0.07), label)
                for label, xy, radius in cylinders
            )
            assert np.isclose(distance[-1], float(row["final_goal_distance_metres"]), atol=1e-4)
            assert np.isclose(distance.min(), float(row["minimum_goal_distance_metres"]), atol=1e-4)
            cases.append(
                {
                    "world_index": world,
                    "event": row["event"],
                    "final_position_metres": p[-1].tolist(),
                    "minimum_goal_distance_metres": float(distance.min()),
                    "final_goal_distance_metres": float(distance[-1]),
                    "last30s_displacement_metres": float(np.linalg.norm(p[tail][-1] - p[tail][0])),
                    "last30s_path_metres": float(
                        np.linalg.norm(np.diff(p[tail], axis=0), axis=-1).sum()
                    ),
                    "last30s_mean_speed_m_s": float(
                        np.linalg.norm(velocity[tail, world], axis=-1).mean()
                    ),
                    "height_range_metres": [float(p[:, 2].min()), float(p[:, 2].max())],
                    "nearest_terminal_cylinder": nearest[1],
                    "terminal_cylinder_clearance_metres": nearest[0],
                }
            )
            axes[index, 0].plot(p[:, 0], p[:, 1], alpha=0.7, lw=1)
            axes[index, 0].plot(*p[-1, :2], marker="x", color="black", ms=5)
            axes[index, 1].plot(time[:, world], distance, alpha=0.7, lw=1)
        for _, xy, radius in cylinders:
            axes[index, 0].add_patch(Circle(xy, radius + 0.07, fc="0.85", ec="0.6", lw=0.5))
        axes[index, 0].set(
            xlim=(0, 100),
            ylim=(-20, 20),
            aspect="equal",
            xlabel="x (m)",
            ylabel="y (m)",
            title=f"{name}, update {update}: eight paths (goal x=98 m)",
        )
        axes[index, 1].set(
            xlim=(0, 300),
            ylim=(0, 100),
            xlabel="Physical time (s)",
            ylabel="Goal distance (m)",
            title="Goal distance; arrival requires <=0.5 m",
        )
        axes[index, 1].axhline(0.5, color="black", ls="--", lw=0.8)
        axes[index, 1].grid(alpha=0.2)
        result["runs"][name] = {
            "checkpoint_update": update,
            "screening_case_events_match": True,
            "cases": cases,
        }
    fig.savefig(OUTPUT / "paths.png", dpi=160)
    fig.savefig(OUTPUT / "paths.svg")
    svg = OUTPUT / "paths.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    (OUTPUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: [
                    r["cases"][0]["final_position_metres"],
                    r["cases"][0]["final_goal_distance_metres"],
                ]
                for name, r in result["runs"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
