"""Export the accepted Navigation8 catalog as inspectable RScope replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from drone_playground.runs.navigation_scene import (
    active_indices,
    create_replay_model,
    obstacle_track,
)
from drone_playground.runs.rscope_io import export_rollout
from drone_playground.tasks.scenes.fixed_navigation import (
    build_fixed_bank,
    load_fixed_catalog,
    scene_by_id,
    validate_fixed_catalog,
)
from drone_playground.tasks.scenes.navigation import clearance_and_collision


def interpolate_path(points, frames):
    points = np.asarray(points, np.float32)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    distance = np.linspace(0.0, cumulative[-1], frames)
    output = []
    for value in distance:
        index = min(np.searchsorted(cumulative, value, side="right") - 1, len(lengths) - 1)
        alpha = (value - cumulative[index]) / lengths[index] if lengths[index] else 0.0
        output.append(points[index] + alpha * (points[index + 1] - points[index]))
    return np.asarray(output, np.float32)


def review_env(bank, scene_id, dt, manifest):
    return SimpleNamespace(
        bank=bank,
        dt=dt,
        component_identity={
            "purpose": "Navigation8 visual verification",
            "scene_id": scene_id,
            "inspection_path_is_policy_input": False,
            "catalog_version": manifest["version"],
        },
        scenario=lambda scenario_id: {
            "scene_id": scene_id,
            "difficulty": bank.labels(int(scenario_id))["difficulty"],
            "purpose": "accepted Navigation8 benchmark scene",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/scene/navigation8.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/navigation8-review"),
    )
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    catalog = load_fixed_catalog(args.catalog)
    review = {row["scene_id"]: row for row in validate_fixed_catalog(catalog)}
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    for scene in catalog["scenes"]:
        scene_id = scene["id"]
        bank, manifest = build_fixed_bank(catalog, [scene_id], validated_reports=review)
        duration = float(scene.get("review_duration_s", scene.get("inspection_duration_s", 40.0)))
        frames = max(2, int(round(duration * args.fps)) + 1)
        times = np.linspace(0.0, duration, frames, dtype=np.float32)
        if "inspection_path" in scene:
            positions = interpolate_path(scene["inspection_path"], frames)
            replay_motion = "legacy inspection route"
        else:
            start = np.asarray(catalog["world"]["start"], np.float32)
            positions = np.repeat(start[None, :], frames, axis=0)
            replay_motion = "drone held at start; no reference or oracle route"
        quaternions = np.zeros((frames, 1, 4), np.float32)
        quaternions[..., 3] = 1.0
        positions_batched = positions[:, None, :]
        active = active_indices(bank, 0)
        obstacle_positions = obstacle_track(bank, 0, times)[:, active][:, None]
        clearance = []
        collision = []
        for position, time in zip(positions, times, strict=True):
            value, hit = clearance_and_collision(
                bank, jnp.int32(0), jnp.float32(time), jnp.asarray(position)
            )
            clearance.append(float(value))
            collision.append(float(hit))
        metric_prefix = "inspection" if "inspection_path" in scene else "review/start"
        trace = {
            "pos": positions_batched,
            "quat": quaternions,
            "time": times[:, None],
            "obs": np.zeros((frames, 1, 1), np.float32),
            "reward": np.zeros((frames, 1), np.float32),
            "actions": np.zeros((frames, 1, 4), np.float32),
            "metrics": {
                f"{metric_prefix}/clearance_m": np.asarray(clearance, np.float32)[:, None],
                f"{metric_prefix}/collision": np.asarray(collision, np.float32)[:, None],
            },
            "obstacle_pos": obstacle_positions,
        }
        env = review_env(bank, scene_id, 1.0 / args.fps, manifest)
        env.component_identity["review_drone_motion"] = replay_motion
        env.component_identity["reference_route"] = (
            "none" if "inspection_path" not in scene else "legacy-inspection-only"
        )
        target = args.output / scene_id
        replay = export_rollout(create_replay_model(env, 0), target, trace)
        (target / "scene-definition.json").write_text(
            json.dumps(
                {
                    "catalog_version": catalog["version"],
                    "boundary_obstacles": catalog.get("boundary_obstacles", []),
                    "scene": scene_by_id(catalog, scene_id),
                    "review": review[scene_id],
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        index.append(
            {
                **review[scene_id],
                "title": scene["title"],
                "source": scene["source"],
                "replay": str(replay.relative_to(args.output)),
            }
        )

    (args.output / "index.json").write_text(
        json.dumps(
            {
                "catalog": catalog.get("name", "navigation8"),
                "catalog_version": catalog["version"],
                "status": catalog["status"],
                "scenes": index,
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Navigation8 scene review",
        "",
        "These are the accepted Navigation8 fixed scenes. Route-free catalogs hold the drone at the",
        "start pose and animate only scene dynamics: no reference/oracle trajectory is exported.",
        "",
        "| ID | type | difficulty | field | boundary | total | direct blocked | A* reachable | max straight run | reference |",
        "|---|---|---|---:|---:|---:|---|---|---:|---|",
    ]
    for row in index:
        topology = row.get("topology", {})
        reachable = topology.get("all_snapshots_reachable")
        longest = topology.get("max_clear_straight_run_m")
        lines.append(
            f"| {row['scene_id']} | {'dynamic' if row['dynamic'] else 'static'} | "
            f"{row['difficulty']} | {row['field_obstacles']} | {row['boundary_obstacles']} | "
            f"{row['obstacles']} | "
            f"{'yes' if row['straight_line_blocked'] else 'no'} | "
            f"{('yes' if reachable else 'no') if reachable is not None else 'legacy'} | "
            f"{(f'{longest:.1f} m' if longest is not None else 'legacy')} | {row['source']} |"
        )
    (args.output / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(args.output), "scenes": index}, indent=2))


if __name__ == "__main__":
    main()
