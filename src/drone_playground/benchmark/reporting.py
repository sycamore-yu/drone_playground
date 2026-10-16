"""Write complete episode reports and replay examples."""

import csv
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from drone_playground.simulation.browser_replay import export as export_browser_replay
from drone_playground.simulation.records import atomic_json
from drone_playground.simulation.replay import export_replay
from drone_playground.simulation.scene import SceneBatch


def write_episodes(path: Path, episodes: list[dict]) -> None:
    """Write all outcomes, including failures, with an explicit stable denominator."""
    if not episodes:
        raise ValueError("Cannot report an empty evaluation as an experiment")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)


def save_evaluation(
    directory: Path,
    env,
    episodes,
    traces,
    *,
    report: dict,
    replay_episodes: int = 2,
    plans=None,
    name="",
    write_tables=True,
    record_trajectories=True,
    replay_directory="replays",
) -> dict:
    """Persist the selected report, actual trajectories and success/failure examples."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if write_tables:
        write_episodes(directory / "episodes.csv", episodes)
    if record_trajectories and traces:
        np.savez_compressed(directory / "trajectories.npz", **traces)
    report = dict(report)
    report.update(
        task=env.task.name,
        task_contract=env.task.contract,
        scene=env.scene.name,
        geometry=env.scene.geometry_identity,
        method=episodes[0]["method"],
    )
    replays, browser_replays = [], []
    indices = []
    for success in (True, False):
        eligible = [
            i for i, episode in enumerate(episodes) if (episode["event"] == "SUCCESS") == success
        ]
        if eligible:
            indices.append(eligible[0])
    indices += [i for i in range(len(episodes)) if i not in indices]
    for index in indices[:replay_episodes]:
        scene = env.scene.scene_for_world(index) if isinstance(env.scene, SceneBatch) else env.scene
        times = traces["time"][:, index]
        keep = np.r_[True, np.diff(times) > 0]
        positions, quaternions = traces["position"][keep, index], traces["quaternion"][keep, index]
        finite = np.isfinite(positions).all(-1) & np.isfinite(quaternions).all(-1)
        frames = np.flatnonzero(keep)[finite]
        if len(frames) < 2:
            continue
        measurements = [
            {
                "points_world": traces["sensor_points"][frame, index],
                "mask": traces["sensor_mask"][frame, index],
            }
            for frame in frames
        ]
        if plans is not None:
            episode_plans = [plans[frame] for frame in frames]
        elif not episodes[index]["method"].startswith("policy"):
            episode_plans = [
                np.asarray(env.task.target(jnp.asarray(times[frame]) + jnp.linspace(0, 1, 12))[0])
                for frame in frames
            ]
        else:
            episode_plans = None
        output = export_replay(
            directory
            / replay_directory
            / f"{name or env.scene.name}-{episodes[index]['episode']:04d}",
            scene,
            times[frames],
            traces["position"][frames, index],
            traces["quaternion"][frames, index],
            measurements,
            episode_plans,
            sensor_times=(traces["sensor_time"][frames, index] if env.sensor is not None else None),
        )
        replays.append(str(output.relative_to(directory)))
        browser = export_browser_replay(output.parent)
        browser_replays.append(str(browser.relative_to(directory)))
    report["replays"] = replays
    report["browser_replays"] = browser_replays
    if write_tables:
        atomic_json(directory / "report.json", report)
    return report
