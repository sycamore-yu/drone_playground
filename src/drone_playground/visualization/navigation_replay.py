"""Export navigation episodes with their actual pre-action state and terminal frame."""

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from drone_playground.artifacts.reporting import save_report
from drone_playground.environments.scenes.geometry import clearance_and_collision


def export_case(task, trace, case, directory):
    """Export a single navigation episode with corresponding scene geometry."""
    from drone_playground.visualization.navigation_scene import (
        active_indices,
        create_replay_model,
        obstacle_track,
    )
    from drone_playground.visualization.rscope_io import export_rollout

    length = int(np.asarray(trace["active"])[:, case].sum())
    single = {
        key: np.asarray(trace[key])[:length, case : case + 1]
        for key in ("pos", "time", "obs", "actions", "reward")
    }
    matrices = np.asarray(trace["rotation"])[:length, case]
    single["quat"] = Rotation.from_matrix(matrices).as_quat()[:, None]
    single["metrics"] = {
        k: np.asarray(v)[:length, case : case + 1] for k, v in trace["metrics"].items()
    }
    # Include the real pre-action state. RScope needs two timestamps even when
    # the first transition terminates; frame count and transition count differ.
    initial_position = np.asarray(trace["observation_pos"])[0, case]
    initial_rotation = np.asarray(trace["observation_rotation"])[0, case]
    initial_centre = initial_position + initial_rotation @ np.array([0.0, 0.0, 0.005])
    initial_clearance = float(
        clearance_and_collision(task.bank, case, 0.0, initial_centre, task.body_radius)[0]
    )
    initial_metrics = {
        "clearance": initial_clearance,
        "goal_distance": float(np.linalg.norm(np.asarray(task.bank.goal[case]) - initial_position)),
        "speed": float(np.linalg.norm(np.asarray(trace["observation_velocity"])[0, case])),
    }
    initial_fields = {
        "pos": initial_position[None, None],
        "quat": Rotation.from_matrix(initial_rotation).as_quat()[None, None],
        "time": np.zeros_like(single["time"][:1]),
        "obs": single["obs"][:1],
        "actions": np.zeros_like(single["actions"][:1]),
        "reward": np.zeros_like(single["reward"][:1]),
    }
    for name, initial in initial_fields.items():
        single[name] = np.concatenate([initial, single[name]], axis=0)
    single["metrics"] = {
        name: np.concatenate([np.full_like(values[:1], initial_metrics[name]), values], axis=0)
        for name, values in single["metrics"].items()
    }
    active = active_indices(task.bank, case)
    single["obstacle_pos"] = obstacle_track(task.bank, case, single["time"][:, 0])[
        :, active, None, :
    ].swapaxes(1, 2)
    model = create_replay_model(task, case)
    directory = Path(directory)
    path = export_rollout(model, directory, single)
    import mujoco
    from rscope import rollout

    rollout.rollouts.clear()
    rollout.num_evals = 0
    rollout.append_unroll(path)
    restored = rollout.rollouts[-1]
    np.testing.assert_allclose(restored.mocap_pos[:, 0, 0], single["pos"][:, 0], atol=1e-6)
    for axis in range(single["actions"].shape[-1]):
        np.testing.assert_allclose(
            restored.metrics[f"action/{axis}"],
            single["actions"][..., axis],
            atol=1e-6,
        )
    loaded_model = mujoco.MjModel.from_xml_path(str(directory / "scene.xml"))
    if loaded_model.nmocap != model.mj_model.nmocap:
        raise ValueError("Replay model lost an obstacle or drone motion body")
    save_report(
        directory / "readback-verification.json",
        dict(
            frames=length + 1,
            transitions=length,
            initial_frame_included=True,
            action_channels=single["actions"].shape[-1],
            positions_and_action_channels_match=True,
            model_xml_recompiled=True,
            motion_bodies=loaded_model.nmocap,
            replay_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
    )
    rollout.rollouts.clear()
    rollout.num_evals = 0
    return path
