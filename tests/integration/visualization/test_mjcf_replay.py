"""Recorded obstacles use the same asset and dynamic poses as policy sensing."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from drone_playground.environments.scenes.catalog import NavigationCatalogScene
from drone_playground.visualization.navigation_scene import create_replay_model, obstacle_track
from drone_playground.visualization.rscope_io import export_rollout


@pytest.mark.parametrize("scene_id", ["S01", "D06"])
def test_replay_attaches_fixed_mjcf_and_preserves_dynamic_positions(scene_id, tmp_path):
    bank, _manifest = NavigationCatalogScene(scene_ids=(scene_id,)).build()
    env = SimpleNamespace(
        bank=bank, dt=0.02, sensor=None, scenario=lambda _: {"scene_id": scene_id}
    )
    replay = create_replay_model(env, 0)
    times = np.array([0.0, 0.02, 0.04], np.float32)
    trace = dict(
        pos=np.broadcast_to(np.asarray(bank.start[0]), (3, 1, 3)),
        quat=np.broadcast_to([0.0, 0.0, 0.0, 1.0], (3, 1, 4)),
        time=times[:, None],
        actions=np.zeros((3, 1, 4)),
        obs=np.zeros((3, 1, 20)),
        reward=np.zeros((3, 1)),
        metrics={},
        obstacle_pos=obstacle_track(bank, 0, times)[:, None],
    )
    for index in range(bank.capacity):
        source = replay.mj_model.geom("obstacle_geom_" + str(index))
        expected = np.asarray(bank.size[0, index]).copy()
        if source.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            expected = np.array([expected[0], expected[1] / 2, 0.0])
        np.testing.assert_allclose(source.size, expected, atol=1e-6)
    path = export_rollout(replay, tmp_path, trace)
    assert path.is_file()
    import pickle

    with path.open("rb") as stream:
        recorded = pickle.load(stream)
    ids = replay.data.core.obstacle_mocap_ids
    np.testing.assert_allclose(
        recorded.mocap_pos[:, 0, ids], trace["obstacle_pos"][:, 0], atol=1e-5
    )


def test_corridor_and_preview_keep_separate_expiry_in_actual_replay(tmp_path):
    from drone_playground.planning.corridors import (
        ConvexPolytope,
        SafeFlightCorridor,
        TrajectoryPreview,
    )
    from drone_playground.references import Trajectory
    from drone_playground.runtime.decision import Decision
    from drone_playground.visualization.layers import layers_from_decisions

    bank, _ = NavigationCatalogScene(scene_ids=("S01",)).build()
    model = create_replay_model(SimpleNamespace(bank=bank, dt=0.02, sensor=None), 0)
    vertices = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 2]])
    corridor = SafeFlightCorridor("candidate", (ConvexPolytope(vertices=vertices),), 0.0, 0.3)
    coefficients = np.zeros((1, 4, 2))
    coefficients[0, 0, 1] = 1.0
    curve = Trajectory(0.0, [1.0], coefficients)
    preview = TrajectoryPreview("backup", curve, 0.0, 0.8)
    layers = layers_from_decisions(
        [
            dict(
                time=0.0,
                reply=Decision(
                    "no_plan",
                    None,
                    "",
                    0.0,
                    0.0,
                    {},
                    corridors=(corridor,),
                    trajectory_previews=(preview,),
                ),
            )
        ]
    )
    assert {frame.layer: frame.valid_until for frame in layers.planning} == {
        "stage-output/sfc/candidate": 0.3,
        "stage-output/preview/backup": 0.8,
    }
    model.replay_visualization = layers
    times = np.array([0.0, 0.4, 0.9], np.float32)
    trace = dict(
        pos=np.broadcast_to(np.asarray(bank.start[0]), (3, 1, 3)),
        quat=np.broadcast_to([0.0, 0.0, 0.0, 1.0], (3, 1, 4)),
        time=times[:, None],
        actions=np.zeros((3, 1, 4)),
        obs=np.zeros((3, 1, 20)),
        reward=np.zeros((3, 1)),
        metrics={},
        obstacle_pos=obstacle_track(bank, 0, times)[:, None],
    )
    result = export_rollout(model, tmp_path, trace)
    assert result.is_file()
    loaded = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    assert loaded.ntendon > 0 and loaded.nmocap > model.mj_model.nmocap
    import pickle

    with result.open("rb") as stream:
        recorded = pickle.load(stream)
    corridor_id = int(loaded.body("viz_plan_0_0_0").mocapid[0])
    preview_id = int(loaded.body("viz_plan_1_0_0").mocapid[0])
    assert recorded.mocap_pos[1, 0, corridor_id, 2] == -10000
    assert recorded.mocap_pos[1, 0, preview_id, 2] != -10000
    assert recorded.mocap_pos[2, 0, preview_id, 2] == -10000
