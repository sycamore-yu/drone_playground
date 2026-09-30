from types import SimpleNamespace

import mujoco
import numpy as np
import pytest


def replay_sim():
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="drone" mocap="true"><geom type="sphere" size=".07"/></body></worldbody></mujoco>'
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(
        spec=spec,
        mj_model=model,
        data=SimpleNamespace(core=SimpleNamespace(drone_mocap_ids=np.array([0]))),
        mjx_data=SimpleNamespace(
            qpos=data.qpos[None],
            qvel=data.qvel[None],
            mocap_pos=data.mocap_pos[None],
            mocap_quat=data.mocap_quat[None],
        ),
    )


def trace():
    return dict(
        pos=np.array([[[1.0, 2.0, 3.0]]] * 4),
        quat=np.array([[[0.0, 0.0, 0.0, 1.0]]] * 4),
        time=np.array([[0.0], [0.2], [0.6], [1.0]]),
        obs=np.ones((4, 1, 2)),
        reward=np.arange(4)[:, None],
        metrics={"speed": np.ones((4, 1))},
    )


def test_depth_view_uses_real_optical_extrinsics():
    from drone_playground.environments.sensors.depth import DepthCamera
    from drone_playground.visualization.layers import sensor_view

    view = sensor_view(DepthCamera().calibration())
    lines = view.line_segments()
    far_points = lines[np.isclose(lines[..., 0], 10.05).all(axis=1)]
    assert len(far_points) == 4
    np.testing.assert_allclose(far_points.mean(axis=(0, 1)), [10.05, 0.0175, 0.0125], atol=1e-6)
    assert view.far_m == view.display_range_m == 10


def test_depth_flight_view_preserves_upward_pitch_and_camera_range():
    from drone_playground.environments.sensors.depth_flight import DepthFlightCamera
    from drone_playground.visualization.layers import sensor_view

    view = sensor_view(DepthFlightCamera().calibration())
    far_edges = view.line_segments()[-4:]
    np.testing.assert_allclose(
        far_edges.mean(axis=(0, 1)),
        [10 * np.cos(np.deg2rad(20)), 0, 10 * np.sin(np.deg2rad(20))],
        atol=1e-6,
    )


def test_mid360_draws_full_azimuth_and_keeps_display_clip_distinct_from_range():
    from drone_playground.environments.sensors.lidar import Mid360Lidar
    from drone_playground.visualization.layers import sensor_view

    view = sensor_view(Mid360Lidar().calibration())
    assert view.far_m == 200 and view.display_range_m == 10
    points = view.line_segments().reshape(-1, 3)
    assert points[:, 0].min() < -8 and points[:, 0].max() > 8
    assert points[:, 1].min() < -8 and points[:, 1].max() > 8
    assert points[:, 2].min() < 0 and points[:, 2].max() > 7


def test_timed_planning_and_sfc_render_in_standard_rscope_without_changing_physics(tmp_path):
    from rscope import rollout

    from drone_playground.native.contracts import Trajectory
    from drone_playground.native.geometry import ConvexPolytope
    from drone_playground.visualization.layers import PlanningFrame, ReplayLayers
    from drone_playground.visualization.rscope_io import export_rollout

    curve = Trajectory(0.0, [0.8], np.array([[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]))
    cube = ConvexPolytope(
        halfspaces=np.array(
            [
                [1, 0, 0, -1],
                [-1, 0, 0, -1],
                [0, 1, 0, -1],
                [0, -1, 0, -1],
                [0, 0, 1, -1],
                [0, 0, -1, -1],
            ],
            float,
        )
    )
    layers = ReplayLayers(
        planning=[
            PlanningFrame(time=0.1, valid_until=0.8, layer="planning", trajectory=curve),
            PlanningFrame(time=0.1, valid_until=0.8, layer="sfc", polytopes=[cube]),
        ],
        trajectory_samples=8,
    )
    sim = replay_sim()
    original = sim.spec.to_xml()
    original_n = sim.mj_model.nmocap
    values = trace()
    path = export_rollout(sim, tmp_path, values, visualization=layers)
    rollout.rollouts.clear()
    rollout.append_unroll(path)
    restored = rollout.rollouts[-1]
    assert sim.spec.to_xml() == original and sim.mj_model.nmocap == original_n
    np.testing.assert_array_equal(restored.mocap_pos[:, :, :original_n], values["pos"][:, :, None])
    np.testing.assert_array_equal(restored.reward, values["reward"])
    np.testing.assert_array_equal(restored.metrics["speed"], values["metrics"]["speed"])
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    data = mujoco.MjData(model)
    assert model.ntendon == 7 + 12
    assert (restored.mocap_pos[0, 0, original_n:, 2] < -100).all()
    assert (restored.mocap_pos[-1, 0, original_n:, 2] < -100).all()
    np.testing.assert_allclose(restored.mocap_pos[1, 0, original_n], [0.2, 0, 1], atol=1e-6)
    data.mocap_pos[:] = restored.mocap_pos[1, 0]
    data.mocap_quat[:] = restored.mocap_quat[1, 0]
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=1000)
    mujoco.mjv_updateScene(
        model, data, mujoco.MjvOption(), None, mujoco.MjvCamera(), mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    assert scene.ngeom >= model.ntendon
    assert model.nq == sim.mj_model.nq


def test_polytope_rejects_unbounded_or_empty_halfspaces():
    from drone_playground.native.geometry import ConvexPolytope
    from drone_playground.visualization.layers import polytope_edges

    for planes in [
        [[1, 0, 0, -1], [-1, 0, 0, -1], [0, 1, 0, -1], [0, -1, 0, -1]],
        [[1, 0, 0, 1], [-1, 0, 0, 1], [0, 1, 0, -1], [0, -1, 0, -1]],
    ]:
        with pytest.raises(ValueError, match="bounded|interior"):
            polytope_edges(ConvexPolytope(halfspaces=planes))


def test_received_future_trajectory_preview_is_visible_before_execution(tmp_path):
    import pickle

    from drone_playground.native.contracts import Trajectory
    from drone_playground.visualization.layers import PlanningFrame, ReplayLayers
    from drone_playground.visualization.rscope_io import export_rollout

    curve = Trajectory(0.4, [0.4], np.array([[[3.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]))
    layers = ReplayLayers(
        planning=[PlanningFrame(0.1, 0.8, "preview", trajectory=curve)], trajectory_samples=8
    )
    path = export_rollout(replay_sim(), tmp_path, trace(), visualization=layers)
    with path.open("rb") as handle:
        recorded = pickle.load(handle)
    np.testing.assert_allclose(recorded.mocap_pos[1, 0, 1], [3, 0, 1])


def test_sensor_views_appear_in_the_original_viewer_model(tmp_path):
    from drone_playground.environments.sensors.depth_flight import DepthFlightCamera
    from drone_playground.visualization.layers import ReplayLayers, sensor_view
    from drone_playground.visualization.rscope_io import export_rollout

    sim = replay_sim()
    export_rollout(
        sim,
        tmp_path,
        trace(),
        visualization=ReplayLayers(sensor=sensor_view(DepthFlightCamera().calibration())),
    )
    model = mujoco.MjModel.from_xml_path(str(tmp_path / "scene.xml"))
    assert model.ngeom > sim.mj_model.ngeom and model.nmocap == sim.mj_model.nmocap
    ids = [i for i in range(model.ngeom) if model.geom(i).name.startswith("viz_sensor_")]
    assert len(ids) == 12
    assert not model.geom_contype[ids].any() and not model.geom_conaffinity[ids].any()
    assert mujoco.MjvOption().geomgroup[model.geom_group[ids]].all()


def test_sensor_descriptor_rejects_nonphysical_range_and_extrinsics():
    from dataclasses import replace

    from drone_playground.environments.sensors.depth import DepthCamera
    from drone_playground.visualization.layers import sensor_view

    view = sensor_view(DepthCamera().calibration())
    for fields in [
        dict(display_range_m=20),
        dict(rotation=np.zeros((3, 3))),
        dict(angles_deg=(0, 58)),
        dict(translation=np.zeros(4)),
    ]:
        with pytest.raises(ValueError):
            replace(view, **fields)


def test_rpc_geometry_survives_recording_and_does_not_become_an_output(tmp_path):
    from drone_playground.evaluation.decision_archive import (
        NativeDecisionRecorder,
        load_native_decisions,
    )
    from drone_playground.native.proto import algorithm_pb2 as pb
    from drone_playground.native.wire import decode_decision

    response = pb.StepResponse(decision=pb.Decision(status=pb.NO_PLAN))
    geometry = response.planner_geometry
    geometry.frame = "world"
    geometry.generated_at = 0.1
    geometry.valid_until = 0.5
    corridor = geometry.corridors.add(name="candidate")
    corridor.polytopes.add(
        halfspaces=[1, 0, 0, -1, -1, 0, 0, -1, 0, 1, 0, -1, 0, -1, 0, -1, 0, 0, 1, -1, 0, 0, -1, -1]
    )
    decision = decode_decision(
        response,
        0.2,
        pb.Capabilities(algorithm="fixture", outputs=["trajectory"], derivatives="none"),
    )
    assert decision.output is None
    with NativeDecisionRecorder(tmp_path / "decisions", "attitude_thrust") as recorder:
        recorder.record(0, 0.2, {}, dict(planner_geometry=decision.planner_geometry), None)
    row = next(load_native_decisions(tmp_path / "decisions"))
    np.testing.assert_array_equal(
        row["reply"]["planner_geometry"].corridors[0].polytopes[0].halfspaces,
        [
            [1, 0, 0, -1],
            [-1, 0, 0, -1],
            [0, 1, 0, -1],
            [0, -1, 0, -1],
            [0, 0, 1, -1],
            [0, 0, -1, -1],
        ],
    )
    geometry.generated_at = 0.3
    with pytest.raises(ValueError, match="future"):
        decode_decision(
            response,
            0.2,
            pb.Capabilities(algorithm="fixture", outputs=["trajectory"], derivatives="none"),
        )


def test_decision_layers_keep_intermediate_curve_and_clear_it_causally():
    from drone_playground.native.contracts import Trajectory
    from drone_playground.visualization.layers import layers_from_decisions

    curve = Trajectory(0.0, [1.0], np.array([[[0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]))
    reply = dict(stages=[dict(stage=0, output=curve, valid_until=1.0)], output=None)
    rows = [
        dict(time=0.1, reply=reply),
        dict(time=0.2, reply=reply),
        dict(time=0.3, reply=dict(stages=[dict(stage=0, output=None)], output=None)),
    ]
    layers = layers_from_decisions(rows)
    assert len(layers.planning) == 2
    assert layers.planning[0].time == 0.1 and layers.planning[0].trajectory is curve
    assert layers.planning[1].time == 0.3 and layers.planning[1].trajectory is None


def test_shared_planner_layers_cannot_be_drawn_on_multiple_episodes(tmp_path):
    from drone_playground.visualization.layers import PlanningFrame, ReplayLayers
    from drone_playground.visualization.rscope_io import export_rollout

    values = trace()
    values = {
        key: (
            {n: np.repeat(v, 2, axis=1) for n, v in value.items()}
            if isinstance(value, dict)
            else np.repeat(value, 2, axis=1)
        )
        for key, value in values.items()
    }
    values["active"] = np.ones((4, 2), bool)
    with pytest.raises(ValueError, match="one episode"):
        export_rollout(
            replay_sim(),
            tmp_path,
            values,
            visualization=ReplayLayers(planning=[PlanningFrame(0.0, 1.0, "plan")]),
        )


def test_ros_corridor_mesh_uses_marker_pose_and_clears_deleted_geometry():
    from drone_playground.native.ros_visualization import corridor_from_markers

    def xyz(x, y, z):
        return SimpleNamespace(x=x, y=y, z=z)

    marker = SimpleNamespace(
        type=11,
        action=0,
        ns="exp_sfc mesh",
        id=1,
        header=SimpleNamespace(frame_id="world"),
        pose=SimpleNamespace(
            position=xyz(10, 0, 0), orientation=SimpleNamespace(x=0, y=0, z=0, w=1)
        ),
        scale=xyz(2, 1, 1),
        points=[xyz(0, 0, 0), xyz(1, 0, 0), xyz(0, 1, 0), xyz(0, 0, 1)],
    )
    corridor = corridor_from_markers([marker], "candidate")
    assert len(corridor.polytopes) == 1
    points = corridor.polytopes[0].vertices
    assert points[:, 0].min() == 10 and points[:, 0].max() == 12
    marker.action = 3
    assert not corridor_from_markers([marker], "candidate").polytopes
    marker.action = 0
    marker.header.frame_id = "map"
    with pytest.raises(ValueError, match="world"):
        corridor_from_markers([marker], "candidate")


def test_enhancing_a_frozen_replay_preserves_all_original_arrays_and_source(tmp_path):
    import hashlib
    import pickle

    from drone_playground.environments.sensors.depth import DepthCamera
    from drone_playground.visualization.layers import ReplayLayers, sensor_view
    from drone_playground.visualization.rscope_io import enhance_replay, export_rollout

    source = export_rollout(replay_sim(), tmp_path / "original", trace())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = enhance_replay(
        source, tmp_path / "enhanced", ReplayLayers(sensor=sensor_view(DepthCamera().calibration()))
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with source.open("rb") as f:
        before = pickle.load(f)
    with destination.open("rb") as f:
        after = pickle.load(f)
    for field in ("qpos", "qvel", "mocap_pos", "mocap_quat", "obs", "reward", "time"):
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
    assert (tmp_path / "enhanced" / "replay-visualization.json").is_file()
    with pytest.raises(FileExistsError):
        enhance_replay(source, tmp_path / "original", ReplayLayers())
