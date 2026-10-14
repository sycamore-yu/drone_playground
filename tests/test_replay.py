"""Round-trip real RScope records and MuJoCo geometry, without a display or GPU."""

import hashlib
import pickle
from pathlib import Path

import mujoco
import numpy as np
import pytest
from rscope import config, model_loader, rollout

from drone_playground.simulation.replay import export_replay

ASSETS = Path(__file__).resolve().parents[1] / "assets/scenes"
TIMES = np.array([0.0, 0.125, 0.4])
POSITIONS = np.array([[2, 0, 3], [2.2, 0.1, 3.1], [2.5, 0.3, 3.0]])
QUATERNIONS = np.array([[0, 0, 0, 1], [0, 0, np.sin(0.2), np.cos(0.2)], [0, 0, 1, 0]])


def decode(path, monkeypatch):
    """Use exactly the upstream APIs used by the native viewer."""
    monkeypatch.setattr(config, "BASE_PATH", path.parent)
    monkeypatch.setattr(config, "META_PATH", path.parent / "rscope_meta.pkl")
    monkeypatch.setattr(rollout, "rollouts", [])
    monkeypatch.setattr(rollout, "num_evals", 0)
    monkeypatch.setattr(rollout, "num_envs", 0)
    monkeypatch.setattr(rollout, "env_ctrl_dt", 0)
    monkeypatch.setattr(rollout, "change_rollout", False)
    rollout.append_unroll(path)
    model, data, meta = model_loader.load_model_and_data()
    return rollout.rollouts[0], model, data, meta


def restore(record, model, data, frame):
    """Provide restore for the surrounding execution."""
    data.qpos[:] = record.qpos[frame, 0]
    data.qvel[:] = record.qvel[frame, 0]
    data.mocap_pos[:] = record.mocap_pos[frame, 0]
    data.mocap_quat[:] = record.mocap_quat[frame, 0]
    data.time = record.time[frame, 0]
    mujoco.mj_forward(model, data)


@pytest.mark.parametrize("scene_name", ["S01", "S02", "S03", "S06", "D01", "D02", "D03", "D06"])
def test_navigation_geometry_and_three_frame_roundtrip(tmp_path, monkeypatch, scene_name):
    """Verify navigation geometry and three frame roundtrip."""
    source = ASSETS / f"navigation/{scene_name}.xml"
    original = mujoco.MjModel.from_xml_path(str(source))
    position_copy, quaternion_copy = POSITIONS.copy(), QUATERNIONS.copy()
    path = export_replay(tmp_path / scene_name, source, TIMES, position_copy, quaternion_copy)
    record, model, data, meta = decode(path, monkeypatch)
    assert record.qpos.shape == (3, 1, 7)
    assert record.qvel.shape == (3, 1, 6)
    assert record.mocap_pos.shape == (3, 1, original.nmocap, 3)
    np.testing.assert_array_equal(record.time[:, 0], TIMES)
    np.testing.assert_array_equal(record.qpos[:, 0, :3], POSITIONS)
    np.testing.assert_allclose(record.qpos[:, 0, 3:], QUATERNIONS[:, [3, 0, 1, 2]])
    np.testing.assert_array_equal(position_copy, POSITIONS)
    np.testing.assert_array_equal(quaternion_copy, QUATERNIONS)
    assert rollout.env_ctrl_dt == TIMES[1] - TIMES[0]
    for i in range(original.ngeom):
        src = original.geom(i)
        dst = model.geom(src.name)
        for field in ("type", "size", "pos", "quat", "rgba", "contype", "conaffinity"):
            np.testing.assert_allclose(getattr(dst, field), getattr(src, field), atol=1e-12)
    for frame in range(3):
        restore(record, model, data, frame)
        np.testing.assert_allclose(data.body("drone").xpos, POSITIONS[frame])
    for name, content in meta["model_assets"].items():
        assert meta["replay"]["asset_sha256"][name] == hashlib.sha256(content).hexdigest()
        assert (path.parent / name).read_bytes() == content


@pytest.mark.parametrize("scene_name,kind", [("D01", 1), ("D06", 2)])
def test_motion_at_recorded_time_not_frame_index(tmp_path, monkeypatch, scene_name, kind):
    """Verify motion at recorded time not frame index."""
    source = ASSETS / f"navigation/{scene_name}.xml"
    path = export_replay(tmp_path / scene_name, source, TIMES + 2.3, POSITIONS, QUATERNIONS)
    record, model, data, _ = decode(path, monkeypatch)
    bodies = np.flatnonzero(model.body_user[:, 0] == kind)
    assert len(bodies)
    body = int(bodies[0])
    _, sx, sy, sz, fourth, fifth = model.body_user[body]
    for frame, time in enumerate(TIMES + 2.3):
        if kind == 1:
            phase = 2 * time / fifth + fourth
            offset = [
                sx * (np.sin(phase) + 2 * np.sin(2 * phase)) / 6,
                sy * (np.cos(phase) - 2 * np.cos(2 * phase)) / 5,
                -sz * np.sin(3 * phase) / 2,
            ]
        else:
            phase = (time / fourth + fifth + 0.5) % 1 - 0.5
            offset = np.array([sx, sy, sz]) * (4 * abs(phase) - 1)
        restore(record, model, data, frame)
        np.testing.assert_allclose(data.xpos[body], model.body_pos[body] + offset, atol=1e-12)
    assert not np.allclose(record.mocap_pos[0], record.mocap_pos[-1])


def test_actual_sensor_hits_and_planner_points_reach_render_scene(tmp_path, monkeypatch):
    """Verify actual sensor hits and planner points reach render scene."""
    source = ASSETS / "navigation/S01.xml"
    source_model = mujoco.MjModel.from_xml_path(str(source))
    source_data = mujoco.MjData(source_model)
    mujoco.mj_forward(source_model, source_data)
    measurements = []
    for origin in POSITIONS:
        hits = []
        for direction in np.array([[-1.0, 0, 0], [0, -1.0, 0]]):
            distance = mujoco.mj_ray(
                source_model,
                source_data,
                origin,
                direction,
                None,
                True,
                -1,
                np.array([-1], dtype=np.int32),
            )
            assert distance > 0
            hits.append(origin + distance * direction)
        measurements.append(
            {"points_world": np.array([*hits, [np.nan] * 3]), "mask": np.array([True, True, False])}
        )
    measurements[1]["mask"][1] = False
    plans = [np.array([[4, 1, 3], [5, 2, 3]]), None, np.array([[7, 3, 3]])]
    path = export_replay(
        tmp_path / "points", source, TIMES, POSITIONS, QUATERNIONS, measurements, plans
    )
    record, model, data, _ = decode(path, monkeypatch)
    np.testing.assert_array_equal(record.metrics["sensor_hits"][:, 0], [2, 1, 2])
    np.testing.assert_array_equal(record.metrics["plan_points"][:, 0], [2, 0, 1])
    render_scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 10)
    for frame in range(3):
        restore(record, model, data, frame)
        points = measurements[frame]["points_world"][measurements[frame]["mask"]]
        for i, point in enumerate(points):
            np.testing.assert_allclose(data.body(f"replay_hit_{i}").xpos, point)
        if plans[frame] is not None:
            for i, point in enumerate(plans[frame]):
                np.testing.assert_allclose(data.body(f"replay_plan_{i}").xpos, point)
        else:
            assert data.body("replay_plan_0").xpos[2] == -1e6
        mujoco.mjv_updateScene(
            model,
            data,
            mujoco.MjvOption(),
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL,
            render_scene,
        )
        geom_id = model.geom("replay_hit_geom_0").id
        rendered = [
            g
            for g in render_scene.geoms[: render_scene.ngeom]
            if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == geom_id
        ]
        assert len(rendered) == 1
        np.testing.assert_allclose(rendered[0].pos, points[0], atol=1e-6)
        assert rendered[0].rgba[3] == 1
        if plans[frame] is not None:
            geom_id = model.geom("replay_plan_geom_0").id
            rendered_plan = [
                g
                for g in render_scene.geoms[: render_scene.ngeom]
                if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid == geom_id
            ]
            assert len(rendered_plan) == 1
            np.testing.assert_allclose(rendered_plan[0].pos, plans[frame][0])


def test_crazyflie_visual_meshes_follow_recorded_orientation(tmp_path, monkeypatch):
    """Replay uses the packaged robot, not an approximation that replaces its appearance."""
    path = export_replay(
        tmp_path / "robot", ASSETS / "navigation/S01.xml", TIMES, POSITIONS, QUATERNIONS
    )
    record, model, data, meta = decode(path, monkeypatch)
    drone_id = model.body("drone").id
    for name in ("cf_pcb", "cf_motors", "cf_prop_0", "cf_prop_3"):
        mesh = model.geom(name)
        assert int(mesh.bodyid[0]) == drone_id
        assert int(mesh.contype[0]) == 0
        assert int(mesh.conaffinity[0]) == 0
    assert int(model.geom("replay_drone").rgba[3] == 0)
    assert any(name.endswith(".stl") for name in meta["model_assets"])
    assert "crazyflie_LICENSE" in meta["model_assets"]
    assert "crazyflie_SOURCE.txt" in meta["model_assets"]
    restore(record, model, data, 0)
    body_rotation = data.body("drone").xmat.reshape(3, 3)
    mesh_rotation = data.geom("cf_pcb").xmat.reshape(3, 3)
    mesh_offset = body_rotation.T @ mesh_rotation
    render_scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 200)
    for index in range(len(TIMES)):
        restore(record, model, data, index)
        np.testing.assert_allclose(data.body("drone").xpos, POSITIONS[index], atol=1e-6)
        np.testing.assert_allclose(
            data.body("drone").xquat, QUATERNIONS[index, [3, 0, 1, 2]], atol=1e-6
        )
        np.testing.assert_allclose(
            data.geom("cf_pcb").xmat.reshape(3, 3),
            data.body("drone").xmat.reshape(3, 3) @ mesh_offset,
            atol=1e-6,
        )
        mujoco.mjv_updateScene(
            model,
            data,
            mujoco.MjvOption(),
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL,
            render_scene,
        )
        assert any(
            geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid == model.geom("cf_pcb").id
            for geom in render_scene.geoms[: render_scene.ngeom]
        )


def test_planner_segments_and_acquisition_times_are_causal(tmp_path, monkeypatch):
    """Show connected plan segments only between their recorded validity times."""
    positions = np.array([[4.0, 1, 3], [5.0, 2, 3], [6.0, 2, 3]])
    plans = [
        {"positions": positions, "received_time": 0.0, "valid_until": 0.2},
        {"positions": positions, "received_time": 0.0, "valid_until": 0.2},
        {"positions": positions, "received_time": 0.0, "valid_until": 0.2},
    ]
    sensor_times = np.array([0.0, 0.1, 0.3])
    path = export_replay(
        tmp_path / "causal",
        ASSETS / "navigation/S01.xml",
        TIMES,
        POSITIONS,
        QUATERNIONS,
        plans=plans,
        sensor_times=sensor_times,
    )
    record, model, data, _ = decode(path, monkeypatch)
    np.testing.assert_array_equal(record.metrics["plan_segments"][:, 0], [2, 2, 0])
    np.testing.assert_array_equal(record.metrics["sensor_acquisition_time"][:, 0], sensor_times)
    assert int(model.tendon("replay_plan_line_0").id) >= 0
    scene = mujoco.MjvScene(model, maxgeom=model.ngeom + 200)
    for frame in (0, 1):
        restore(record, model, data, frame)
        np.testing.assert_allclose(data.body("replay_plan_segment_0_start").xpos, positions[0])
        np.testing.assert_allclose(data.body("replay_plan_segment_0_end").xpos, positions[1])
        np.testing.assert_allclose(data.body("replay_plan_segment_1_start").xpos, positions[1])
        np.testing.assert_allclose(data.body("replay_plan_segment_1_end").xpos, positions[2])
        mujoco.mjv_updateScene(
            model,
            data,
            mujoco.MjvOption(),
            None,
            mujoco.MjvCamera(),
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        lines = [g for g in scene.geoms[: scene.ngeom] if g.objtype == mujoco.mjtObj.mjOBJ_TENDON]
        assert len(lines) == 2
        np.testing.assert_allclose(lines[0].pos, (positions[0] + positions[1]) / 2)
    restore(record, model, data, 2)
    assert data.body("replay_plan_segment_0_start").xpos[2] == -1e6
    assert data.body("replay_plan_0").xpos[2] == -1e6


def test_replay_rejects_uncausal_time_metadata(tmp_path):
    """Plan and sensor timing violations must not silently make a believable replay."""
    scene = ASSETS / "navigation/S01.xml"
    with pytest.raises(ValueError, match="Sensor acquisition"):
        export_replay(
            tmp_path / "future-sensor",
            scene,
            TIMES,
            POSITIONS,
            QUATERNIONS,
            sensor_times=[0.0, 0.2, 0.3],
        )
    with pytest.raises(ValueError, match="Plan reception"):
        export_replay(
            tmp_path / "bad-plan",
            scene,
            TIMES,
            POSITIONS,
            QUATERNIONS,
            plans=[{"positions": [[1, 2, 3]], "received_time": 2, "valid_until": 1}, None, None],
        )


def test_lsy_embeds_textures_and_preserves_gate_orientations(tmp_path, monkeypatch):
    """Verify lsy embeds textures and preserves gate orientations."""
    source = ASSETS / "racing/lsy_level0.xml"
    original = mujoco.MjModel.from_xml_path(str(source))
    path = export_replay(tmp_path / "racing", source, TIMES, POSITIONS, QUATERNIONS)
    # Model loader must succeed from embedded resources even when copied files disappear.
    for file in path.parent.iterdir():
        if file.suffix in (".xml", ".png"):
            file.unlink()
    record, model, data, meta = decode(path, monkeypatch)
    assert model.ntex == original.ntex == 4
    assert len([name for name in meta["model_assets"] if name.endswith(".png")]) == 4
    for frame in range(3):
        restore(record, model, data, frame)
        for gate in range(4):
            name = f"gate:{gate}"
            np.testing.assert_allclose(data.body(name).xpos, original.body(name).pos)
            np.testing.assert_allclose(data.body(name).xquat, original.body(name).quat)


def test_scene_object_empty_and_live_state_are_not_mutated(tmp_path, monkeypatch):
    """Verify scene object empty and live state are not mutated."""
    from drone_playground.simulation.scene import Scene

    for name in ("empty", "D01"):
        scene = Scene(name)
        before = scene.model.body_pos.copy()
        expected = np.asarray(scene.positions(TIMES))
        path = export_replay(tmp_path / name, scene, TIMES, POSITIONS, QUATERNIONS)
        record, model, data, meta = decode(path, monkeypatch)
        assert meta["replay"]["geometry_hash"] == scene.geometry_hash
        for frame in range(3):
            restore(record, model, data, frame)
            for i, geom_name in enumerate(scene.geom_names):
                np.testing.assert_allclose(data.geom(geom_name).xpos, expected[frame, i], atol=1e-5)
        np.testing.assert_array_equal(scene.model.body_pos, before)


def test_included_mesh_is_portable(tmp_path, monkeypatch):
    """Verify included mesh is portable."""
    source = tmp_path / "source"
    (source / "meshes").mkdir(parents=True)
    (source / "meshes/tetra.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
    )
    (source / "shapes.xml").write_text(
        '<mujocoinclude><asset><mesh name="tetra" file="tetra.obj"/></asset>'
        '<worldbody><geom name="mesh_geom" type="mesh" mesh="tetra" pos="5 2 1"/>'
        "</worldbody></mujocoinclude>"
    )
    xml = source / "model.xml"
    xml.write_text('<mujoco><compiler meshdir="meshes"/><include file="shapes.xml"/></mujoco>')
    original = mujoco.MjModel.from_xml_path(str(xml))
    path = export_replay(tmp_path / "portable", xml, TIMES, POSITIONS, QUATERNIONS)
    (source / "meshes/tetra.obj").unlink()
    _, model, _, meta = decode(path, monkeypatch)
    source_mesh, replay_mesh = original.mesh("tetra"), model.mesh("tetra")
    start, size = int(replay_mesh.faceadr[0]), int(replay_mesh.facenum[0])
    np.testing.assert_array_equal(
        model.mesh_face[start : start + size],
        original.mesh_face[int(source_mesh.faceadr[0]) : int(source_mesh.faceadr[0]) + size],
    )
    start, size = int(replay_mesh.vertadr[0]), int(replay_mesh.vertnum[0])
    np.testing.assert_allclose(
        model.mesh_vert[start : start + size],
        original.mesh_vert[int(source_mesh.vertadr[0]) : int(source_mesh.vertadr[0]) + size],
    )
    assert len([k for k in meta["model_assets"] if k.endswith(".obj")]) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("times", [0, 0, 1]),
        ("times", [0, np.nan, 1]),
        ("times", [0]),
        ("positions", np.zeros((2, 3))),
        ("positions", np.full((3, 3), np.inf)),
        ("quaternions_xyzw", np.zeros((3, 4))),
        ("measurements", [np.zeros((1, 3))]),
        ("measurements", [np.ones((1, 2))] * 3),
        ("measurements", [np.full((1, 3), np.nan)] * 3),
        ("measurements", [{"points_world": [[1, 2, 3]], "mask": [1]}] * 3),
    ],
)
def test_invalid_records_are_rejected_before_writing(tmp_path, field, value):
    """Verify invalid records are rejected before writing."""
    kwargs = dict(times=TIMES, positions=POSITIONS, quaternions_xyzw=QUATERNIONS)
    kwargs[field] = value
    destination = tmp_path / "invalid"
    with pytest.raises(ValueError):
        export_replay(destination, ASSETS / "navigation/S01.xml", **kwargs)
    assert not destination.exists()


def test_existing_records_are_never_overwritten(tmp_path):
    """Verify existing records are never overwritten."""
    source = ASSETS / "navigation/S01.xml"
    path = export_replay(tmp_path / "episode", source, TIMES, POSITIONS, QUATERNIONS)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        export_replay(path.parent, source, TIMES, POSITIONS + 1, QUATERNIONS)
    assert path.read_bytes() == before
    with path.open("rb") as stream:
        assert isinstance(pickle.load(stream), rollout.Rollout)
