"""Functional geometry/sensor checks against analytical cases and MuJoCo."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from shutil import copytree

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from jax.scipy.spatial.transform import Rotation

from drone_playground.simulation.scene import Scene
from drone_playground.simulation.sensors import (
    Measurement,
    SensorConfig,
    measure,
    preprocess_depth,
    reduce_points,
    sensor_rays,
)


@pytest.fixture
def make_scene(tmp_path):
    """Provide make scene for the surrounding execution."""

    def build(body, extra=""):
        path = tmp_path / "scene.xml"
        path.write_text(
            f'<mujoco><compiler angle="degree"/>{extra}<worldbody>{body}</worldbody></mujoco>'
        )
        return Scene(path)

    return build


def _mj_ranges(scene, origins, directions, t=0.0):
    """Independent MuJoCo transforms + native ray oracle at an exact time."""
    model = scene.model
    data = mujoco.MjData(model)
    for body in range(model.nbody):
        if model.nuser_body != 6:
            continue
        kind, sx, sy, sz, a, b = model.body_user[body]
        if kind == 1:
            u = 2 * t / b + a
            delta = [
                sx / 6 * (np.sin(u) + 2 * np.sin(2 * u)),
                sy / 5 * (np.cos(u) - 2 * np.cos(2 * u)),
                -sz / 2 * np.sin(3 * u),
            ]
        elif kind == 2:
            u = t / a + b
            delta = np.array([sx, sy, sz]) * (2 * abs(2 * (u - np.floor(u + 0.5))) - 1)
        else:
            continue
        data.mocap_pos[model.body_mocapid[body]] = model.body_pos[body] + delta
    mujoco.mj_forward(model, data)
    hits = []
    for origin, direction in zip(origins, directions, strict=True):
        direction = direction / np.linalg.norm(direction)
        hits.append(
            mujoco.mj_ray(
                model,
                data,
                np.asarray(origin, dtype=np.float64),
                np.asarray(direction, dtype=np.float64),
                None,
                1,
                -1,
                np.zeros(1, dtype=np.int32),
            )
        )
    return np.asarray(hits), data


@pytest.mark.parametrize(
    "name", ["empty", "S01", "S02", "S03", "D01", "D02", "D03", "S06", "D06", "racing"]
)
def test_assets_compile_and_identity(name):
    """Verify assets compile and identity."""
    scene = Scene(name)
    assert isinstance(scene.model, mujoco.MjModel)
    assert scene.positions(jnp.array([0.0, 2.5])).shape == (2, scene.model.ngeom, 3)
    assert len(scene.geom_names) == scene.model.ngeom
    assert len(scene.geometry_hash) == 64
    assert scene.geometry_hash == Scene(name).geometry_hash
    if name != "empty":
        assert scene.xml_path.is_file()
        assert scene.model.ngeom > 0
    if name == "racing":
        assert scene.gate_order == (1, 2, 3, 4, 2)
        np.testing.assert_allclose(scene.gate_positions[0], [0.5, 0.25, 0.7])
        assert scene.gate_rotations.shape == (4, 3, 3)


def test_empty_batches_jit_and_zero_gradient():
    """Verify empty batches jit and zero gradient."""
    scene = Scene("empty")
    pos = jnp.zeros((2, 3))
    assert np.isinf(jax.jit(scene.clearance)(pos, jnp.array([1.0, 2.0]))).all()
    rays = jnp.eye(3)
    np.testing.assert_array_equal(
        jax.jit(scene.raycast)(pos, rays, jnp.array([1.0, 2.0]), 7), np.full((2, 3), 7)
    )
    np.testing.assert_array_equal(
        jax.grad(lambda p: scene.raycast(p, rays).sum())(pos), np.zeros((2, 3))
    )


def test_packaged_resources_take_precedence(monkeypatch, tmp_path):
    """Verify packaged resources take precedence."""
    import drone_playground.simulation.scene as scene_module

    source = Scene("S01")
    package = tmp_path / "drone_playground"
    copytree(Path(source.xml_path).parents[1], package / "assets/scenes")
    monkeypatch.setattr(scene_module.resources, "files", lambda name: package)
    packaged = Scene("S01")
    assert packaged.xml_path == package / "assets/scenes/navigation/S01.xml"
    assert packaged.geometry_hash == source.geometry_hash
    assert Scene("racing").model.ngeom > 0  # Relative texture files resolve too.


@pytest.mark.parametrize(
    "geom,point,clearance,ray,expected",
    [
        ('type="box" size="1 2 3"', [4, 0, 0], 3, [-2, 0, 0], 3),
        ('type="sphere" size="1"', [4, 0, 0], 3, [-1, 0, 0], 3),
        ('type="cylinder" size="1 2"', [4, 0, 0], 3, [-1, 0, 0], 3),
        ('type="cylinder" size="1 2"', [0, 0, 4], 2, [0, 0, -1], 2),
        ('type="capsule" size="1 2"', [0, 0, 4], 1, [0, 0, -1], 1),
        ('type="plane" size="0 0 .1"', [0, 0, 4], 4, [0, 0, -1], 4),
    ],
)
def test_primitive_distance_and_hits(make_scene, geom, point, clearance, ray, expected):
    """Verify primitive distance and hits."""
    scene = make_scene(f"<geom {geom}/>")
    np.testing.assert_allclose(scene.clearance(jnp.array(point)), clearance)
    np.testing.assert_allclose(scene.raycast(jnp.array(point), jnp.array(ray)), expected)
    inside = jnp.array([0.1, 0.1, -0.1])
    assert scene.clearance(inside) < 0
    gradient = jax.grad(scene.clearance)(jnp.array(point, dtype=jnp.float32))
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 0


def test_occlusion_rotated_nested_bodies_and_mujoco(make_scene):
    """Verify occlusion rotated nested bodies and mujoco."""
    scene = make_scene("""
      <body pos="4 0 0" euler="0 0 90">
        <body pos="0 1 0"><geom name="near" type="box" size="2 .5 1"/></body>
      </body>
      <geom name="far" type="sphere" size="1" pos="8 0 0"/>
      <geom name="floor" type="plane" size="0 0 .1" pos="0 0 -2"/>
    """)
    np.testing.assert_allclose(scene.positions()[scene.model.geom("near").id], [3, 0, 0], atol=1e-6)
    origins = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 0]], dtype=float)
    rays = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=float)
    expected, _ = _mj_ranges(scene, origins, rays)
    expected = np.where(expected < 0, 40, expected)
    actual = jax.jit(scene.raycast)(jnp.array(origins), jnp.array(rays[:, None]), jnp.zeros(4))[
        ..., 0
    ]
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_allclose(actual[:2], [2.5, 2], atol=2e-6)
    np.testing.assert_allclose(scene.clearance(jnp.array([0.0, 0, 0])), 2)


@pytest.mark.parametrize("kind", ["box", "sphere", "cylinder", "capsule"])
def test_random_rotations_match_native_rays(make_scene, kind):
    """Verify random rotations match native rays."""
    size = "1.2" if kind == "sphere" else "1.2 2 0.7" if kind == "box" else "1.2 2"
    scene = make_scene(
        f'<body pos="2 -1 3" euler="28 -32 61"><geom type="{kind}" size="{size}"/></body>'
    )
    rng = np.random.default_rng(4)
    origins = rng.normal(size=(259, 3)) * 3 + [2, -1, 3]
    directions = np.array([2, -1, 3]) - origins + rng.normal(size=(259, 3))
    expected, _ = _mj_ranges(scene, origins, directions)
    expected = np.where(expected < 0, 40, expected)
    actual = jax.jit(scene.raycast)(
        jnp.array(origins), jnp.array(directions[:, None]), jnp.zeros(len(origins))
    )[..., 0]
    np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-5)


def test_contact_visibility_and_unsupported_geometry(make_scene):
    """Verify contact visibility and unsupported geometry."""
    scene = make_scene("""
        <geom name="visual" type="box" pos="3 0 0" size=".5 1 1"
              contype="0" conaffinity="0"/>
        <geom name="collision" type="box" pos="6 0 0" size=".5 1 1" rgba="1 1 1 0"/>
    """)
    np.testing.assert_allclose(scene.clearance(jnp.zeros(3)), 5.5)
    np.testing.assert_allclose(scene.raycast(jnp.zeros(3), jnp.array([1.0, 0, 0])), 2.5)
    with pytest.raises(ValueError, match=r"Unsupported geom.*ellipsoid"):
        make_scene('<geom name="ellipsoid" type="ellipsoid" size="1 2 3"/>')
    with pytest.raises(ValueError, match=r"motion"):
        make_scene('<body mocap="true" user="3 1 1 1 1 1"><geom size="1"/></body>')


def test_rotated_plane_front_face_and_rendered_extent(make_scene):
    """Verify rotated plane front face and rendered extent."""
    scene = make_scene('<geom type="plane" pos="1 0 1" euler="0 90 0" size="1 2 .1"/>')
    origins = np.array([[3.0, 0, 1], [-1.0, 0, 1], [3.0, 3, 1], [3.0, 0, 3]])
    rays = np.array([[-1.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0], [-1.0, 0, 0]])
    expected, _ = _mj_ranges(scene, origins, rays)
    actual = jax.jit(scene.raycast)(jnp.array(origins), jnp.array(rays[:, None]))[:, 0]
    np.testing.assert_allclose(actual, np.where(expected < 0, 40, expected), atol=1e-6)
    np.testing.assert_allclose(actual, [2, 40, 40, 40])
    # MuJoCo contact planes remain infinite halfspaces, including outside the visual rectangle.
    np.testing.assert_allclose(scene.clearance(jnp.array(origins)), [2, -2, 2, 2])


@pytest.mark.parametrize("name", ["D01", "D02", "D03", "D06"])
def test_dynamic_asset_positions_and_rays_at_exact_times(name):
    """Verify dynamic asset positions and rays at exact times."""
    scene = Scene(name)
    origin = np.array([2.0, 0.0, 3.0])
    direction = np.random.default_rng(7).normal(size=(9, 3))
    times = jnp.array([0.0, 0.371, 2.917])
    positions = jax.jit(scene.positions)(times)
    actual = jax.jit(scene.raycast)(jnp.broadcast_to(origin, (3, 3)), jnp.array(direction), times)
    for i, t in enumerate(times):
        expected, data = _mj_ranges(scene, np.broadcast_to(origin, (9, 3)), direction, float(t))
        np.testing.assert_allclose(positions[i], data.geom_xpos, atol=1e-5)
        np.testing.assert_allclose(
            actual[i], np.where(expected < 0, 40, np.minimum(expected, 40)), atol=2e-4, rtol=1e-5
        )


def test_motion_distance_rays_and_gradients(make_scene):
    """Verify motion distance rays and gradients."""
    scene = make_scene(
        '<body pos="5 0 0" mocap="true" user="2 2 0 0 4 0"><geom type="box" size=".5 1 1"/></body>'
    )
    times = jnp.array([0.25, 0.75, 1.25])
    origins = jnp.zeros((3, 3))
    expected = 2.5 + 2 * times
    np.testing.assert_allclose(jax.jit(scene.clearance)(origins, times), expected)
    np.testing.assert_allclose(
        jax.jit(scene.raycast)(origins, jnp.array([1.0, 0, 0]), times), expected
    )
    derivative = jax.jit(jax.grad(lambda t: scene.raycast(jnp.zeros(3), jnp.array([1.0, 0, 0]), t)))
    np.testing.assert_allclose(derivative(0.75), 2)
    gradient = jax.jit(jax.grad(lambda p: scene.raycast(p, jnp.array([1.0, 0, 0]), 0.75)))
    np.testing.assert_allclose(gradient(jnp.zeros(3)), [-1, 0, 0])
    with pytest.raises(ValueError):
        scene.raycast(origins, jnp.eye(3), jnp.ones((3, 3)))


def test_geometry_hash_changes_with_shape_transform_and_motion(make_scene):
    """Verify geometry hash changes with shape transform and motion."""
    hashes = []
    for size, x, motion in [
        (1, 0, "0 0 0 0 0 0"),
        (2, 0, "0 0 0 0 0 0"),
        (1, 1, "0 0 0 0 0 0"),
        (1, 0, "2 1 0 0 4 0"),
    ]:
        hashes.append(
            make_scene(
                f'<body pos="{x} 0 0" mocap="true" user="{motion}"><geom size="{size}"/></body>'
            ).geometry_hash
        )
    assert len(set(hashes)) == 4


def test_device_profiles_fov_timing_phase_and_immutability():
    """Verify device profiles fov timing phase and immutability."""
    full = SensorConfig.d435i()
    training = SensorConfig.d435i("training64x48")
    assert (full.width, full.height, full.frequency_hz) == (1280, 720, 30)
    assert (training.width, training.height, training.pitch_deg) == (64, 48, 20)
    assert (training.horizontal_fov_deg, training.vertical_fov_deg) == (87, 58)
    with pytest.raises(FrozenInstanceError):
        full.width = 10
    with pytest.raises(ValueError, match=r"error model"):
        SensorConfig.mid360(error_model="vendor")
    config = SensorConfig.mid360()
    rays, offsets = jax.jit(lambda t: sensor_rays(config, t))(jnp.array([1.0, 1.1]))
    assert rays.shape == (2, 20000, 3)
    assert not np.allclose(rays[0], rays[1])
    np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1, atol=1e-6)
    np.testing.assert_allclose(np.diff(offsets), 1 / 200000, atol=1e-8)
    assert float(offsets[-1]) == 0
    elevation = np.rad2deg(np.arcsin(rays[..., 2]))
    assert elevation.min() >= -7.0001 and elevation.max() <= 52.0001
    azimuth = np.rad2deg(np.arctan2(rays[..., 1], rays[..., 0]))
    assert azimuth.max() - azimuth.min() > 359
    uniform, offsets = sensor_rays(SensorConfig.uniform_lidar_paper())
    assert uniform.shape == (5400, 3)
    np.testing.assert_allclose(
        np.rad2deg(np.arcsin(uniform[::180, 2])), np.arange(-30, 30, 2), atol=1e-5
    )
    np.testing.assert_array_equal(offsets, 0)


def test_axial_depth_extrinsics_masks_jit_batches_and_gradient(make_scene):
    """Verify axial depth extrinsics masks jit batches and gradient."""
    scene = make_scene('<geom type="box" pos="5.5 0 0" size=".5 50 50"/>')
    config = SensorConfig.d435i("training64x48", pitch_deg=0, translation=(0.5, 0, 0), latency=0.02)
    positions = jnp.array([[0.0, 0, 0], [1.0, 0, 0]])
    quaternion = jnp.array([0.0, 0, 0, 1.0])
    render = jax.jit(lambda p: measure(scene, config, p, quaternion, jnp.array([1.0, 2.0])))
    result = render(positions)
    assert result.values.shape == (2, 48, 64)
    assert result.mask.all()
    np.testing.assert_allclose(result.values[0], 4.5, atol=1e-6)
    np.testing.assert_allclose(result.values[1], 3.5, atol=1e-6)
    np.testing.assert_allclose(
        result.points_body[..., 0],
        np.broadcast_to(np.array([5.0, 4.0])[:, None, None], (2, 48, 64)),
        atol=1e-6,
    )
    np.testing.assert_allclose(result.available_time, [1.02, 2.02])
    gradient = jax.jit(jax.grad(lambda p: render(p).values.mean()))(positions)
    np.testing.assert_allclose(gradient, [[-0.5, 0, 0], [-0.5, 0, 0]], atol=1e-5)
    blind = SensorConfig.d435i(width=3, height=1, max_range=4.0)
    missing = measure(scene, blind, jnp.zeros(3), quaternion, 1.0)
    assert not missing.mask.any()
    np.testing.assert_array_equal(missing.points_body, 0)
    np.testing.assert_array_equal(missing.values, 0)
    assert np.isfinite(missing.directions_body).all()


def test_camera_pitch_yaw_and_range_cutoff(make_scene):
    """Verify camera pitch yaw and range cutoff."""
    scene = make_scene('<geom type="box" pos="0 5.5 0" size="50 .5 50"/>')
    config = SensorConfig.d435i("training64x48", width=1, height=1, pitch_deg=20)
    quat = Rotation.from_euler("z", jnp.pi / 2).as_quat()
    result = measure(scene, config, jnp.zeros(3), quat, 0.0)
    np.testing.assert_allclose(
        result.directions_body[0, 0], [np.cos(np.deg2rad(20)), 0, np.sin(np.deg2rad(20))], atol=1e-6
    )
    np.testing.assert_allclose(result.values, 5 / np.cos(np.deg2rad(20)), atol=2e-6)
    wall = make_scene('<geom type="box" pos="9.5 0 0" size=".5 50 50"/>')
    wide = SensorConfig.d435i(width=3, height=1, max_range=10)
    values = measure(wall, wide, jnp.zeros(3), jnp.array([0.0, 0, 0, 1.0]), 0.0)
    assert values.mask.all()  # Slant distance exceeds 10 m at the edge, axial depth does not.
    np.testing.assert_allclose(values.values, 9, atol=1e-6)


def test_full_resolution_depth(make_scene):
    """Verify full resolution depth."""
    scene = make_scene('<geom type="box" pos="1.5 0 0" size=".5 50 50"/>')
    config = SensorConfig.d435i()
    result = jax.jit(lambda p: measure(scene, config, p, jnp.array([0.0, 0, 0, 1.0]), 0.0))(
        jnp.zeros(3)
    )
    assert result.values.shape == (720, 1280)
    assert result.mask.all()
    np.testing.assert_allclose(result.values, 1, atol=1e-6)


def test_mid360_full_frame_and_per_ray_moving_geometry(make_scene):
    # A growing offset between sensor and a moving sphere makes timestamp errors observable.
    """Verify mid360 full frame and per ray moving geometry."""
    scene = make_scene(
        '<body pos="0 0 0" mocap="true" user="2 2 0 0 4 0"><geom type="sphere" size="20"/></body>'
    )
    config = SensorConfig.mid360()
    result = jax.jit(
        lambda t: measure(scene, config, jnp.zeros(3), jnp.array([0.0, 0, 0, 1.0]), t)
    )(0.75)
    assert result.values.shape == (20000,)
    assert result.mask.all()
    centers_x = -2 + 2 * np.asarray(result.times)
    dx = np.asarray(result.directions_body[:, 0])
    expected = centers_x * dx + np.sqrt(400 - centers_x**2 * (1 - dx**2))
    np.testing.assert_allclose(result.values, expected, atol=5e-6)
    reduced = jax.jit(lambda m: reduce_points(m, 64))(result)
    indices = np.linspace(0, 19999, 64).astype(int)
    np.testing.assert_array_equal(reduced.times, result.times[indices])
    np.testing.assert_array_equal(reduced.mask, result.mask[indices])
    np.testing.assert_array_equal(reduced.points_body, result.points_body[indices])


def test_scan_ego_motion_and_mask_preservation(make_scene):
    """Verify scan ego motion and mask preservation."""
    scene = make_scene('<geom type="sphere" size="20"/>')
    config = SensorConfig.mid360(points_per_frame=257)

    def pose_at(times):
        positions = jnp.stack((times, jnp.zeros_like(times), jnp.zeros_like(times)), axis=-1)
        quats = jnp.broadcast_to(jnp.array([0.0, 0, 0, 1.0]), (*times.shape, 4))
        return positions, quats

    result = jax.jit(
        lambda t: measure(
            scene, config, jnp.zeros((2, 3)), jnp.array([0.0, 0, 0, 1.0]), t, pose_at=pose_at
        )
    )(jnp.array([1.0, 2.0]))
    world = result.points_body + pose_at(result.times)[0]
    np.testing.assert_allclose(np.linalg.norm(world, axis=-1), 20, atol=4e-6)
    empty = measure(Scene("empty"), config, jnp.zeros(3), jnp.array([0.0, 0, 0, 1.0]), 0.0)
    reduced = reduce_points(empty, 16)
    assert not reduced.mask.any()
    assert (np.diff(reduced.times) > 0).all()


def test_preprocessing_inverse_pool_masks_and_gradient():
    """Verify preprocessing inverse pool masks and gradient."""
    depth = jnp.full((48, 64), 6.0).at[1, 1].set(1.0)
    mask = jnp.ones_like(depth, dtype=bool).at[:4, 4:8].set(False)
    measurement = Measurement(
        depth,
        mask,
        jnp.zeros((48, 64, 3)),
        jnp.zeros((48, 64, 3)),
        jnp.zeros_like(depth),
        jnp.array(0.0),
        jnp.array(0.0),
    )
    values, valid = jax.jit(preprocess_depth)(measurement)
    assert values.shape == valid.shape == (12, 16)
    np.testing.assert_allclose(values[0, 0], 2.4)
    assert values[0, 1] == 0 and not valid[0, 1]
    np.testing.assert_allclose(values[1, 1], -0.1, atol=1e-7)
    gradient = jax.grad(lambda d: preprocess_depth(measurement._replace(values=d))[0].sum())(depth)
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient[1, 1], -3)
    np.testing.assert_array_equal(gradient[:4, 4:8], 0)
