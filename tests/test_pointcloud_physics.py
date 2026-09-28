"""Paper-method physical and sensor contracts, independently specified."""

import importlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def physics():
    name = "drone_playground.models.point_mass"
    assert importlib.util.find_spec(name), "paper point-mass module is required"
    return importlib.import_module(name)


def test_point_mass_forward_lag_and_zero_acceleration_hover():
    m = physics()
    state = m.PointMassState.create(jnp.array([0.0, 0.0, 2.0]))
    model = m.PointMassLag(time_constant=1 / 12, backward="direct")
    hover = model.step(state, jnp.zeros(3), 0.1)
    np.testing.assert_allclose(hover.pos, state.pos, atol=1e-7)
    command = jnp.array([2.0, -1.0, 0.0])
    actual = model.step(state, command, 0.1)
    acceleration = (1 - np.exp(-1.2)) * np.asarray(command)
    np.testing.assert_allclose(actual.acc, acceleration, rtol=1e-6)
    np.testing.assert_allclose(actual.vel, 0.05 * acceleration, rtol=1e-6)
    np.testing.assert_allclose(actual.pos, state.pos, atol=1e-7)


def test_point_mass_direct_and_decayed_state_jacobians():
    m = physics()
    x = jnp.array([0.1, 0.2, 2.0, 0.4, -0.2, 0.0, 0.1, 0.0, 0.0])
    u = jnp.array([1.0, -0.3, 0.2])
    direct = m.PointMassLag(time_constant=0.2, backward="direct")
    decay = m.PointMassLag(time_constant=0.2, backward="exponential", decay_rate=0.7)

    def transition(model, state, action):
        return model.step(m.PointMassState.from_vector(state), action, 0.1).vector()

    np.testing.assert_array_equal(transition(direct, x, u), transition(decay, x, u))
    dx = jax.jacrev(lambda s: transition(direct, s, u))(x)
    du = jax.jacrev(lambda a: transition(direct, x, a))(u)
    np.testing.assert_allclose(
        jax.jacrev(lambda s: transition(decay, s, u))(x), np.exp(-0.07) * dx, atol=1e-6
    )
    np.testing.assert_allclose(jax.jacrev(lambda a: transition(decay, x, a))(u), du, atol=1e-6)
    eps = 1e-3
    fd = np.stack(
        [
            (transition(direct, x + eps * d, u) - transition(direct, x - eps * d, u)) / (2 * eps)
            for d in jnp.eye(9)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(dx, fd, atol=1e-4, rtol=1e-3)


def test_point_mass_attitude_is_finite_at_zero_speed_and_vertical_thrust():
    m = physics()
    state = m.PointMassState.create(jnp.array([0.0, 0.0, 2.0]))
    nxt = jax.jit(m.PointMassLag().step)(state, jnp.array([0.0, 0.0, -9.80665]), 0.1)
    assert np.isfinite(np.asarray(nxt.rotation)).all()
    np.testing.assert_allclose(nxt.rotation.T @ nxt.rotation, np.eye(3), atol=1e-5)


def test_uniform_mid360_has_paper_ray_count_and_body_frame():
    name = "drone_playground.environments.sensors.pointcloud"
    assert importlib.util.find_spec(name), "paper MID-360 module is required"
    sensor = importlib.import_module(name).UniformMid360Lidar()
    rays = np.asarray(sensor.directions(0))
    assert rays.shape == (5400, 3)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1, atol=1e-6)
    assert sensor.source_rate_hz == 10.0
    assert sensor.points_per_frame == 5400
    elevations = np.degrees(np.arcsin(rays[:, 2]))
    assert len(np.unique(np.round(elevations, 3))) == 30


def test_shared_geometry_supports_sphere_distance_and_ray_hit():
    from drone_playground.environments.scenes import navigation as n
    from drone_playground.environments.sensors import rays

    assert hasattr(n, "KIND_SPHERE"), "shared geometry needs the paper sphere primitive"
    kind = jnp.int32(n.KIND_SPHERE)
    size = jnp.array([1.0, 1.0, 1.0])
    centre = jnp.array([4.0, 0.0, 2.0])
    assert float(n.signed_distance(kind, size, centre, jnp.array([2.0, 0.0, 2.0]))) == 1.0
    hit = rays.primitive_hit(
        kind, size, centre, jnp.array([[0.0, 0.0, 2.0]]), jnp.array([[1.0, 0.0, 0.0]]), None
    )
    np.testing.assert_allclose(hit, [3.0], atol=1e-6)
    jac = jax.grad(lambda p: n.signed_distance(kind, size, centre, p))(jnp.array([2.0, 0.0, 2.0]))
    np.testing.assert_allclose(jac, [-1.0, 0.0, 0.0], atol=1e-6)


def test_acceleration_controller_preserves_units_and_rejects_wrong_shape():
    name = "drone_playground.execution.controllers.acceleration"
    assert importlib.util.find_spec(name), "acceleration controller is required"
    controller = importlib.import_module(name).AccelerationControl()
    u = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(controller.physical_action(u), u)
    with pytest.raises(ValueError):
        controller.physical_action(jnp.ones(4))


def test_parallel_box_ray_hits_when_parallel_slabs_contain_origin():
    from drone_playground.environments.sensors.rays import primitive_hit

    hit = primitive_hit(
        jnp.int32(2),
        jnp.ones(3),
        jnp.array([4.0, 0.0, 2.0]),
        jnp.array([[0.0, 0.0, 2.0]]),
        jnp.array([[1.0, 0.0, 0.0]]),
        None,
    )
    np.testing.assert_allclose(hit, [3.0], atol=1e-6)


def test_static_obstacle_motion_has_finite_zero_time_derivative():
    from drone_playground.environments.scenes.navigation import SceneBank, obstacle_positions

    bank = SceneBank(
        kind=jnp.ones((1, 1), jnp.int32),
        size=jnp.ones((1, 1, 3)),
        origin=jnp.array([[[4.0, 0.0, 2.0]]]),
        motion=jnp.zeros((1, 1), jnp.int32),
        params=jnp.zeros((1, 1, 5)),
        active=jnp.ones((1, 1), bool),
        start=jnp.zeros((1, 3)),
        goal=jnp.ones((1, 3)),
        difficulty=jnp.zeros(1, jnp.int32),
        subtype=jnp.zeros(1, jnp.int32),
        world_low=jnp.zeros(3),
        world_high=jnp.ones(3),
    )
    derivative = jax.jacrev(lambda t: obstacle_positions(bank, 0, t))(jnp.float32(0.1))
    np.testing.assert_array_equal(derivative, jnp.zeros((1, 3)))
