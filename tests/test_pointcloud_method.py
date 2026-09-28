"""Published network and loss equations, plus reconstruction-scene contracts."""

import importlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np


def required(name):
    assert importlib.util.find_spec(name), f"Required paper component: {name}"
    return importlib.import_module(name)


def test_pointnet_permutation_invariance_and_empty_returns():
    network = required("drone_playground.learning.pointcloud_network").PointCloudPolicy()
    points = jax.random.normal(jax.random.PRNGKey(1), (2, 7, 3))
    valid = jnp.array([[1, 1, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0]], bool)
    proprio = jnp.ones((2, 10))
    hidden = jnp.zeros((2, 192))
    params = network.init(jax.random.PRNGKey(0), points, valid, proprio, hidden)
    actual = network.apply(params, points, valid, proprio, hidden)
    reverse = network.apply(params, points[:, ::-1], valid[:, ::-1], proprio, hidden)
    np.testing.assert_allclose(actual[0], reverse[0], atol=1e-6)
    assert actual[0].shape == (2, 3) and actual[1].shape == (2, 192)
    encoded = network.apply(params, points, valid, method=network.encode)
    np.testing.assert_array_equal(encoded[1], jnp.zeros(192))
    assert np.isfinite(np.asarray(encoded)).all()
    more = network.apply(
        params,
        jnp.concatenate([points, points], 1),
        jnp.concatenate([valid, valid], 1),
        proprio,
        hidden,
    )
    np.testing.assert_allclose(actual[0], more[0], atol=1e-6)


def test_gru_keeps_history_and_encoder_parameters_receive_gradients():
    network = required("drone_playground.learning.pointcloud_network").PointCloudPolicy()
    points = jax.random.normal(jax.random.PRNGKey(3), (1, 8, 3))
    valid = jnp.ones((1, 8), bool)
    proprio = jnp.ones((1, 10))
    hidden = jnp.zeros((1, 192))
    params = network.init(jax.random.PRNGKey(2), points, valid, proprio, hidden)
    first, h = network.apply(params, points, valid, proprio, hidden)
    second, _ = network.apply(params, points, valid, proprio, h)
    assert not np.allclose(first, second)
    reset, _ = network.apply(params, points, valid, proprio, jnp.zeros_like(h))
    np.testing.assert_array_equal(first, reset)
    grad = jax.grad(lambda p: network.apply(p, points, valid, proprio, hidden)[0].sum())(params)
    leaves = jax.tree.leaves(grad["params"]["point_0"])
    assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(grad))
    assert sum(float(jnp.sum(g * g)) for g in leaves) > 0


def test_paper_velocity_loss_follows_vector_error_norm_equation():
    objective = required("drone_playground.learning.pointcloud_objective").PaperObjective(
        velocity_window=1
    )
    trace = dict(
        velocity=jnp.zeros((2, 1, 3)),
        target_velocity=jnp.tile(jnp.array([3.0, 4.0, 0.0]), (2, 1, 1)),
        acceleration=jnp.zeros((2, 1, 3)),
        clearance=jnp.full((2, 1), 10.0),
        approaching_speed=jnp.zeros((2, 1)),
    )
    total, parts = objective(trace, 0.1)
    expected = 0.8 * (5 - 0.5) + 0.6 * ((3 - 0.5) + (4 - 0.5))
    np.testing.assert_allclose(parts["velocity"], expected, rtol=1e-6)
    np.testing.assert_allclose(total, expected, rtol=1e-6)


def test_paper_collision_and_jerk_terms_have_explicit_units():
    objective = required("drone_playground.learning.pointcloud_objective").PaperObjective(
        velocity_weight=0, collision_weight=1, acceleration_weight=0, jerk_weight=0
    )
    clearance = jnp.array([[0.3], [0.2]])
    trace = dict(
        velocity=jnp.zeros((2, 1, 3)),
        target_velocity=jnp.zeros((2, 1, 3)),
        acceleration=jnp.zeros((2, 1, 3)),
        clearance=clearance,
        approaching_speed=jnp.ones((2, 1)) * 2,
    )
    loss, _ = objective(trace, 0.1)
    expected = jnp.mean(
        2 * (jnp.maximum(1 - clearance, 0) ** 2 + (4 / 3) * jax.nn.softplus(-32 * clearance))
    )
    np.testing.assert_allclose(loss, expected, rtol=1e-6)
    gradient = jax.grad(lambda c: objective({**trace, "clearance": c}, 0.1)[0])(clearance)
    assert np.all(np.asarray(gradient) < 0)
    jerk_only = required("drone_playground.learning.pointcloud_objective").PaperObjective(
        velocity_weight=0,
        collision_weight=0,
        acceleration_weight=0,
        jerk_weight=1,
        jerk_mean_weight=1,
        jerk_variance_weight=0,
    )
    trace["acceleration"] = jnp.array([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
    np.testing.assert_allclose(jerk_only(trace, 0.1)[0], 10.0, rtol=1e-6)


def test_paper_train_scene_is_static_and_seeded_independently():
    scene = required("drone_playground.tasks.scenes.pointcloud").PaperPrimitiveScene(
        obstacles_per_kind=2
    )
    a = scene.sample(jax.random.PRNGKey(1), 2)
    b = scene.sample(jax.random.PRNGKey(1), 2)
    c = scene.sample(jax.random.PRNGKey(2), 2)
    assert a.kind.shape == (2, 6)
    np.testing.assert_array_equal(a.origin, b.origin)
    assert not np.allclose(a.origin, c.origin)
    assert set(np.asarray(a.kind).ravel()) == {1, 2, 3}
    np.testing.assert_array_equal(a.motion, jnp.zeros((2, 6)))
    assert "navigation8" not in scene.name
