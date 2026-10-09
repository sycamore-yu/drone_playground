"""CPU contract tests for shared actors, named objectives and temporal derivatives."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.learning.losses import (
    approach_weighted_clearance,
    causal_velocity_average,
    liu_loss,
    temporal_gradient_decay,
    velocity_auxiliary_loss,
    velocity_huber_loss,
    zhang_loss,
)
from drone_playground.simulation.networks import (
    Actor,
    Critic,
    sample_tanh_gaussian,
    tanh_gaussian_log_prob,
)


def observations(kind, batch=2):
    """Provide observations for the surrounding execution."""
    state_key, sensor_key = jax.random.split(jax.random.key(42))
    obs = {"state": jax.random.normal(state_key, (batch, 10))}
    if kind == "depth":
        obs["depth"] = jax.random.uniform(sensor_key, (batch, 12, 16, 1))
    if kind == "lidar":
        obs["points"] = jax.random.normal(sensor_key, (batch, 7, 3))
        obs["mask"] = jnp.broadcast_to(jnp.array([1, 0, 1, 1, 0, 1, 0], bool), (batch, 7))
    return obs


def assert_tree_equal(left, right):
    """Provide assert tree equal for the surrounding execution."""
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kind,action_size", [("state", 4), ("depth", 3), ("lidar", 3)])
def test_actor_contract_determinism_jit_and_algorithm_independent_init(kind, action_size):
    """Verify actor contract determinism jit and algorithm independent init."""
    obs = observations(kind)
    actor = Actor(kind=kind, action_size=4 if kind == "state" else 3)
    memory = actor.initialize_memory(2)
    key = jax.random.key(10)
    params = actor.init(key, obs, memory)
    action, updated, auxiliary = actor.apply(params, obs, memory)
    assert action.shape == (2, action_size)
    assert updated.shape == memory.shape
    assert auxiliary.shape == (2, 3)
    assert_tree_equal(actor.apply(params, obs, memory), (action, updated, auxiliary))
    compiled = jax.jit(actor.apply)(params, obs, memory)
    for actual, expected in zip(compiled, (action, updated, auxiliary), strict=True):
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        assert np.isfinite(actual).all()
    # No algorithm field or algorithm-specific parameter initialization exists.
    assert "algorithm" not in inspect.signature(Actor).parameters
    for _algorithm in ("ppo", "apg", "shac"):
        assert_tree_equal(
            params, Actor(kind=kind, action_size=4 if kind == "state" else 3).init(key, obs, memory)
        )
    if kind == "state":
        sentinel = jnp.full_like(memory, 2.0)
        result = actor.apply(params, obs, sentinel)
        np.testing.assert_array_equal(result[0], action)
        np.testing.assert_array_equal(result[1], sentinel)
    else:
        next_result = actor.apply(params, obs, updated)
        assert not np.allclose(next_result[1], updated)
        reset = updated.at[0].set(0)
        reset_result = actor.apply(params, obs, reset)
        np.testing.assert_allclose(reset_result[0][0], action[0], atol=1e-7)
        np.testing.assert_allclose(reset_result[0][1], next_result[0][1], atol=1e-7)
    if kind != "depth":
        np.testing.assert_array_equal(auxiliary, 0)
    else:
        assert np.any(np.asarray(auxiliary) != 0)


@pytest.mark.parametrize("kind", ["state", "depth", "lidar"])
def test_actor_parameter_and_observation_gradients(kind):
    """Verify actor parameter and observation gradients."""
    obs = observations(kind)
    actor = Actor(kind=kind, action_size=4 if kind == "state" else 3)
    memory = actor.initialize_memory(2)
    params = actor.init(jax.random.key(2), obs, memory)

    def objective(variables, inputs):
        action, _, auxiliary = actor.apply(variables, inputs, memory)
        return jnp.sum(action**2) + jnp.sum(auxiliary**2)

    param_grad = jax.jit(jax.grad(objective))(params, obs)
    for gradient in jax.tree.leaves(param_grad):
        assert np.isfinite(gradient).all()
    assert sum(float(jnp.sum(jnp.abs(x))) for x in jax.tree.leaves(param_grad)) > 0
    fields = ("state",) if kind == "state" else ("state", "depth" if kind == "depth" else "points")
    for field in fields:
        gradient = jax.grad(lambda x, field=field: objective(params, {**obs, field: x}))(obs[field])
        assert np.isfinite(gradient).all()
        assert np.any(np.asarray(gradient) != 0)


def test_independent_batched_parameter_trees():
    """Verify independent batched parameter trees."""
    actor = Actor(kind="state")
    obs = observations("state")
    memory = actor.initialize_memory(2)
    keys = jax.random.split(jax.random.key(8), 2)
    params = jax.vmap(actor.init, in_axes=(0, None, None))(keys, obs, memory)
    actions, _, _ = jax.vmap(actor.apply, in_axes=(0, None, None))(params, obs, memory)
    assert actions.shape == (2, 2, 4)
    assert not np.allclose(actions[0], actions[1])


def test_lidar_mask_permutation_padding_no_hits_and_empty_cloud():
    """Verify lidar mask permutation padding no hits and empty cloud."""
    obs = observations("lidar")
    memory = jnp.zeros((2, 192))
    actor = Actor(kind="lidar")
    params = actor.init(jax.random.key(3), obs, memory)
    expected = actor.apply(params, obs, memory)
    permutation = jnp.array([6, 3, 0, 2, 5, 4, 1])
    permuted = {**obs, "points": obs["points"][:, permutation], "mask": obs["mask"][:, permutation]}
    assert_tree_equal(actor.apply(params, permuted, memory), expected)
    for invalid in (jnp.nan, jnp.inf, -1e30):
        padded = {**obs, "points": jnp.where(obs["mask"][..., None], obs["points"], invalid)}
        assert_tree_equal(actor.apply(params, padded, memory), expected)

    no_hits = {
        **obs,
        "points": jnp.full_like(obs["points"], jnp.nan),
        "mask": jnp.zeros_like(obs["mask"]),
    }
    no_hit_result = jax.jit(actor.apply)(params, no_hits, memory)
    for value in no_hit_result:
        assert np.isfinite(value).all()
    param_grad, point_grad = jax.grad(
        lambda p, points: actor.apply(p, {**no_hits, "points": points}, memory)[0].sum(),
        argnums=(0, 1),
    )(params, no_hits["points"])
    for value in jax.tree.leaves(param_grad):
        assert np.isfinite(value).all()
    np.testing.assert_array_equal(point_grad, 0)
    empty = {**obs, "points": jnp.empty((2, 0, 3)), "mask": jnp.empty((2, 0), dtype=bool)}
    empty_result = actor.apply(params, empty, memory)
    for actual, expected_value in zip(empty_result, no_hit_result, strict=True):
        np.testing.assert_allclose(actual, expected_value, atol=1e-7)


def test_actor_rejects_invalid_contracts():
    """Verify actor rejects invalid contracts."""
    memory = jnp.zeros((2, 192))
    key = jax.random.key(0)
    with pytest.raises(ValueError, match="kind"):
        Actor(kind="unknown").init(key, observations("state"), memory)
    with pytest.raises(ValueError, match="memory"):
        Actor().init(key, observations("state"), jnp.zeros((2, 191)))
    with pytest.raises(ValueError, match="depth"):
        bad_depth = {**observations("depth"), "depth": jnp.ones((2, 16, 12, 1))}
        Actor(kind="depth").init(key, bad_depth, memory)
    obs = observations("lidar")
    with pytest.raises(ValueError, match="boolean mask"):
        Actor(kind="lidar").init(key, {**obs, "mask": obs["mask"].astype(float)}, memory)


def test_critic_standalone_dict_and_state_and_gradients():
    """Verify critic standalone dict and state and gradients."""
    critic = Critic()
    obs = observations("state")
    params = critic.init(jax.random.key(9), obs)
    value = critic.apply(params, obs)
    assert value.shape == (2,)
    np.testing.assert_array_equal(value, critic.apply(params, obs["state"]))
    np.testing.assert_allclose(jax.jit(critic.apply)(params, obs), value, atol=1e-7)
    gradient = jax.grad(lambda p: jnp.sum(critic.apply(p, obs) ** 2))(params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(gradient))
    assert sum(float(jnp.sum(jnp.abs(x))) for x in jax.tree.leaves(gradient)) > 0


def test_tanh_gaussian_density_and_saturation_gradients():
    """Verify tanh gaussian density and saturation gradients."""
    pre_tanh = jnp.array([[0.0, 0.3, -0.7], [0.2, -0.2, 0.5]])
    mean = jnp.full_like(pre_tanh, 0.1)
    log_std = jnp.array([-0.4, 0.0, 0.5])
    actual = tanh_gaussian_log_prob(pre_tanh, mean, log_std)
    expected = (
        -0.5 * ((pre_tanh - mean) / jnp.exp(log_std)) ** 2
        - log_std
        - 0.5 * jnp.log(2 * jnp.pi)
        - jnp.log(1 - jnp.tanh(pre_tanh) ** 2)
    ).sum(-1)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    extremes = jnp.array([[-100.0, 0.0, 100.0]])
    assert np.isfinite(tanh_gaussian_log_prob(extremes, mean[:1], log_std)).all()
    gradients = jax.grad(lambda m, s: tanh_gaussian_log_prob(extremes, m, s).sum(), (0, 1))(
        mean[:1], log_std
    )
    assert all(np.isfinite(x).all() and np.any(np.asarray(x) != 0) for x in gradients)
    key = jax.random.key(42)
    action, raw, log_prob = sample_tanh_gaussian(key, mean, log_std)
    assert_tree_equal(sample_tanh_gaussian(key, mean, log_std), (action, raw, log_prob))
    np.testing.assert_array_equal(action, jnp.tanh(raw))
    np.testing.assert_array_equal(log_prob, tanh_gaussian_log_prob(raw, mean, log_std))
    assert log_prob.shape == (2,)


def test_causal_velocity_smoothing_and_exact_huber_formula():
    """Verify causal velocity smoothing and exact huber formula."""
    velocity = jnp.arange(5, dtype=float)[:, None, None] * jnp.ones((1, 2, 3))
    average = causal_velocity_average(velocity, window=3)
    np.testing.assert_allclose(average[:, 0, 0], [0, 0.5, 1, 2, 3])
    changed_future = velocity.at[-1].set(100)
    np.testing.assert_array_equal(
        causal_velocity_average(changed_future, window=3)[:-1], average[:-1]
    )
    errors = jnp.array([[[0.3, 0.4, 0.0]], [[3.0, 4.0, 0.0]]])
    radial_expected = (0.5 * 0.5**2 + (5 - 0.5)) / 2
    component_expected = (0.5 * (0.3**2 + 0.4**2) + (3 - 0.5) + (4 - 0.5)) / 2
    actual = velocity_huber_loss(errors, jnp.zeros_like(errors), window=1)
    np.testing.assert_allclose(actual, radial_expected)
    liu_velocity = velocity_huber_loss(
        errors, jnp.zeros_like(errors), window=1, norm_weight=0.8, component_weight=0.6
    )
    np.testing.assert_allclose(liu_velocity, 0.8 * radial_expected + 0.6 * component_expected)
    # This also protects the non-default delta scale (Huber, not Huber/delta).
    np.testing.assert_allclose(
        velocity_huber_loss(errors, jnp.zeros_like(errors), window=1, delta=2),
        (0.125 + 2 * (5 - 1)) / 2,
    )
    gradient = jax.grad(lambda v: velocity_huber_loss(v, v * 0))(jnp.zeros((2, 1, 3)))
    np.testing.assert_array_equal(gradient, 0)


def test_clearance_exact_terms_mask_and_detached_approach():
    """Verify clearance exact terms mask and detached approach."""
    distance = jnp.array([[-0.1, 0.5, 2.0, jnp.inf, jnp.nan]])
    speed = jnp.array([[2.0, 0.2, -3.0, jnp.inf, jnp.nan]])
    mask = jnp.array([[1, 1, 1, 0, 0]], bool)
    barrier, collision = approach_weighted_clearance(distance, speed, mask=mask)
    np.testing.assert_allclose(barrier, (2 * 1.1**2 + 0.5**2) / 3, rtol=1e-6)
    expected_collision = (
        np.sum(np.array([2, 1, 1]) * np.logaddexp(0, -32 * np.array([-0.1, 0.5, 2]))) / 3
    )
    np.testing.assert_allclose(collision, expected_collision, rtol=1e-6)
    d_grad, speed_grad = jax.grad(
        lambda d, s: sum(approach_weighted_clearance(d, s, mask=mask)), (0, 1)
    )(distance, speed)
    assert np.isfinite(d_grad).all()
    assert np.all(np.asarray(d_grad)[0, :2] < 0)
    np.testing.assert_array_equal(d_grad[~mask], 0)
    np.testing.assert_array_equal(speed_grad, 0)
    assert_tree_equal(
        approach_weighted_clearance(distance, speed, mask=jnp.zeros_like(mask)),
        (jnp.array(0.0), jnp.array(0.0)),
    )
    retreat = approach_weighted_clearance(
        distance[:, :3], -jnp.ones((1, 3)), minimum_approach_speed=0
    )
    assert_tree_equal(retreat, (jnp.array(0.0), jnp.array(0.0)))


def test_named_objectives_exact_acceleration_jerk_and_weighted_totals():
    """Verify named objectives exact acceleration jerk and weighted totals."""
    velocity = jnp.zeros((3, 2, 3))
    acceleration = jnp.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ]
    )
    clearance = jnp.full((3, 2), 0.5)
    speed = jnp.ones_like(clearance)
    total, terms = zhang_loss(
        velocity,
        velocity,
        acceleration,
        clearance,
        speed,
        dt=0.5,
        velocity_prediction=jnp.ones_like(velocity),
        velocity_aux_weight=2,
    )
    np.testing.assert_allclose(terms["acceleration"], 37 / 6)
    np.testing.assert_allclose(terms["jerk"], (4 + 16 + 36 + 16) / 4)
    np.testing.assert_allclose(terms["velocity_aux"], 1)
    expected = 1.5 * terms["clearance"] + 2 * terms["collision"] + 0.01 * 37 / 6 + 0.001 * 18 + 2
    np.testing.assert_allclose(total, expected)
    total, terms = liu_loss(velocity, velocity, acceleration, clearance, speed, dt=0.5)
    np.testing.assert_allclose(terms["acceleration"], 37 / 6)
    np.testing.assert_allclose(terms["jerk_mean"], 4)
    np.testing.assert_allclose(terms["jerk_variance"], 2)
    np.testing.assert_allclose(terms["jerk"], 4.2)
    expected = (
        1.5 * (terms["clearance"] + (4 / 3) * terms["collision"]) + 0.01 * 37 / 6 + 0.001 * 4.2
    )
    np.testing.assert_allclose(total, expected)


def test_auxiliary_velocity_detaches_only_target():
    """Verify auxiliary velocity detaches only target."""
    prediction = jnp.ones((2, 1, 3))
    velocity = jnp.zeros_like(prediction)
    p_grad, v_grad = jax.grad(velocity_auxiliary_loss, (0, 1))(prediction, velocity)
    np.testing.assert_allclose(p_grad, 2 / prediction.size)
    np.testing.assert_array_equal(v_grad, 0)
    with pytest.raises(ValueError, match="velocity_prediction"):
        zhang_loss(
            velocity,
            velocity,
            velocity,
            velocity[..., 0],
            velocity[..., 0],
            dt=0.1,
            velocity_aux_weight=1,
        )


@pytest.mark.parametrize("objective", [zhang_loss, liu_loss])
def test_objectives_jit_nonzero_gradients_and_zero_short_rollouts(objective):
    """Verify objectives jit nonzero gradients and zero short rollouts."""
    velocity = jnp.full((4, 2, 3), 0.2)
    acceleration = jnp.arange(24, dtype=float).reshape(4, 2, 3) / 10
    distance = jnp.full((4, 2), 0.4)

    def evaluate(v, a, c):
        return objective(v, jnp.zeros_like(v), a, c, jnp.ones_like(c), dt=0.1)[0]

    value, gradients = jax.jit(jax.value_and_grad(evaluate, (0, 1, 2)))(
        velocity, acceleration, distance
    )
    assert np.isfinite(value)
    assert all(np.isfinite(g).all() and np.any(np.asarray(g) != 0) for g in gradients)
    zeros = jnp.zeros((1, 2, 3))
    _, terms = objective(zeros, zeros, zeros, distance[:1], jnp.ones_like(distance[:1]), dt=0.1)
    assert float(terms["jerk"]) == 0
    gradients = jax.grad(evaluate, (0, 1, 2))(zeros, zeros, distance[:1])
    assert all(np.isfinite(g).all() for g in gradients)


def test_temporal_decay_exact_forward_state_jacobian_and_undamped_action():
    """Verify temporal decay exact forward state jacobian and undamped action."""
    state = jnp.array([0.2, -0.7])
    action = jnp.array([0.4, 0.3])
    alpha, dt = 0.7, 0.1
    factor = np.exp(-alpha * dt)

    def step(x, u):
        return jnp.array([x[0] ** 2 + 3 * x[1] + 2 * u[0], x[0] * u[1] + jnp.sin(x[1])])

    def decayed_step(x, u):
        return step(temporal_gradient_decay(x, alpha=alpha, dt=dt), u)

    np.testing.assert_array_equal(decayed_step(state, action), step(state, action))
    dx, du = jax.jacrev(step, (0, 1))(state, action)
    decayed_dx, decayed_du = jax.jit(jax.jacrev(decayed_step, (0, 1)))(state, action)
    np.testing.assert_allclose(decayed_dx, factor * dx, rtol=1e-6)
    np.testing.assert_array_equal(decayed_du, du)
    np.testing.assert_allclose(jax.jacfwd(decayed_step)(state, action), factor * dx, rtol=1e-6)
    tree = {"x": state, "counter": jnp.array([1, 2]), "done": jnp.array([False, True])}
    assert_tree_equal(temporal_gradient_decay(tree, alpha=alpha, dt=dt), tree)
    np.testing.assert_array_equal(
        jax.jacrev(lambda x: temporal_gradient_decay(x, alpha=0, dt=dt))(state), jnp.eye(2)
    )


def test_temporal_decay_scan_horizon_and_closed_loop_action_path():
    """Verify temporal decay scan horizon and closed loop action path."""
    alpha, dt, steps = 0.8, 0.2, 5
    factor = np.exp(-alpha * dt)

    def rollout(x, actions):
        def body(state, action):
            return 2 * temporal_gradient_decay(state, alpha=alpha, dt=dt) + 3 * action, None

        return jax.lax.scan(body, x, actions)[0]

    actions = jnp.arange(steps, dtype=float)
    dx, du = jax.jit(jax.grad(rollout, (0, 1)))(jnp.array(0.4), actions)
    np.testing.assert_allclose(dx, (2 * factor) ** steps, rtol=1e-6)
    np.testing.assert_allclose(du, 3 * (2 * factor) ** np.arange(steps - 1, -1, -1), rtol=1e-6)
    np.testing.assert_array_equal(du[-1], 3)

    def closed_loop(x, theta):
        action = theta * x
        return 2 * temporal_gradient_decay(x, alpha=alpha, dt=dt) + 3 * action

    dx, dtheta = jax.grad(closed_loop, (0, 1))(jnp.array(0.4), jnp.array(0.7))
    np.testing.assert_allclose(dx, 2 * factor + 3 * 0.7, rtol=1e-6)
    np.testing.assert_allclose(dtheta, 3 * 0.4, rtol=1e-6)
