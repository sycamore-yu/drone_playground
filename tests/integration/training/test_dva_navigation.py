"""P5-05: D.VA detached-observation gradients and training contract."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs

from drone_playground.composition import (
    build_environment,
    compose_experiment,
    native_training_config,
)
from drone_playground.learning.algorithms.dva import (
    critic_observation_from_pipeline,
    detached_policy_action,
    load_training_state,
    terminal_critic_observation,
    train,
)
from drone_playground.learning.algorithms.shac import lambda_returns
from drone_playground.networks.factory import network_factory


def small_config(sensor: str) -> tuple[dict, dict]:
    overrides = [
            "runtime.device=cpu",
            "training.num_envs=2",
            "training.policy_updates=1",
            "training.num_timesteps=4",
            "training.num_evals=2",
            "algorithm.horizon_length=2",
            "algorithm.critic_updates=1",
            "training.max_wall_seconds=600",
    ]
    if sensor == "lidar":
        overrides.extend(
            [
                "sensor@env.sensor=mid360",
                "observation@env.observation=navigation_lidar",
            ]
        )
    elif sensor != "depth":
        raise ValueError(f"Unknown D.VA sensor fixture: {sensor}")
    resolved = compose_experiment('papers/dva', overrides=overrides)
    return resolved, native_training_config(resolved)


@pytest.mark.parametrize("experiment", ["depth", "lidar"])
def test_detached_actor_keeps_parameter_gradient_but_blocks_observation_gradient(experiment):
    _, config = small_config(experiment)
    factory = network_factory(config)
    networks = factory((config["observation_size"],), 4)
    normalizer = running_statistics.init_state(
        specs.Array((config["observation_size"],), jnp.float32)
    )
    policy = networks.policy_network.init(jax.random.PRNGKey(1))
    observation = jax.random.uniform(jax.random.PRNGKey(2), (config["observation_size"],))

    def from_observation(obs):
        return jnp.sum(
            detached_policy_action(
                networks,
                normalizer,
                policy,
                obs,
                jax.random.PRNGKey(3),
                deterministic=True,
            )
        )

    np.testing.assert_array_equal(jax.grad(from_observation)(observation), 0.0)

    def from_policy(params):
        return jnp.sum(
            detached_policy_action(
                networks,
                normalizer,
                params,
                observation,
                jax.random.PRNGKey(3),
                deterministic=True,
            )
        )

    grad = jax.grad(from_policy)(policy)
    squared_norm = sum(float(jnp.sum(jnp.square(x))) for x in jax.tree.leaves(grad))
    assert squared_norm > 0
    # Fixed observations and a common sample define the detached surrogate.
    norm = np.sqrt(squared_norm)
    direction = jax.tree.map(lambda g: g / norm, grad)
    eps = 1e-3
    plus = jax.tree.map(lambda p, d: p + eps * d, policy, direction)
    minus = jax.tree.map(lambda p, d: p - eps * d, policy, direction)
    finite_difference = (from_policy(plus) - from_policy(minus)) / (2 * eps)
    np.testing.assert_allclose(finite_difference, norm, rtol=2e-2, atol=1e-4)


def test_navigation_action_has_a_finite_nonzero_dynamics_derivative():
    resolved, _ = small_config("depth")
    env = build_environment(resolved, "cpu", "train", 1)
    try:
        state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
        action = jnp.array([0.01, -0.02, 0.0, 0.40], jnp.float32)

        def next_vertical_velocity(a):
            nxt = env.step(state, a)
            return nxt.pipeline_state.sim_data.states.vel[0, 0, 2]

        gradient = jax.grad(next_vertical_velocity)(action)
        assert np.isfinite(np.asarray(gradient)).all()
        assert float(jnp.linalg.norm(gradient)) > 1e-7

        eps = 1e-3
        plus = next_vertical_velocity(action.at[3].add(eps))
        minus = next_vertical_velocity(action.at[3].add(-eps))
        finite_difference = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(gradient[3], finite_difference, rtol=1e-2, atol=1e-4)
    finally:
        env.close()


def test_lambda_target_masks_true_termination_but_bootstraps_timeout():
    rewards = jnp.array([[1.0], [2.0]])
    values = jnp.array([[10.0], [20.0]])
    done = jnp.array([[False], [True]])
    terminated = jnp.array([[False], [True]])
    terminated_target = lambda_returns(rewards, values, done, terminated, 0.9, 0.95)
    assert np.isclose(float(terminated_target[-1, 0]), 2.0)

    timeout = jnp.array([[False], [False]])
    bootstrapped = lambda_returns(rewards, values, done, timeout, 0.9, 0.95)
    assert float(bootstrapped[-1, 0]) > 2.0


def test_terminal_critic_input_bypasses_sensor_but_keeps_state_derivative():
    resolved, config = small_config("depth")
    env = build_environment(resolved, "cpu", "train", 2)
    try:
        from drone_playground.learning.wrappers import wrap_for_training
        from drone_playground.networks.perception import SensorLayout

        layout = SensorLayout.from_dict(config["sensor_layout"])
        wrapped = wrap_for_training(env, env.episode_length)
        state = wrapped.reset(jax.random.split(jax.random.PRNGKey(7), 2))

        def first_position(action):
            nxt = wrapped.step(state, action)
            critic_obs = critic_observation_from_pipeline(env, nxt.pipeline_state, layout)
            return critic_obs[0, 0]

        action = jnp.tile(env.hover_action, (2, 1))
        grad = jax.grad(lambda x: first_position(x).sum())(action)
        assert np.isfinite(np.asarray(grad)).all()
        assert np.allclose(
            np.asarray(
                critic_observation_from_pipeline(env, state.pipeline_state, layout)[
                    :, layout.proprioception_size :
                ]
            ),
            0,
        )
    finally:
        env.close()


@pytest.mark.parametrize("experiment", ["depth", "lidar"])
def test_real_dva_update_is_finite_moves_encoder_and_saves_full_state(experiment):
    resolved, config = small_config(experiment)
    env = build_environment(resolved, "cpu", "train", 2)
    snapshots = []

    def capture(step, maker, params):
        snapshots.append((int(step), jax.tree.map(np.asarray, params)))

    try:
        with tempfile.TemporaryDirectory() as folder:
            maker, params, metrics = train(
                env,
                config,
                policy_params_fn=capture,
                state_directory=Path(folder),
            )
            assert metrics["actual_steps"] == 4
            assert metrics["actor_parameter_delta_l2"] > 0
            assert metrics["critic_parameter_delta_l2"] > 0
            assert all(
                np.isfinite(float(value))
                for key, value in metrics.items()
                if key.startswith("training/")
            )
            assert len(snapshots) == 2
            before, after = snapshots[0][1][1], snapshots[1][1][1]
            differences = [
                np.square(np.asarray(b) - np.asarray(a)).sum()
                for (path, a), b in zip(
                    jax.tree_util.tree_flatten_with_path(before)[0],
                    jax.tree.leaves(after),
                    strict=True,
                )
                if "Encoder" in jax.tree_util.keystr(path)
            ]
            assert differences and sum(differences) > 0, "The sensor encoder must actually update"
            state_file = Path(folder) / "update-0000001.pkl"
            restored, meta = load_training_state(state_file)
            assert int(restored.updates) == 1
            assert meta["kind"] == "dva-full-training-state"
            action = maker(params, deterministic=True)(
                env.reset(jax.random.PRNGKey(9), jnp.int32(0)).obs,
                jax.random.PRNGKey(10),
            )[0]
            assert np.isfinite(np.asarray(action)).all()
    finally:
        env.close()


def test_restore_rejects_different_sensor_contract():
    resolved, config = small_config("depth")
    env = build_environment(resolved, "cpu", "train", 2)
    try:
        with tempfile.TemporaryDirectory() as folder:
            train(env, config, state_directory=Path(folder))
            state_file = Path(folder) / "update-0000001.pkl"
            other = copy.deepcopy(config)
            other["sensor_layout"] = native_training_config(small_config("lidar")[0])[
                "sensor_layout"
            ]
            with pytest.raises(ValueError, match=r"sensor_layout"):
                train(env, other, restore_state=state_file)
    finally:
        env.close()


def test_timeout_retains_pre_reset_proprioception_and_derivative():
    from drone_playground.learning.wrappers import wrap_for_training
    from drone_playground.networks.perception import SensorLayout

    resolved, config = small_config("depth")
    env = build_environment(resolved, "cpu", "train", 1)
    try:
        wrapped = wrap_for_training(env, 1)
        state = wrapped.reset(jax.random.split(jax.random.PRNGKey(20), 1))
        layout = SensorLayout.from_dict(config["sensor_layout"])
        action = env.hover_action.at[3].add(0.1)[None]

        def terminal_velocity(a):
            nxt = wrapped.step(state, a)
            return terminal_critic_observation(nxt, layout)[0, 9]

        nxt = wrapped.step(state, action)
        assert float(nxt.info["time_out"][0]) == 1.0
        assert float(nxt.info["terminated"][0]) == 0.0
        assert float(nxt.pipeline_state.sim_data.states.vel[0, 0, 0, 2]) == 0.0
        assert abs(float(terminal_velocity(action))) > 1e-6
        derivative = jax.grad(terminal_velocity)(action)
        assert np.isfinite(derivative).all()
        assert abs(float(derivative[0, 3])) > 1e-6
    finally:
        env.close()


def test_resume_matches_uninterrupted_training_and_rejects_schedule_change(tmp_path):
    resolved, config = small_config("depth")
    config.update(policy_updates=2, num_timesteps=8, num_evals=3)
    env = build_environment(resolved, "cpu", "train", 2)
    try:
        train(env, config, state_directory=tmp_path / "continuous")
        checkpoint = tmp_path / "continuous" / "update-0000001.pkl"
        train(env, config, restore_state=checkpoint, state_directory=tmp_path / "resumed")
        continuous, _ = load_training_state(tmp_path / "continuous" / "update-0000002.pkl")
        resumed, _ = load_training_state(tmp_path / "resumed" / "update-0000002.pkl")
        for a, b in zip(jax.tree.leaves(continuous), jax.tree.leaves(resumed), strict=True):
            if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
                a, b = jax.random.key_data(a), jax.random.key_data(b)
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        with pytest.raises(ValueError, match=r"policy_updates"):
            train(env, {**config, "policy_updates": 3}, restore_state=checkpoint)
    finally:
        env.close()


def test_collision_distance_has_finite_interior_and_surface_subgradients():
    from drone_playground.environments.scenes.geometry import (
        KIND_BOX,
        KIND_CYLINDER,
        signed_distance,
    )

    for kind in (KIND_BOX, KIND_CYLINDER):

        def distance(p, kind=kind):
            return signed_distance(jnp.int32(kind), jnp.ones(3), jnp.zeros(3), p)

        for point in ([0.0, 0.0, 0.0], [0.2, 0.1, 0.1], [1.0, 0.0, 0.0], [2.0, 0.1, 0.1]):
            assert np.isfinite(jax.grad(distance)(jnp.array(point))).all()

    def distance(p):
        return signed_distance(jnp.int32(KIND_BOX), jnp.ones(3), jnp.zeros(3), p)

    np.testing.assert_allclose(jax.grad(distance)(jnp.array([0.2, 0.1, 0.1])), [1.0, 0.0, 0.0])


def test_fixed_observation_surrogate_matches_end_to_end_finite_difference():
    from drone_playground.learning.algorithms.shac import bootstrap_value, segment_objective
    from drone_playground.networks.perception import SensorLayout

    resolved, config = small_config("depth")
    env = build_environment(resolved, "cpu", "train", 1)
    try:
        layout = SensorLayout.from_dict(config["sensor_layout"])
        networks = network_factory(config)((env.observation_size,), 4)
        normalizer = running_statistics.init_state(
            specs.Array((env.observation_size,), jnp.float32)
        )
        policy = networks.policy_network.init(jax.random.PRNGKey(80))
        critic = networks.value_network.init(jax.random.PRNGKey(81))
        initial = env.reset(jax.random.PRNGKey(82), jnp.int32(0))
        current, observations = initial, []
        for tick in range(2):
            observations.append(current.obs)
            action = detached_policy_action(
                networks,
                normalizer,
                policy,
                current.obs,
                jax.random.PRNGKey(tick),
                deterministic=True,
            )
            current = env.step(current, action)

        def surrogate(offset):
            params = copy.deepcopy(policy)
            params["params"]["mean_head"]["bias"] = (
                params["params"]["mean_head"]["bias"].at[3].add(offset)
            )
            current = initial
            rewards, values, dones = [], [], []
            for tick, frozen in enumerate(observations):
                action = detached_policy_action(
                    networks,
                    normalizer,
                    params,
                    frozen,
                    jax.random.PRNGKey(tick),
                    deterministic=True,
                )
                current = env.step(current, action)
                value = bootstrap_value(
                    lambda weights, obs: networks.value_network.apply(normalizer, weights, obs),
                    critic,
                    terminal_critic_observation(current, layout),
                    current.done.astype(bool),
                )
                rewards.append(current.reward)
                values.append(value)
                dones.append(current.done.astype(bool))
            return segment_objective(
                jnp.array(rewards)[:, None],
                jnp.array(values)[:, None],
                jnp.array(dones)[:, None],
                0.99,
            )

        derivative = jax.jit(jax.grad(surrogate))(jnp.float32(0))
        forward = jax.jit(surrogate)
        eps = jnp.float32(0.01)
        finite_difference = (forward(eps) - forward(-eps)) / (2 * eps)
        assert abs(float(derivative)) > 1e-7
        np.testing.assert_allclose(derivative, finite_difference, rtol=3e-2, atol=2e-4)
    finally:
        env.close()
