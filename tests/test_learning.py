"""Real physics updates, episode semantics and exact recoverable training."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.learning.apg import rollout_objective
from drone_playground.learning.checkpoint import load_inference, load_state, save_state
from drone_playground.learning.ppo import (
    collect_rollout,
    generalized_advantage_estimate,
    normalize_advantages,
    recurrent_policy,
)
from drone_playground.learning.shac import update_target
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.environment import Environment
from drone_playground.simulation.networks import Actor, Critic
from drone_playground.simulation.tasks import Event


def test_additional_failure_cost_reaches_public_update_metrics():
    """Apply an extra terminal cost while retaining identical reset behavior."""
    env = Environment(task="navigation", scene="S01", sensor="depth", duration=0.02)
    base = Trainer(env, kind="depth", config={"horizon": 1})
    penalized = Trainer(env, kind="depth", config={"horizon": 1, "failure_cost": 160.0})
    base_state, base_metrics = base.update(base.initialize())
    penalized_state, penalized_metrics = penalized.update(penalized.initialize())
    assert float(base_metrics["done_fraction"]) == 1.0
    assert float(penalized_metrics["done_fraction"]) == 1.0
    np.testing.assert_allclose(
        penalized_metrics["reward"] - base_metrics["reward"], -160.0, atol=2e-5
    )
    np.testing.assert_allclose(penalized_metrics["failure_cost"], 160.0, atol=2e-5)
    np.testing.assert_array_equal(base_state.env_state.time, penalized_state.env_state.time)


def test_navigation_progress_reward_matches_goal_distance_change():
    """Reward physical progress once per active transition, without changing physics."""
    env = Environment(task="navigation", scene="S01", sensor="depth", num_envs=1)
    base = Trainer(env, kind="depth", config={"horizon": 1, "perception_weight": 0})
    shaped = Trainer(
        env,
        kind="depth",
        config={"horizon": 1, "perception_weight": 0, "progress_reward_scale": 0.5},
    )
    initialized = base.initialize()
    before = initialized.env_state
    moved = before.physics.states.pos.at[0, 0, 0].add(1.0)
    after = before.replace(
        physics=before.physics.replace(states=before.physics.states.replace(pos=moved))
    )
    action = jnp.zeros((1, 3))
    cost, _, _ = base._cost(before, after, action, initialized.objective_state)
    shaped_cost, _, terms = shaped._cost(before, after, action, initialized.objective_state)
    expected = 0.5 * (
        jnp.linalg.norm(env.task.goal - before.physics.states.pos[0, 0])
        - jnp.linalg.norm(env.task.goal - after.physics.states.pos[0, 0])
    )
    np.testing.assert_allclose(cost - shaped_cost, [expected], rtol=1e-5)
    np.testing.assert_allclose(terms["progress_reward"], [expected], rtol=1e-5)
    done = before.replace(task=before.task.replace(event=jnp.array([Event.COLLISION])))
    _, _, done_terms = shaped._cost(done, after, action, initialized.objective_state)
    np.testing.assert_array_equal(done_terms["progress_reward"], 0)


def test_height_boundary_cost_is_continuous_and_gated_after_done():
    """Penalize proximity to both height limits without changing existing reward terms."""
    env = Environment(task="navigation", scene="S01", sensor="depth", num_envs=3)
    base = Trainer(env, kind="depth", config={"perception_weight": 0})
    penalized = Trainer(
        env, kind="depth", config={"perception_weight": 0, "height_boundary_weight": 20.0}
    )
    initialized = base.initialize()
    state = initialized.env_state
    positions = state.physics.states.pos.at[:, 0, 2].set(jnp.array([0.6, 3.0, 5.9]))
    state = state.replace(
        physics=state.physics.replace(states=state.physics.states.replace(pos=positions))
    )
    action = jnp.zeros((3, 3))
    cost, _, _ = base._cost(state, state, action, initialized.objective_state)
    changed, _, terms = penalized._cost(state, state, action, initialized.objective_state)
    np.testing.assert_allclose(changed - cost, [0.324, 0.0, 0.324], atol=1e-6)
    np.testing.assert_allclose(terms["height_boundary_cost"], [0.324, 0.0, 0.324], atol=1e-6)

    def boundary_cost(heights):
        positions = state.physics.states.pos.at[:, 0, 2].set(heights)
        after = state.replace(
            physics=state.physics.replace(states=state.physics.states.replace(pos=positions))
        )
        return penalized._cost(state, after, action, initialized.objective_state)[2][
            "height_boundary_cost"
        ].sum()

    np.testing.assert_allclose(
        jax.grad(boundary_cost)(jnp.array([0.6, 3.0, 5.9])), [-0.72, 0.0, 0.72], atol=1e-6
    )
    done = state.replace(task=state.task.replace(event=jnp.full(3, Event.COLLISION)))
    _, _, terms = penalized._cost(done, done, action, initialized.objective_state)
    np.testing.assert_array_equal(terms["height_boundary_cost"], 0)


@pytest.mark.parametrize("kind", ["depth", "lidar"])
def test_optional_ppo_critic_consumes_actor_visible_sensor(kind):
    """The optional value encoder uses measured data; the old value encoder does not."""
    state = jnp.ones((2, 10), dtype=jnp.float32)
    observation = {"state": state}
    changed = {"state": state}
    if kind == "depth":
        observation["depth"] = jnp.zeros((2, 12, 16, 1))
        changed["depth"] = jnp.ones((2, 12, 16, 1))
    else:
        observation["points"] = jnp.ones((2, 8, 3))
        changed["points"] = jnp.full((2, 8, 3), 2.0)
        observation["mask"] = changed["mask"] = jnp.ones((2, 8), dtype=bool)
    critic = Critic(kind=kind)
    variables = critic.init(jax.random.PRNGKey(5), observation)
    assert critic.apply(variables, observation).shape == (2,)
    assert not np.array_equal(
        critic.apply(variables, observation), critic.apply(variables, changed)
    )
    state_critic = Critic()
    default_variables = state_critic.init(jax.random.PRNGKey(5), observation)
    np.testing.assert_array_equal(
        state_critic.apply(default_variables, observation),
        state_critic.apply(default_variables, changed),
    )


def test_sensor_critic_performs_real_depth_ppo_update():
    """Train the independent visual value function through the PPO update path."""
    env = Environment(task="navigation", scene="S01", sensor="depth", num_envs=1, duration=0.02)
    trainer = Trainer(
        env,
        kind="depth",
        algorithm="ppo",
        config={
            "horizon": 1,
            "ppo_epochs": 1,
            "minibatches": 1,
            "critic_uses_sensor": True,
        },
    )
    before = trainer.initialize()
    after, metrics = trainer.update(before)
    assert np.isfinite(metrics["critic_loss"])
    assert np.isfinite(metrics["policy_loss"])
    assert tree_changed(before.critic_params, after.critic_params)


@pytest.mark.parametrize("kind", ["depth", "lidar"])
def test_goal_observation_ppo_recipe_updates_and_restores(kind, tmp_path):
    """Train the expanded goal input with immediate velocity and continuous height cost."""
    env = Environment(
        task="navigation",
        scene="S01",
        sensor=kind,
        num_envs=2,
        duration=0.02,
        navigation_goal_observation=True,
    )
    trainer = Trainer(
        env,
        kind=kind,
        config={
            "horizon": 2,
            "ppo_epochs": 1,
            "critic_uses_sensor": True,
            "progress_reward_scale": 0.5,
            "height_boundary_weight": 20.0,
            "perception_loss": {"velocity_window": 1},
        },
    )
    state = trainer.initialize()
    assert env.observe(state.env_state)["state"].shape == (2, 12)
    updated, metrics = trainer.update(state)
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert tree_changed(state.params, updated.params)
    assert tree_changed(state.critic_params, updated.critic_params)
    path = tmp_path / "goal.training.zip"
    trainer.save_state(path, updated, provenance={})
    restored, _ = trainer.load_state(path)
    assert_tree_equal(updated, restored)


def test_optional_ppo_critic_rejects_unsupported_actor_or_algorithm():
    """Keep pathwise value gradients and original state-policy contracts unchanged."""
    env = Environment(num_envs=1)
    with pytest.raises(ValueError, match="Sensor-aware critic"):
        Trainer(env, config={"critic_uses_sensor": True})
    with pytest.raises(ValueError, match="Navigation"):
        Trainer(env, config={"progress_reward_scale": 0.5})


def test_privileged_critic_changes_value_without_changing_actor():
    """Keep true geometry and physical state out of the deployed actor inputs."""
    env = Environment(task="navigation", scene="D03", sensor="depth", num_envs=1)
    trainer = Trainer(env, kind="depth", config={"critic_uses_privileged": True})
    state = trainer.initialize()
    observation = trainer._observe(state.env_state)
    assert np.isfinite(observation["privileged_state"]).all()
    changed = {**observation, "privileged_state": observation["privileged_state"] + 1.0}
    memory = state.recurrent_memory
    for before, after in zip(
        jax.tree.leaves(trainer.actor.apply(state.params, observation, memory)),
        jax.tree.leaves(trainer.actor.apply(state.params, changed, memory)),
        strict=True,
    ):
        np.testing.assert_array_equal(before, after)
    assert not np.array_equal(
        trainer.critic.apply(state.critic_params, observation),
        trainer.critic.apply(state.critic_params, changed),
    )
    default = Trainer(env, kind="depth")
    assert "privileged_state" not in default._observe(state.env_state)
    later = state.env_state.replace(
        physics=state.env_state.physics.replace(
            core=state.env_state.physics.core.replace(
                steps=state.env_state.physics.core.steps + 500
            )
        )
    )
    for old, new in zip(
        jax.tree.leaves(env.observe(state.env_state)),
        jax.tree.leaves(env.observe(later)),
        strict=True,
    ):
        np.testing.assert_array_equal(old, new)
    assert not np.array_equal(
        env.observe_privileged(state.env_state), env.observe_privileged(later)
    )


def test_asymmetric_ppo_real_update_and_recovery(tmp_path):
    """Train and restore the separate value input, including pre-reset bootstrap."""
    env = Environment(task="navigation", scene="S01", sensor="depth", num_envs=2, duration=0.02)
    trainer = Trainer(
        env,
        kind="depth",
        config={"critic_uses_privileged": True, "horizon": 2, "ppo_epochs": 1},
    )
    before = trainer.initialize()
    after, metrics = trainer.update(before)
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["critic_loss"])
    assert tree_changed(before.params, after.params)
    assert tree_changed(before.critic_params, after.critic_params)
    path = tmp_path / "asymmetric.training.zip"
    trainer.save_state(path, after, provenance={})
    restored, _ = trainer.load_state(path)
    assert_tree_equal(after, restored)
    inference = tmp_path / "asymmetric.policy.zip"
    trainer.save_inference(inference, after, provenance={})
    saved_actor = load_inference(inference)
    actor, params = Actor(kind=saved_actor["kind"]), saved_actor["params"]
    measured = env.observe(after.env_state)
    expected = trainer.actor.apply(after.params, measured, after.recurrent_memory)
    actual = actor.apply(params, measured, after.recurrent_memory)
    for saved, loaded in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(saved, loaded)


@pytest.mark.parametrize(
    ("algorithm", "options"),
    [("shac", {}), ("ppo", {"critic_uses_sensor": True}), ("ppo", {"critic_uses_privileged": 1})],
)
def test_privileged_critic_rejects_ambiguous_or_unsupported_configuration(algorithm, options):
    """Require a boolean choice of one PPO critic observation source."""
    env = Environment(task="navigation", scene="S01", sensor="depth")
    with pytest.raises(ValueError, match=r"[Pp]rivileged|[Cc]ritic"):
        Trainer(
            env,
            kind="depth",
            algorithm=algorithm,
            config={"critic_uses_privileged": True, **options},
        )


def test_integer_log_standard_deviation_allows_ppo_update():
    """Accept numerically valid integer options without creating integer parameters."""
    trainer = Trainer(
        Environment(num_envs=2),
        algorithm="ppo",
        config={"horizon": 1, "ppo_epochs": 1, "initial_log_std": -3},
    )
    state, metrics = trainer.update(trainer.initialize())
    assert np.isfinite(metrics["actor_loss"])
    assert jnp.issubdtype(state.log_std.dtype, jnp.floating)


def assert_tree_equal(left, right):
    """Provide assert tree equal for the surrounding execution."""
    assert jax.tree.structure(left) == jax.tree.structure(right)
    for actual, expected in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if isinstance(actual, jax.Array) and jax.dtypes.issubdtype(
            actual.dtype, jax.dtypes.prng_key
        ):
            actual, expected = jax.random.key_data(actual), jax.random.key_data(expected)
        np.testing.assert_array_equal(actual, expected)


def tree_changed(left, right):
    """Provide tree changed for the surrounding execution."""
    return any(
        not np.array_equal(x, y)
        for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


@pytest.fixture(scope="module")
def state_env():
    """Provide state env for the surrounding execution."""
    return Environment(num_envs=2)


def test_gae_termination_truncation_and_rollout_boundary():
    """Verify gae termination truncation and rollout boundary."""
    rewards = jnp.array([[1.0, 1.0, 1.0], [4.0, 4.0, 4.0]])
    values = jnp.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    next_values = jnp.array([[10.0, 10.0, 10.0], [5.0, 5.0, 5.0]])
    terminated = jnp.array([[True, False, False], [False, False, False]])
    done = jnp.array([[True, True, False], [False, False, False]])
    advantages, targets = generalized_advantage_estimate(
        rewards, values, next_values, terminated, done, gamma=0.9, gae_lambda=0.8
    )
    # Second row: delta=4+.9*5-3=5.5; traces stop on both kinds of reset.
    expected = [[-1.0, 8.0, 8.0 + 0.9 * 0.8 * 5.5], [5.5, 5.5, 5.5]]
    np.testing.assert_allclose(advantages, expected, rtol=1e-6)
    np.testing.assert_allclose(targets, np.asarray(expected) + values, rtol=1e-6)
    np.testing.assert_allclose(
        normalize_advantages(jnp.array([1.0, 2.0, 8.0])).std(), 1.0, rtol=1e-6
    )
    np.testing.assert_array_equal(normalize_advantages(jnp.ones(4)), 0)


def test_navigation_altitude_cost_matches_physical_height_error():
    """Verify navigation altitude cost matches physical height error."""
    env = Environment(task="navigation", scene="S01", sensor="depth", num_envs=1)
    base = Trainer(env, kind="depth", config={"horizon": 1, "perception_weight": 0})
    height = Trainer(
        env, kind="depth", config={"horizon": 1, "perception_weight": 0, "altitude_weight": 2}
    )
    initial = base.initialize()
    physics = initial.env_state.physics
    physics = physics.replace(
        states=physics.states.replace(pos=physics.states.pos.at[0, 0, 2].set(2))
    )
    initial = initial.replace(env_state=initial.env_state.replace(physics=physics))
    plain, _ = rollout_objective(base, initial, initial.params)
    weighted, (after, _, _) = rollout_objective(height, initial, initial.params)
    expected = 2 * env.dt * (after.env_state.physics.states.pos[0, 0, 2] - 3) ** 2
    np.testing.assert_allclose(weighted - plain, expected, rtol=1e-5)


def test_recurrent_recomputation_resets_memory_and_keeps_gradients():
    """Verify recurrent recomputation resets memory and keeps gradients."""

    class AccumulatingActor:
        def apply(self, params, obs, memory):
            memory = memory + params * obs["state"]
            return memory, memory, jnp.zeros_like(memory)

    trainer = SimpleNamespace(actor=AccumulatingActor())
    observations = {"state": jnp.ones((3, 2, 1))}
    done = jnp.array([[True, False], [False, False], [False, True]])
    initial = jnp.array([[5.0], [7.0]])
    final, (raw, _) = recurrent_policy(trainer, 2.0, observations, initial, done)
    np.testing.assert_array_equal(raw[..., 0], [[7.0, 9.0], [2.0, 11.0], [4.0, 13.0]])
    np.testing.assert_array_equal(final, [[4.0], [0.0]])
    gradient = jax.grad(
        lambda p: recurrent_policy(trainer, p, observations, initial, done)[1][0].sum()
    )(2.0)
    assert float(gradient) == 10.0


@pytest.mark.parametrize("algorithm", ["ppo", "apg", "shac"])
def test_real_physics_update_and_exact_checkpoint_continuation(state_env, tmp_path, algorithm):
    """Verify real physics update and exact checkpoint continuation."""
    trainer = Trainer(
        state_env,
        algorithm=algorithm,
        config={
            "horizon": 2,
            "ppo_epochs": 1,
            "critic_epochs": 1,
            "minibatches": 2,
        },
    )
    initial = trainer.initialize()
    state, metrics = trainer.update(initial)
    assert all(np.isfinite(value).all() for value in metrics.values())
    assert float(metrics["actor_grad_norm"]) > 0
    assert tree_changed(initial.params, state.params)
    assert tree_changed(initial.optimizer_state["actor"], state.optimizer_state["actor"])
    assert not np.array_equal(initial.rng, state.rng)
    assert np.all(np.asarray(state.env_state.time) > 0)
    assert state.updates == 1 and state.interactions == 4
    if algorithm != "apg":
        assert tree_changed(initial.critic_params, state.critic_params)
    if algorithm == "ppo":
        assert not np.array_equal(initial.log_std, state.log_std)
    if algorithm == "shac":
        expected = update_target(
            initial.target_critic_params, state.critic_params, trainer.config["target_alpha"]
        )
        for a, b in zip(
            jax.tree.leaves(expected), jax.tree.leaves(state.target_critic_params), strict=True
        ):
            np.testing.assert_allclose(a, b, atol=1e-7)
    path = tmp_path / f"{algorithm}.checkpoint"
    trainer.save_state(path, state, provenance={"test": "official Crazyflow"})
    restored, metadata = trainer.load_state(path)
    assert metadata["config"]["learning"] == trainer.resolved_config
    assert_tree_equal(state, restored)
    uninterrupted, expected_metrics = trainer.update(state)
    continued, actual_metrics = trainer.update(restored)
    assert_tree_equal(uninterrupted, continued)
    assert_tree_equal(expected_metrics, actual_metrics)
    export = tmp_path / "inference.checkpoint"
    trainer.save_inference(export, state, provenance={"test": algorithm})
    frozen = load_inference(export)
    observation = state_env.observe(state.env_state)
    expected = trainer.actor.apply(state.params, observation, state.recurrent_memory)
    actual = trainer.actor.apply(frozen["params"], observation, state.recurrent_memory)
    assert_tree_equal(expected, actual)
    assert frozen["kind"] == "state"
    # A long run must not wrap its counters at the default JAX int32 boundary.
    long_state = state.replace(updates=2**31, interactions=2**32)
    result, _ = trainer.update(long_state)
    assert result.updates == 2**31 + 1 and result.interactions == 2**32 + 4


def test_bootstrap_uses_final_observation_and_resets_recurrent_state():
    """Verify bootstrap uses final observation and resets recurrent state."""
    env = Environment(num_envs=2)
    real_step = env.step

    def truncate(state, action):
        after = real_step(state, action)
        return after.replace(task=after.task.replace(event=jnp.full(2, Event.TIMEOUT)))

    env.step = truncate
    trainer = Trainer(env, config={"horizon": 1})
    initial = trainer.initialize().replace(recurrent_memory=jnp.ones((2, 192)))
    state, rollout, _ = collect_rollout(trainer, initial)
    final = real_step(initial.env_state, jnp.tanh(rollout.pre_tanh[0]))
    expected = trainer.critic.apply(initial.critic_params, env.observe(final))
    np.testing.assert_allclose(rollout.next_values[0], expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(state.recurrent_memory, 0)
    np.testing.assert_array_equal(state.objective_state.count, 0)
    np.testing.assert_array_equal(state.env_state.time, 0)
    assert np.asarray(rollout.done).all() and not np.asarray(rollout.terminated).any()


@pytest.mark.parametrize("event", [Event.RUNNING, Event.TIMEOUT, Event.COLLISION])
def test_shac_terminal_value_state_gradient(event):
    """Verify shac terminal value state gradient."""
    env = Environment(num_envs=1)
    real_step = env.step

    def event_step(state, action):
        after = real_step(state, action)
        return after.replace(task=after.task.replace(event=jnp.full(1, event)))

    env.step = event_step
    trainer = Trainer(env, algorithm="shac", config={"horizon": 1, "task_weight": 0.0})
    state = trainer.initialize()
    gradient = jax.jit(
        jax.grad(lambda params: rollout_objective(trainer, state, params, bootstrap=True)[0])
    )(state.params)
    norm = np.sqrt(sum(float(jnp.sum(x * x)) for x in jax.tree.leaves(gradient)))
    if event == Event.COLLISION:
        assert norm == 0
    else:
        assert norm > 1e-7


def test_temporal_decay_affects_only_incoming_physics_derivative(state_env):
    """Verify temporal decay affects only incoming physics derivative."""
    normal = Trainer(state_env, algorithm="apg", config={"temporal_gradient_alpha": 0.0})
    decayed = Trainer(state_env, algorithm="apg", config={"temporal_gradient_alpha": 5.0})
    state = state_env.reset(jax.random.PRNGKey(1))
    action = jnp.zeros((2, 4))
    a = normal._step(state, action, differentiable=True)
    b = decayed._step(state, action, differentiable=True)
    assert_tree_equal(a, b)

    def output(trainer, offset, action):
        physics = state.physics.replace(
            states=state.physics.states.replace(pos=state.physics.states.pos + offset)
        )
        return trainer._step(
            state.replace(physics=physics), action, differentiable=True
        ).physics.states.pos.sum()

    incoming, direct = jax.jit(jax.grad(lambda x, a: output(normal, x, a), (0, 1)))(0.0, action)
    damped, unaffected = jax.jit(jax.grad(lambda x, a: output(decayed, x, a), (0, 1)))(0.0, action)
    np.testing.assert_allclose(damped, incoming * np.exp(-5 * state_env.dt), rtol=2e-6)
    np.testing.assert_allclose(unaffected, direct, rtol=2e-6, atol=1e-8)
    assert float(jnp.linalg.norm(direct)) > 0


def test_checkpoint_validation_and_atomic_failure(state_env, tmp_path, monkeypatch):
    """Verify checkpoint validation and atomic failure."""
    trainer = Trainer(state_env)
    state = trainer.initialize()
    path = tmp_path / "state.checkpoint"
    save_state(path, state, config=trainer.resolved_config, provenance={"source": "test"})
    original = path.read_bytes()
    import drone_playground.learning.checkpoint as checkpoint

    def fail_replace(*args):
        raise OSError("simulated interrupted replacement")

    monkeypatch.setattr(checkpoint.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        save_state(path, state, config=trainer.resolved_config, provenance={"source": "test"})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(ValueError, match="shape or dtype"):
        load_state(path, state.replace(log_std=jnp.zeros(3)))
    with pytest.raises(ValueError, match="purpose"):
        load_inference(path)


@pytest.mark.parametrize(
    "config",
    [
        {"horizon": 0},
        {"lr": float("nan")},
        {"minibatches": 3},
        {"unknown_option": 2},
        {"velocity_aux_weight": 1},
    ],
)
def test_invalid_configuration_is_rejected(state_env, config):
    """Verify invalid configuration is rejected."""
    with pytest.raises(ValueError):
        Trainer(state_env, config=config)


@pytest.fixture(scope="module", params=["depth", "lidar"])
def perception_env(request):
    # Small LiDAR acquisition keeps this a CPU integration check; camera shape is unchanged.
    """Provide perception env for the surrounding execution."""
    sensor_config = {"points_per_frame": 32} if request.param == "lidar" else {}
    env = Environment(
        task="navigation",
        scene="S01",
        num_envs=1,
        sensor=request.param,
        sensor_config=sensor_config,
        point_count=32,
    )
    return request.param, env


@pytest.mark.parametrize("algorithm", ["ppo", "apg", "shac"])
def test_perception_real_update_and_complete_sensor_checkpoint(perception_env, algorithm, tmp_path):
    """Verify perception real update and complete sensor checkpoint."""
    kind, env = perception_env
    horizon = 6 if kind == "lidar" else 2
    trainer = Trainer(
        env,
        kind=kind,
        algorithm=algorithm,
        config={
            "horizon": horizon,
            "ppo_epochs": 1,
            "critic_epochs": 1,
            "velocity_aux_weight": 0.1 if kind == "depth" else 0.0,
            "temporal_gradient_alpha": 1.0,
        },
    )
    initial = trainer.initialize()
    state, metrics = trainer.update(initial)
    assert all(np.isfinite(value).all() for value in metrics.values())
    assert float(metrics["actor_grad_norm"]) > 0
    assert float(metrics["perception_cost"]) > 0
    assert tree_changed(initial.params, state.params)
    assert np.linalg.norm(state.recurrent_memory) > 0
    assert state.env_state.observation is not None
    assert np.all(np.asarray(state.objective_state.count) == horizon)
    assert int(state.env_state.observation.frame[0]) > 0
    assert np.asarray(state.env_state.observation.measurement.mask).any()
    path = tmp_path / "perception.dp"
    trainer.save_state(path, state, provenance={"integration": kind})
    restored, _ = trainer.load_state(path)
    assert_tree_equal(state, restored)
    resumed, actual = trainer.update(restored)
    continued, expected = trainer.update(state)
    assert_tree_equal(resumed, continued)
    assert_tree_equal(actual, expected)
    export = tmp_path / "perception-inference.dp"
    trainer.save_inference(export, state, provenance={"integration": kind})
    frozen = load_inference(export)
    observation = env.observe(state.env_state)
    assert_tree_equal(
        trainer.actor.apply(state.params, observation, state.recurrent_memory),
        trainer.actor.apply(frozen["params"], observation, state.recurrent_memory),
    )


def test_common_named_objective_weights_and_history(perception_env):
    """Verify common named objective weights and history."""
    from drone_playground.learning.losses import liu_loss, zhang_loss

    kind, env = perception_env
    trainer = Trainer(
        env,
        kind=kind,
        algorithm="apg",
        config={
            "task_weight": 0.3,
            "perception_weight": 1.7,
        },
    )
    initial = trainer.initialize()
    action = jnp.array([[0.2, -0.1, 0.05]])
    before = initial.env_state
    after = env.step(before, action)
    cost, history, terms = trainer._cost(before, after, action, initial.objective_state)
    velocity = after.physics.states.vel[:, 0]
    delta = env.task.goal - after.physics.states.pos[:, 0]
    distance = jnp.linalg.norm(delta, axis=-1, keepdims=True)
    target = delta / jnp.maximum(distance, 1e-6) * jnp.minimum(distance, 3.0)
    clearance = env.scene.clearance(after.physics.states.pos[:, 0], after.time) - env.task.radius
    old = env.scene.clearance(before.physics.states.pos[:, 0], before.time) - env.task.radius
    kwargs = {**trainer.config["perception_loss"], "velocity_window": 1}
    recipe = zhang_loss if kind == "depth" else liu_loss
    expected, _ = recipe(
        velocity[None],
        target[None],
        (action * 6.0)[None],
        clearance[None],
        ((old - clearance) / env.dt)[None],
        dt=env.dt,
        **kwargs,
    )
    np.testing.assert_allclose(
        cost, 0.3 * env.cost(before, after, action) + 1.7 * env.dt * expected, rtol=1e-6
    )
    np.testing.assert_array_equal(terms["named_jerk"], 0)
    for algorithm in ("ppo", "shac"):
        other = Trainer(
            env,
            kind=kind,
            algorithm=algorithm,
            config={
                "task_weight": 0.3,
                "perception_weight": 1.7,
            },
        )
        other_cost, _, _ = other._cost(before, after, action, initial.objective_state)
        np.testing.assert_array_equal(other_cost, cost)
    next_action = action * -0.5
    next_state = env.step(after, next_action)
    _, next_history, next_terms = trainer._cost(after, next_state, next_action, history)
    acceleration = next_action * 6.0
    expected_acceleration = jnp.sum(acceleration**2, axis=-1)
    np.testing.assert_allclose(next_terms["named_acceleration"], expected_acceleration, rtol=1e-6)
    assert float(next_terms["named_jerk"][0]) > 0
    assert int(next_history.count[0]) == 2
    if kind == "lidar":
        third_action = action * 0.25
        third_state = env.step(next_state, third_action)
        _, _, third_terms = trainer._cost(next_state, third_state, third_action, next_history)
        magnitudes = jnp.stack(
            (
                jnp.linalg.norm((next_action - action) * 6.0 / env.dt, axis=-1),
                jnp.linalg.norm((third_action - next_action) * 6.0 / env.dt, axis=-1),
            )
        )
        np.testing.assert_allclose(
            third_terms["named_jerk_variance"], jnp.var(magnitudes, axis=0), rtol=1e-6
        )
    finished = next_state.replace(task=next_state.task.replace(event=jnp.array([Event.TIMEOUT])))
    _, memory, history = trainer._reset(
        jax.random.PRNGKey(5), finished, jnp.ones((1, 192)), next_history
    )
    np.testing.assert_array_equal(memory, 0)
    for leaf in jax.tree.leaves(history):
        np.testing.assert_array_equal(leaf, 0)
