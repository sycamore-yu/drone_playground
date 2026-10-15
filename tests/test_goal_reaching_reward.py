"""Goal-arrival ordering, finite-horizon bootstrapping and physical gradients."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.learning.apg import rollout_objective
from drone_playground.learning.checkpoint import load_state, save_state
from drone_playground.learning.ppo import collect_rollout, generalized_advantage_estimate
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.tasks import Event


def _environment(**kwargs):
    return make_environment(
        task="navigation",
        scene="S01",
        method_hz=10,
        task_config={"reward_profile": "goal_reaching", "finite_horizon": True},
        **kwargs,
    )


def test_safe_arrival_outranks_early_failure_and_hovering():
    """Full undiscounted returns prefer even a slow, costly success to failure."""
    from drone_playground.simulation.task_rewards import goal_reaching_reward

    env = _environment(num_envs=1)
    try:
        # Include a failure arbitrarily close to the goal, plus a full-deadline hover.
        steps = 3000
        lengths = jnp.array([2999, 960, 2999, 3000])
        final_distance = jnp.array([0.5, 40.0, 0.0, 96.0])
        events = jnp.array([Event.SUCCESS, Event.COLLISION, Event.COLLISION, Event.TIMEOUT])
        goal = env.task.goal

        def state(index):
            fraction = jnp.minimum(index / lengths, 1.0)
            remaining = 96.0 + (final_distance - 96.0) * fraction
            position = jnp.broadcast_to(goal, (4, 3)).at[:, 0].add(-remaining)
            event = jnp.where(index >= lengths, events, Event.RUNNING)
            return SimpleNamespace(
                physics=SimpleNamespace(states=SimpleNamespace(pos=position[:, None])),
                task=SimpleNamespace(event=event),
                done=event != Event.RUNNING,
                time=jnp.minimum(index, lengths) * env.dt,
                scene_state=None,
            )

        def reward(index):
            return goal_reaching_reward(env, state(index), state(index + 1), jnp.ones((4, 3)))[0]

        returns = jax.jit(lambda: jax.vmap(reward)(jnp.arange(steps)).sum(0))()
        assert np.isfinite(returns).all()
        assert float(returns[0] - returns[1:].max()) > 1.45
        # Distance progress telescopes; repeated backwards/forwards motion adds no progress.
        before, after = state(0), state(1)
        _, forward = goal_reaching_reward(env, before, after, jnp.zeros((4, 3)))
        _, backward = goal_reaching_reward(env, after, before, jnp.zeros((4, 3)))
        np.testing.assert_allclose(forward["progress_reward"], -backward["progress_reward"])
    finally:
        env.sim.close()


@pytest.mark.parametrize("algorithm", ["ppo", "apg", "shac"])
def test_finite_horizon_timeout_stops_bootstrap_and_exposes_remaining_time(algorithm, tmp_path):
    """A task deadline is terminal, while a rollout cut still bootstraps."""
    env = _environment(num_envs=1, duration=0.1)
    try:
        trainer = Trainer(
            env,
            algorithm=algorithm,
            config={
                "gamma": 1.0,
                "horizon": 1,
                "perception_weight": 0.0,
                "ppo_epochs": 1,
                "critic_epochs": 1,
            },
        )
        initial = trainer.initialize()
        assert float(env.observe(initial.env_state)["state"][0, -1]) == 1.0
        state, rollout, _ = collect_rollout(trainer, initial)
        assert bool(rollout.done[0, 0]) and bool(rollout.terminated[0, 0])
        _, targets = generalized_advantage_estimate(
            rollout.rewards,
            rollout.values,
            jnp.full_like(rollout.values, 12345.0),
            rollout.terminated,
            rollout.done,
            gamma=1.0,
        )
        np.testing.assert_allclose(targets, rollout.rewards, atol=1e-6)
        assert state.env_state.finite_horizon
        assert float(env.observe(state.env_state)["state"][0, -1]) == 1.0
        final = env.step(initial.env_state, jnp.zeros((1, env.action_size)))
        assert bool(final.terminated[0]) and not bool(final.truncated[0])
        assert float(env.observe(final)["state"][0, -1]) == 0.0
        if algorithm == "shac":
            ordinary = rollout_objective(trainer, initial, initial.params, bootstrap=False)[0]
            bootstrapped = rollout_objective(trainer, initial, initial.params, bootstrap=True)[0]
            np.testing.assert_allclose(ordinary, bootstrapped, atol=1e-6)
        updated, metrics = trainer.update(initial)
        assert all(np.isfinite(value).all() for value in metrics.values())
        path = tmp_path / "training.zip"
        save_state(path, updated, config=trainer.resolved_config, provenance={})
        restored, _ = load_state(path, trainer.initialize())
        assert restored.env_state.finite_horizon
        expected, _ = trainer.update(updated)
        continued, _ = trainer.update(restored)
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(continued), strict=True):
            if hasattr(a, "dtype") and jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
                a, b = jax.random.key_data(a), jax.random.key_data(b)
            np.testing.assert_array_equal(a, b)
    finally:
        env.sim.close()


def test_goal_reward_has_finite_physical_action_gradient_and_no_post_terminal_reward():
    """The shared task reward supplies a real Crazyflow gradient to APG/SHAC."""
    env = _environment(num_envs=1)
    try:
        before = env.reset(jax.random.key(2))
        action = jnp.zeros((1, env.action_size))
        gradient = jax.jit(
            jax.grad(lambda a: env.task_reward_components(before, env.step(before, a), a)[0].sum())
        )(action)
        assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 1e-8
        done = before.replace(task=before.task.replace(event=jnp.array([Event.SUCCESS])))
        cost, terms = env.task_reward_components(done, done, action)
        assert float(cost[0]) == 0.0
        assert all(float(value[0]) == 0.0 for value in terms.values())
        with pytest.raises(ValueError, match="gamma=1"):
            Trainer(env, config={"gamma": 0.99})
    finally:
        env.sim.close()
