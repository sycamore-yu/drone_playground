"""Complete on-policy tails replace bootstrap without advancing live learner worlds."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from drone_playground.learning.ppo import generalized_advantage_estimate, monte_carlo_tail
from drone_playground.learning.trainer import Trainer, TrainingState
from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.tasks import Event


@struct.dataclass
class TaskState:
    """Terminal event in the analytic clock task."""

    event: jax.Array


@struct.dataclass
class ClockState:
    """Minimal batched episode state with a finite deadline."""

    time: jax.Array
    task: TaskState

    @property
    def done(self):
        """Stop physical advancement after a terminal event."""
        return self.task.event != Event.RUNNING


def test_complete_tails_mask_terminals_preserve_live_state_and_stop_gradients():
    """An analytic two-world task checks returns, padding and finite-horizon GAE."""
    state = TrainingState(
        params={"bias": jnp.asarray(0.0)},
        critic_params=None,
        target_critic_params=None,
        optimizer_state=None,
        rng=jax.random.key(7),
        env_state=ClockState(jnp.array([0.0, 1.0]), TaskState(jnp.zeros(2, jnp.int32))),
        recurrent_memory=jnp.zeros((2, 1)),
        objective_state=None,
        updates=0,
        interactions=0,
        log_std=jnp.zeros(1),
    )

    def step(current, action, *, differentiable):
        assert not differentiable
        time = current.time + (~current.done)
        event = jnp.where(~current.done & (time >= 3), Event.SUCCESS, current.task.event)
        return ClockState(time, TaskState(event))

    trainer = SimpleNamespace(
        env=SimpleNamespace(task=SimpleNamespace(duration=3.0), dt=1.0, num_envs=2),
        actor=SimpleNamespace(
            apply=lambda params, obs, memory: (jnp.full((2, 1), params["bias"]), memory + 1, None)
        ),
        method=SimpleNamespace(sample=lambda key, raw, std: (raw, raw, jnp.zeros(2))),
        objective=SimpleNamespace(
            cost=lambda before, after, action, history: (
                -(1 + action[:, 0] + 10 * after.done),
                history,
                {},
            )
        ),
        _observe=lambda current: {"time": current.time},
        _step=step,
    )
    active = jnp.array([False, True])
    returns, metrics = jax.jit(lambda s: monte_carlo_tail(trainer, s, active))(state)
    np.testing.assert_array_equal(returns, [0, 12])
    assert int(metrics["monte_carlo_tail_interactions"]) == 2
    assert int(metrics["monte_carlo_tail_allocated_slots"]) == 4
    assert int(metrics["monte_carlo_tail_unfinished"]) == 0
    assert int(metrics["monte_carlo_tail_successes"]) == 1
    np.testing.assert_array_equal(state.env_state.task.event, [Event.RUNNING] * 2)
    np.testing.assert_array_equal(state.env_state.time, [0, 1])
    np.testing.assert_array_equal(
        jax.random.key_data(state.rng), jax.random.key_data(jax.random.key(7))
    )
    gradient = jax.grad(
        lambda bias: monte_carlo_tail(trainer, state.replace(params={"bias": bias}), active)[
            0
        ].sum()
    )(jnp.asarray(0.0))
    assert float(gradient) == 0
    empty, metrics = monte_carlo_tail(trainer, state, jnp.zeros_like(active))
    np.testing.assert_array_equal(empty, [0, 0])
    assert int(metrics["monte_carlo_tail_interactions"]) == 0
    rewards = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    values = jnp.array([[7.0, 8.0], [9.0, 10.0]])
    next_values = jnp.array([[9.0, 10.0], [123.0, 999.0]]).at[-1].set(returns)
    done = jnp.array([[False, False], [True, False]])
    _, targets = generalized_advantage_estimate(
        rewards, values, next_values, done, done, gamma=1, gae_lambda=1
    )
    np.testing.assert_array_equal(targets, [[4, 18], [3, 16]])


@pytest.mark.parametrize(
    "algorithm,finite,options",
    [
        ("ppo", True, {"ppo_monte_carlo_returns": "yes"}),
        ("shac", True, {}),
        ("ppo", False, {}),
        ("ppo", True, {"gamma": 0.99}),
        ("ppo", True, {"gae_lambda": 0.95}),
    ],
)
def test_monte_carlo_requires_finite_undiscounted_ppo(algorithm, finite, options):
    """Reject estimators whose return definition differs from the implemented tail sum."""
    env = SimpleNamespace(task=SimpleNamespace(name="navigation", finite_horizon=finite))
    with pytest.raises(ValueError, match=r"Monte Carlo|must be boolean"):
        Trainer(
            env,
            algorithm=algorithm,
            config={"ppo_monte_carlo_returns": True, "gamma": 1, "gae_lambda": 1, **options},
        )


def test_real_update_counts_tail_interactions_and_preserves_resumable_live_stream(tmp_path):
    """Extra target sampling costs are counted, while live physical time advances one step."""
    env = make_environment(
        task="navigation",
        scene="S01",
        num_envs=2,
        method_hz=10,
        duration=0.3,
        task_config={"reward_profile": "goal_reaching", "finite_horizon": True},
        action={"level": "acceleration", "heading": "target"},
    )
    try:
        trainer = Trainer(
            env,
            config={
                "horizon": 1,
                "ppo_epochs": 1,
                "minibatches": 1,
                "gamma": 1,
                "gae_lambda": 1,
                "ppo_monte_carlo_returns": True,
            },
        )
        initial = trainer.initialize()
        state, metrics = trainer.update(initial)
        assert int(metrics["monte_carlo_tail_interactions"]) == 4
        assert state.interactions == 6 and state.updates == 1
        np.testing.assert_allclose(state.env_state.time, 0.1, atol=1e-5)
        assert all(np.isfinite(v).all() for v in metrics.values())
        archive = tmp_path / "state.zip"
        trainer.save_state(archive, state, provenance={})
        restored, _ = trainer.load_state(archive)
        expected, _ = trainer.update(state)
        actual, _ = trainer.update(restored)
        assert expected.interactions == actual.interactions == 10
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            if hasattr(a, "dtype") and jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
                a, b = jax.random.key_data(a), jax.random.key_data(b)
            np.testing.assert_array_equal(a, b)
    finally:
        env.sim.close()
