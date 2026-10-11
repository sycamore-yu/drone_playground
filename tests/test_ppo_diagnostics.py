"""Training-only PPO diagnostic invariants, independent of frozen outcomes."""

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.learning.ppo import (
    advantage_progress_correlation,
    completed_navigation_metrics,
)
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.environment import make_environment


def test_advantage_progress_detects_positive_negative_and_zero_variance():
    """A diagnostic must distinguish correlation from critic loss magnitude."""
    advantage = jnp.array([-2.0, -1.0, 1.0, 2.0])
    progress = jnp.array([-0.2, -0.1, 0.1, 0.2])
    np.testing.assert_allclose(
        jax.jit(advantage_progress_correlation)(advantage, progress), 1.0, atol=1e-6
    )
    np.testing.assert_allclose(
        advantage_progress_correlation(advantage, -progress), -1.0, atol=1e-6
    )
    assert float(advantage_progress_correlation(advantage, progress * 0)) == 0.0


def test_completed_episode_rates_use_episodes_not_rollout_steps():
    """One success among two completed episodes is a 50% training success rate."""
    terms = {
        "navigation_success_fraction": jnp.array([[0, 0], [1, 0]], dtype=jnp.float32),
        "navigation_collision_fraction": jnp.array([[0, 1], [0, 0]], dtype=jnp.float32),
        "navigation_out_of_bounds_fraction": jnp.zeros((2, 2)),
    }
    done = jnp.array([[False, True], [True, False]])
    metrics = jax.jit(completed_navigation_metrics)(done, terms)
    assert float(metrics["navigation_completed_episodes"]) == 2.0
    assert float(metrics["navigation_completed_success_rate"]) == 0.5
    assert float(metrics["navigation_completed_collision_rate"]) == 0.5
    empty = completed_navigation_metrics(
        jnp.zeros_like(done), {k: jnp.zeros_like(v) for k, v in terms.items()}
    )
    assert float(empty["navigation_completed_success_rate"]) == 0.0


def test_ppo_navigation_diagnostics_are_finite_on_real_simulation():
    """Return and step-progress evidence come from the same on-policy rollout."""
    env = make_environment(
        task="navigation",
        scene="S01",
        sensor="depth",
        sensor_config={"width": 64, "height": 48, "pitch_deg": 20, "max_range": 10},
        num_envs=2,
        device="cpu",
        duration=0.2,
        action={"level": "acceleration", "heading": "target"},
    )
    trainer = Trainer(
        env,
        kind="depth",
        algorithm="ppo",
        config={
            "horizon": 1,
            "ppo_epochs": 1,
            "critic_epochs": 1,
            "minibatches": 1,
        },
    )
    _, metrics = trainer.update(trainer.initialize())
    for name in (
        "navigation_progress_metres",
        "navigation_remaining_metres",
        "navigation_collision_fraction",
        "navigation_out_of_bounds_fraction",
        "navigation_success_fraction",
        "navigation_completed_episodes",
        "navigation_completed_success_rate",
        "navigation_completed_collision_rate",
        "navigation_completed_out_of_bounds_rate",
        "navigation_completed_timeout_rate",
        "navigation_completed_numerical_failure_rate",
        "navigation_completed_successes",
        "navigation_success_mean_flight_seconds",
        "navigation_completed_min_clearance_mean_metres",
        "advantage_progress_correlation",
        "advantage_positive_fraction",
        "critic_return_rmse_preupdate",
        "depth_valid_fraction",
    ):
        assert name in metrics
        assert np.isfinite(np.asarray(metrics[name])).all(), name
