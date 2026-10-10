"""Training-only PPO diagnostic invariants, independent of frozen outcomes."""

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.learning.ppo import advantage_progress_correlation
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


def test_ppo_navigation_diagnostics_are_finite_on_real_simulation():
    """Return and step-progress evidence come from the same on-policy rollout."""
    env = make_environment(
        task="navigation",
        scene="S01",
        sensor="depth",
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
        "advantage_progress_correlation",
        "advantage_positive_fraction",
        "critic_return_rmse_preupdate",
        "depth_valid_fraction",
    ):
        assert name in metrics
        assert np.isfinite(np.asarray(metrics[name])).all(), name
