"""Check the standalone policy-cloning diagnostic without importing its CLI."""

import json
import runpy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/ppo_rehearsal_diagnostic.py")
)
KL = MODULE["gaussian_kl"]
INSTALL_COMPLETE = MODULE["install_complete_rehearsal"]


def test_teacher_policy_has_zero_kl_and_zero_gradient():
    """The anchor is stationary at its source policy, including exploration scale."""
    mean, log_std = jnp.array([[0.2, -0.4, 0.8]]), jnp.array([-2.5, -1.0, 0.2])
    value, gradients = jax.value_and_grad(
        lambda current_mean, current_std: KL(mean, log_std, current_mean, current_std).mean(),
        argnums=(0, 1),
    )(mean, log_std)
    np.testing.assert_array_equal(value, 0.0)
    for gradient in gradients:
        np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))


def test_policy_shift_has_known_kl_and_a_restoring_mean_gradient():
    """Same unit covariance gives half squared mean distance, with correct KL direction."""
    teacher, current = jnp.zeros((1, 3)), jnp.array([[1.0, -2.0, 0.5]])
    value, gradient = jax.value_and_grad(
        lambda mean: KL(teacher, jnp.zeros(3), mean, jnp.zeros(3)).mean()
    )(current)
    np.testing.assert_allclose(value, 2.625)
    np.testing.assert_allclose(gradient, current)
    assert float(KL(teacher, jnp.zeros(3), teacher, jnp.full(3, jnp.log(2.0))).mean()) > 0


def test_complete_history_cloning_preserves_actual_ppo_samples_and_critic(tmp_path):
    """Exercise zero-weight equivalence, masking, restoring gradients and real Actor updates."""
    from drone_playground.learning import ppo
    from drone_playground.learning.trainer import Trainer
    from drone_playground.simulation.environment import make_environment

    env = make_environment(
        num_envs=8,
        task="navigation",
        scene=["S01", "D01"],
        sensor="depth",
        sensor_config={"width": 64, "height": 48},
        point_count=16,
    )
    try:
        options = {"horizon": 2, "ppo_epochs": 2, "minibatches": 2}
        control = Trainer(env, kind="depth", config=options)
        treatment = Trainer(env, kind="depth", config=options)
        initial = control.initialize()
        observed = env.observe(initial.env_state)
        observations = {
            key: jnp.repeat(observed[key][None], 4, axis=0)
            for key in ("state", "depth", "depth_mask")
        }
        lengths = np.array([2, 3, 4, 2, 3, 4, 2, 3])
        active = np.arange(4)[:, None] < lengths
        done = np.arange(4)[:, None] >= lengths - 1
        _, (means, _) = ppo.recurrent_policy(
            control,
            initial.params,
            observations,
            control.actor.initialize_memory(8),
            jnp.asarray(done),
        )
        # This synthetic optimizer fixture deliberately gives padding incorrect teacher targets.
        data = {f"obs__{key}": np.asarray(value) for key, value in observations.items()}
        data.update(
            teacher_mean=np.where(active[..., None], np.asarray(means), 1000),
            teacher_log_std=np.asarray(initial.log_std),
            active=active,
            done=done,
            source_world=np.arange(8),
            source_scene_index=np.arange(8) // 4,
            lengths=lengths,
        )
        np.savez(tmp_path / "corpus.npz", **data)
        source = tmp_path / "synthetic_source"
        metadata = {
            "collection_gate_passed": True,
            "source_policy_recompute_max_error": 0.0,
            "source_checkpoint": str(source),
            "source_provenance_reused": {"updates": 0, "interactions": 0},
            "actor": control.actor.specification,
            "scenes": ["S01", "D01"],
            "worlds_per_scene": 4,
            "episodes": [
                {
                    "world": i,
                    "steps": int(length),
                    "event": "SUCCESS",
                    "scene": "S01" if i < 4 else "D01",
                }
                for i, length in enumerate(lengths)
            ],
            "active_interactions": int(lengths.sum()),
            "total_wall_seconds": 0.0,
        }
        path = tmp_path / "corpus.json"
        path.write_text(json.dumps(metadata))
        with pytest.raises(ValueError, match="match the source"):
            INSTALL_COMPLETE(control, initial, tmp_path / "wrong_source", tmp_path, weight=0)
        data["active"][0, 0] = False
        np.savez(tmp_path / "corpus.npz", **data)
        with pytest.raises(ValueError, match="Invalid complete"):
            INSTALL_COMPLETE(control, initial, source, tmp_path, weight=0)
        data["active"][0, 0] = True
        np.savez(tmp_path / "corpus.npz", **data)

        expected, expected_metrics = control.update(initial)
        original_apply = control._apply_actor
        loss, record = INSTALL_COMPLETE(control, initial, source, tmp_path, weight=0)
        assert control._apply_actor == original_apply
        variables = {"params": initial.params, "log_std": initial.log_std}
        assert abs(float(loss(variables, initial.rng))) < 1e-6
        value, gradient = jax.value_and_grad(
            lambda log_std: loss({**variables, "log_std": log_std}, initial.rng)
        )(initial.log_std + 0.2)
        assert value > 0 and (gradient > 0).all()
        assert record["sequences_per_minibatch"] == 2
        actual, _ = control.update(initial)
        for first, second in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            if hasattr(first, "dtype") and jax.dtypes.issubdtype(first.dtype, jax.dtypes.prng_key):
                first, second = jax.random.key_data(first), jax.random.key_data(second)
            np.testing.assert_array_equal(first, second)

        INSTALL_COMPLETE(treatment, initial, source, tmp_path, weight=0.1)
        changed, changed_metrics = treatment.update(initial)
        assert changed.interactions == expected.interactions == 16
        assert changed.updates == expected.updates == 1
        np.testing.assert_array_equal(
            jax.random.key_data(changed.rng), jax.random.key_data(expected.rng)
        )
        for name in ("critic_params", "env_state", "recurrent_memory", "objective_state"):
            for first, second in zip(
                jax.tree.leaves(getattr(expected, name)),
                jax.tree.leaves(getattr(changed, name)),
                strict=True,
            ):
                if hasattr(first, "dtype") and jax.dtypes.issubdtype(
                    first.dtype, jax.dtypes.prng_key
                ):
                    first, second = jax.random.key_data(first), jax.random.key_data(second)
                np.testing.assert_array_equal(first, second)
        for name in ("reward", "value_target_mean", "value_target_std"):
            np.testing.assert_array_equal(expected_metrics[name], changed_metrics[name])
        for optimizer in ("actor", "critic"):
            counts = [
                int(leaf)
                for leaf in jax.tree.leaves(changed.optimizer_state[optimizer])
                if leaf.shape == () and np.issubdtype(leaf.dtype, np.integer)
            ]
            assert counts == [4]
        assert all(np.isfinite(value).all() for value in changed_metrics.values())
        assert (
            float(
                optax.tree.norm(jax.tree.map(lambda a, b: a - b, changed.params, expected.params))
            )
            > 1e-8
        )
    finally:
        env.sim.close()
