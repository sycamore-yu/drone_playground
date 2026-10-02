"""Training geometry must remain separate from the frozen evaluation catalog."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def configuration(method="differentiable_pointcloud"):
    from drone_playground.composition import compose_experiment

    if method == "differentiable_pointcloud":
        return compose_experiment("navigation/differentiable_pointcloud")
    return compose_experiment("papers/depth_diffphysics")


@pytest.mark.parametrize("method", ["differentiable_pointcloud", "depth_diffphysics"])
def test_training_samples_independent_geometry_while_checkpoint_eval_keeps_navigation_benchmark(method):
    from drone_playground.environments.tasks.navigation.recurrent import RecurrentNavigationEnv
    from drone_playground.evaluation.navigation.recurrent import RecurrentNavigationEvaluator

    task = RecurrentNavigationEnv(configuration(method))
    nominal_digest = task.bank.digest()
    assert task.training_bank.digest() != nominal_digest
    assert task.training_manifest["instance_count"] == 24
    assert task.training_manifest["role"] == "training"
    assert task.training_manifest["generator_seed"] == 81000
    assert task.training_bank.num_instances == 24
    bank, state, clocks, _, _ = jax.jit(lambda key: task.training_initial(key, 32))(
        jax.random.PRNGKey(17)
    )
    np.testing.assert_allclose(bank.goal[:, 0], 15.5)
    assert float(jnp.min(task.clearance(bank, state, clocks))) >= 0.15
    assert set(np.asarray(bank.motion).ravel()) >= {0, 1, 2}
    evaluator = RecurrentNavigationEvaluator(task, None, repeats=8)
    assert evaluator.task.bank.num_instances == 64
    np.testing.assert_allclose(evaluator.task.bank.goal[:, 0], 98)
    assert task.bank.digest() == nominal_digest
    assert task.training_manifest["bank_digest"] == task.training_bank.digest()


def test_training_geometry_seed_changes_only_training_and_is_reproducible():
    from drone_playground.environments.tasks.navigation.recurrent import RecurrentNavigationEnv

    cfg = configuration()
    first = RecurrentNavigationEnv(cfg)
    repeat = RecurrentNavigationEnv(cfg)
    assert first.training_bank.digest() == repeat.training_bank.digest()
    cfg["training"]["scene_distribution"]["scene"]["seed"] += 1
    other = RecurrentNavigationEnv(cfg)
    assert first.training_bank.digest() != other.training_bank.digest()
    assert first.bank.digest() == other.bank.digest()


def test_fixed_training_distribution_uses_the_configured_scene():
    from drone_playground.environments.tasks.navigation.recurrent import RecurrentNavigationEnv
    from drone_playground.learning.algorithms.recurrent_navigation_bptt import adaptation_contract

    cfg = configuration()
    cfg["training"]["scene_distribution"] = {"type": "fixed", "scene": None}
    task = RecurrentNavigationEnv(cfg)
    assert task.training_bank is task.bank
    assert task.training_manifest == task.manifest
    current = adaptation_contract(cfg)
    assert cfg["training"]["scene_distribution"]["type"] == "fixed"
    cfg["training"]["scene_distribution"]["scene"] = None
    assert adaptation_contract(cfg) == current


def test_explicit_training_catalog_cannot_silently_reuse_benchmark_geometry():
    from drone_playground.environments.tasks.navigation.recurrent import RecurrentNavigationEnv

    cfg = configuration()
    cfg["training"]["scene_distribution"]["scene"] = dict(cfg["env"]["scene"])
    with pytest.raises(ValueError, match=r"Independent training geometry"):
        RecurrentNavigationEnv(cfg)


@pytest.mark.parametrize("method", ["differentiable_pointcloud", "depth_diffphysics"])
def test_real_rollout_preserves_finite_nonzero_policy_gradients_on_training_bank(method):
    from drone_playground.environments.tasks.navigation.recurrent import RecurrentNavigationEnv
    from drone_playground.learning.algorithms.recurrent_bptt import initialize
    from drone_playground.learning.algorithms.recurrent_navigation_bptt import rollout_loss

    cfg = configuration(method)
    task = RecurrentNavigationEnv(cfg)
    state, network, _ = initialize(task, cfg)
    (loss, _), gradient = jax.jit(jax.value_and_grad(
        lambda params: rollout_loss(task, network, params, jax.random.PRNGKey(9), 2, 2),
        has_aux=True,
    ))(state.params)
    assert np.isfinite(loss)
    leaves = jax.tree.leaves(gradient)
    assert all(np.isfinite(leaf).all() for leaf in leaves)
    assert sum(float(jnp.sum(jnp.abs(leaf))) for leaf in leaves) > 0
