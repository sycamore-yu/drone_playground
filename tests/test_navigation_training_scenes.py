"""Training geometry must remain separate from the frozen evaluation catalog."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def configuration(method="learning/pointcloud_navigation"):
    from drone_playground.composition import compose_method

    return compose_method(method, overrides=["training=navigation_independent"])


@pytest.mark.parametrize("method", ["learning/pointcloud_navigation", "learning/depth_navigation"])
def test_training_samples_independent_geometry_while_development_keeps_navigation8(method):
    from drone_playground.environments.tasks.pointcloud_navigation import PointCloudNavigationTask
    from drone_playground.evaluation.pointcloud_navigation import PointCloudNavigationEvaluator

    task = PointCloudNavigationTask(configuration(method))
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
    evaluator = PointCloudNavigationEvaluator(task, None, repeats=8)
    assert evaluator.task.bank.num_instances == 64
    np.testing.assert_allclose(evaluator.task.bank.goal[:, 0], 98)
    assert task.bank.digest() == nominal_digest
    assert task.training_manifest["bank_digest"] == task.training_bank.digest()


def test_training_geometry_seed_changes_only_training_and_is_reproducible():
    from drone_playground.environments.tasks.pointcloud_navigation import PointCloudNavigationTask

    cfg = configuration()
    first = PointCloudNavigationTask(cfg)
    repeat = PointCloudNavigationTask(cfg)
    assert first.training_bank.digest() == repeat.training_bank.digest()
    cfg["training"]["scene"]["seed"] += 1
    other = PointCloudNavigationTask(cfg)
    assert first.training_bank.digest() != other.training_bank.digest()
    assert first.bank.digest() == other.bank.digest()


def test_default_and_legacy_checkpoints_keep_the_original_training_geometry():
    from drone_playground.composition import compose_method
    from drone_playground.environments.tasks.pointcloud_navigation import PointCloudNavigationTask
    from drone_playground.learning.algorithms.pointcloud_navigation_bptt import adaptation_contract

    cfg = compose_method("learning/pointcloud_navigation")
    task = PointCloudNavigationTask(cfg)
    assert task.training_bank is task.bank
    assert task.training_manifest == task.manifest
    current = adaptation_contract(cfg)
    cfg["training"].pop("scene")
    assert adaptation_contract(cfg) == current


def test_explicit_training_catalog_cannot_silently_reuse_benchmark_geometry():
    from drone_playground.composition import compose_method
    from drone_playground.environments.tasks.pointcloud_navigation import PointCloudNavigationTask

    cfg = compose_method("learning/pointcloud_navigation")
    cfg["training"]["scene"] = dict(cfg["env"]["scene"])
    with pytest.raises(ValueError, match="Independent training geometry"):
        PointCloudNavigationTask(cfg)


@pytest.mark.parametrize("method", ["learning/pointcloud_navigation", "learning/depth_navigation"])
def test_real_rollout_preserves_finite_nonzero_policy_gradients_on_training_bank(method):
    from drone_playground.environments.tasks.pointcloud_navigation import PointCloudNavigationTask
    from drone_playground.learning.algorithms.pointcloud_bptt import initialize
    from drone_playground.learning.algorithms.pointcloud_navigation_bptt import rollout_loss

    cfg = configuration(method)
    task = PointCloudNavigationTask(cfg)
    state, network, _ = initialize(task, cfg)
    (loss, _), gradient = jax.jit(jax.value_and_grad(
        lambda params: rollout_loss(task, network, params, jax.random.PRNGKey(9), 2, 2),
        has_aux=True,
    ))(state.params)
    assert np.isfinite(loss)
    leaves = jax.tree.leaves(gradient)
    assert all(np.isfinite(leaf).all() for leaf in leaves)
    assert sum(float(jnp.sum(jnp.abs(leaf))) for leaf in leaves) > 0
