"""Training and evaluation construct distinct scene components from their own conditions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.composition import compose_experiment
from drone_playground.environments.environment import build_environment


def configuration(method="differentiable_pointcloud"):
    recipe = (
        "navigation/differentiable_pointcloud"
        if method == "differentiable_pointcloud"
        else "papers/depth_diffphysics"
    )
    return compose_experiment(recipe, overrides=["runtime.device=cpu"])


@pytest.mark.parametrize("method", ["differentiable_pointcloud", "depth_diffphysics"])
def test_training_geometry_is_independent_of_the_frozen_evaluation_catalog(method):
    config = configuration(method)
    training = build_environment(config, role="train")
    evaluation = build_environment(config, role="eval")
    try:
        assert training.bank.digest() != evaluation.bank.digest()
        assert training.task.manifest["instance_count"] == 24
        assert training.task.manifest["generator_seed"] == 81000
        bank, state, clocks, _, _ = jax.jit(lambda key: training.task.training_initial(key, 8))(
            jax.random.key(17)
        )
        np.testing.assert_allclose(bank.goal[:, 0], 15.5)
        assert float(jnp.min(training.task.clearance(bank, state, clocks))) >= 0.15
        np.testing.assert_allclose(evaluation.bank.goal[:, 0], 98)
    finally:
        training.close()
        evaluation.close()


def test_training_seed_changes_only_training_geometry():
    config = configuration()
    first = build_environment(config, role="train")
    evaluation = build_environment(config, role="eval")
    config["training"]["scene_distribution"]["scene"]["seed"] += 1
    other = build_environment(config, role="train")
    repeated_eval = build_environment(config, role="eval")
    try:
        assert first.bank.digest() != other.bank.digest()
        assert evaluation.bank.digest() == repeated_eval.bank.digest()
    finally:
        for env in (first, evaluation, other, repeated_eval):
            env.close()


def test_explicit_fixed_training_distribution_uses_the_requested_scene():
    config = configuration()
    config["training"]["scene_distribution"] = {"type": "fixed", "scene": None}
    training = build_environment(config, role="train")
    evaluation = build_environment(config, role="eval")
    try:
        assert training.bank.digest() == evaluation.bank.digest()
        assert training.task.training_bank is training.task.bank
    finally:
        training.close()
        evaluation.close()
