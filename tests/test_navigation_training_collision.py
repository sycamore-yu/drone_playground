"""Continuous training collision loss never changes hard deployment event semantics."""

import jax
import jax.numpy as jnp
import numpy as np


def test_continuous_collision_training_retains_penetration_and_escape_gradient():
    from drone_playground.environments.scenes.navigation import KIND_BOX
    from drone_playground.learning.objectives.navigation import MotionNavigationObjective
    from tests.test_navigation import place, synthetic_env

    obstacle = dict(kind=KIND_BOX, size=(0.5, 0.5, 0.5), origin=(8.0, 0.0, 2.0))
    env = synthetic_env(
        [obstacle],
        training_collision_mode="continuous_loss",
        objective=MotionNavigationObjective(
            failure_penalty=0.0,
            progress_scale=0.05,
            velocity_scale=0.2,
            altitude_scale=0.2,
            clearance_scale=8.0,
        ),
    )
    try:
        state = env.reset(jax.random.PRNGKey(0), jnp.int32(0))
        state = place(state, env, (7.55, 0.1, 2.0))
        nxt = env.step(state, env.hover_action)
        assert float(nxt.metrics["collision"]) == 1.0
        assert float(nxt.metrics["clearance"]) < 0.0
        assert float(nxt.done) == 0.0
        assert int(nxt.info["outcome"]) == 2
        derivative = jax.grad(
            lambda x: env.step(place(state, env, (x, 0.1, 2.0)), env.hover_action).reward
        )(jnp.float32(7.55))
        assert np.isfinite(derivative) and derivative < 0
    finally:
        env.close()


def test_public_development_always_keeps_hard_collision_termination():
    from drone_playground.composition import build_environment, compose_method

    config = compose_method(
        "learning/navigation_bptt",
        "navigation/static_velocity",
        ["+training.navigation_collision_mode=continuous_loss"],
    )
    for split in ("train", "dev", "heldout"):
        env = build_environment(config, "cpu", split, 2)
        try:
            assert env.training_collision_mode == (
                "continuous_loss" if split == "train" else "terminate"
            )
        finally:
            env.close()
