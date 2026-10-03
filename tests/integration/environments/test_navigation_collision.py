"""Navigation collision semantics after complete environment construction."""

import jax
import jax.numpy as jnp
import numpy as np


def test_public_eval_always_keeps_hard_collision_termination():
    from drone_playground.composition import compose_experiment
    from drone_playground.environments.environment import build_environment

    config = compose_experiment(
        "navigation/bptt",
        "navigation/static",
        ["+training.navigation_collision_mode=continuous_loss"],
    )
    for role in ("train", "eval"):
        env = build_environment(config, "cpu", role, 2)
        try:
            expected = "continuous_loss" if role == "train" else "terminate"
            assert env.training_collision_mode == expected
        finally:
            env.close()


def test_continuous_collision_training_retains_penetration_and_escape_gradient():
    from drone_playground.environments.scenes.geometry import KIND_BOX
    from drone_playground.learning.objectives.navigation import MotionNavigationObjective
    from tests.helpers.scenes import place
    from tests.helpers.scenes import synthetic_navigation_env as synthetic_env

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
