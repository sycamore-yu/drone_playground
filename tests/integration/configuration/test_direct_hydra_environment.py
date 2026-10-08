"""Hydra is the only component constructor and preset source."""

import importlib

import jax
import jax.numpy as jnp
from brax.envs.base import State


def test_environment_only_configuration_does_not_load_a_training_recipe():
    from drone_playground.configuration import load_config

    config = load_config("environment", ["env=hovering", "runtime.device=cpu"])
    assert not {"algorithm", "method", "training", "network", "objective"} & config.keys()
    assert set(config["env"]) == {
        "_target_",
        "name",
        "dynamics",
        "controller",
        "reference",
        "scene",
        "sensor",
        "task",
        "freq",
    }
    assert "reward" in config["env"]["task"]
    assert "observation" in config["env"]["task"]
    assert "freq" not in config["env"]["task"]
    assert "physics_freq" not in config["env"]["task"]
    assert (
        not {"adapter", "trainer", "evaluation_entrypoint", "policy_evaluator"}
        & config["env"]["task"].keys()
    )


def test_public_load_executes_six_components_and_returns_brax_state():
    import drone_playground as dp
    from drone_playground.environments.base import DroneEnvironment

    env = dp.load("hovering", overrides=["runtime.device=cpu"])
    try:
        assert isinstance(env.unwrapped, DroneEnvironment)
        assert env.task.name == "hovering"
        assert env.reference.name == "hover"
        state = jax.jit(env.reset)(jax.random.PRNGKey(4))
        state = jax.jit(env.step)(state, env.hover_action)
        assert isinstance(state, State)
        assert bool(jnp.isfinite(state.reward))
        assert env.dynamics.forward == "so_rpy"
        assert importlib.util.find_spec("drone_playground.registry") is None
    finally:
        env.close()
