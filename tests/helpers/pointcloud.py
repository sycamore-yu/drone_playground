"""Point-cloud fixtures that require a fully constructed environment."""

import jax
import jax.numpy as jnp

from drone_playground.configuration import compose_experiment
from drone_playground.environments.factory import build_environment


def task_and_bank():
    config = compose_experiment(
        "papers/differentiable_pointcloud",
        overrides=[
            "runtime.device=cpu",
            "training.num_envs=1",
            "env.scene.obstacles_per_kind=1",
            "env.sensor.azimuth_count=12",
            "env.sensor.elevation_count=3",
            "env.task.duration=0.3",
        ],
    )
    task = build_environment(config, "cpu").task
    bank = task.scene.sample(jax.random.PRNGKey(1), 1)
    bank = bank.replace(
        active=jnp.array([[True, False, False]]),
        kind=jnp.array([[2, 2, 2]]),
        size=jnp.array([[[0.02, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]),
        origin=jnp.array([[[2.0, 0.0, 3.0], [30.0, 0.0, 3.0], [32.0, 0.0, 3.0]]]),
        start=jnp.array([[0.5, 0.0, 3.0]]),
        goal=jnp.array([[40.0, 0.0, 3.0]]),
        subtype_names=("fixture",),
    )
    return config, task, bank
