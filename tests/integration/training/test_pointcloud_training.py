"""Composed paper training, state gradients and exact CPU continuation."""

import importlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.composition import compose_experiment, validate_config


def module():
    name = "drone_playground.learning.algorithms.recurrent_bptt"
    assert importlib.util.find_spec(name), "Paper BPTT trainer must be implemented"
    return importlib.import_module(name)


def small_config():
    return compose_experiment(
        "papers/differentiable_pointcloud",
        overrides=[
            "training.num_envs=2",
            "training.policy_updates=2",
            "training.num_evals=2",
            "runtime.device=cpu",
            "algorithm.horizon_length=4",
            "env.scene.obstacles_per_kind=1",
            "env.sensor.azimuth_count=12",
            "env.sensor.elevation_count=3",
            "env.task.loss.velocity_window=2",
        ],
    )


def test_reconstruction_composition_selects_real_module_slots():
    module()
    from drone_playground.environments.environment import build_environment

    cfg = small_config()
    validate_config(cfg)
    task = build_environment(cfg, "cpu", "train", 2).task
    assert task.dynamics.forward == "point_mass_lag"
    assert task.dynamics.backward == "exponential"
    assert task.sensor.points_per_frame == 36
    assert task.controller.input_kind == "state"
    assert task.action_size == 3
    assert task.dt == 0.1
    assert cfg["algorithm"]["name"] == "bptt"


def test_reconstruction_one_update_changes_encoder_and_has_finite_metrics():
    trainer = module()
    from drone_playground.environments.environment import build_environment

    cfg = small_config()
    task = build_environment(cfg, "cpu", "train", 2).task
    state, network, optimizer = trainer.initialize(task, cfg)
    update = trainer.make_update(task, network, optimizer, cfg)
    changed, metrics = update(state)
    assert int(changed.updates) == 1
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert float(metrics["gradient_norm"]) > 0
    old = state.params["params"]["point_0"]["kernel"]
    new = changed.params["params"]["point_0"]["kernel"]
    assert np.linalg.norm(np.asarray(new - old)) > 0


def test_reconstruction_cpu_resume_preserves_optimizer_rng_and_update(tmp_path):
    trainer = module()
    from drone_playground.artifacts.training_state import load_training_state, save_training_state
    from drone_playground.environments.environment import build_environment

    cfg = small_config()
    task = build_environment(cfg, "cpu", "train", 2).task
    state, network, optimizer = trainer.initialize(task, cfg)
    update = trainer.make_update(task, network, optimizer, cfg)
    first, _ = update(state)
    save_training_state(tmp_path / "state.pkl", first, cfg)
    restored, metadata = load_training_state(tmp_path / "state.pkl")
    assert metadata["config"] == cfg
    a, ma = update(first)
    b, mb = update(restored)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)
    for key in ma:
        np.testing.assert_array_equal(ma[key], mb[key])


def test_observation_detaches_sensor_but_keeps_velocity_gradient():
    trainer = module()
    del trainer
    from drone_playground.environments.environment import build_environment

    cfg = small_config()
    task = build_environment(cfg, "cpu", "train", 2).task
    bank = task.scene.sample(jax.random.PRNGKey(4), 2)
    state = task.initial_state(bank)
    derivative = jax.jacrev(
        lambda v: task.observation.proprioception(
            state.replace(vel=v), bank.goal, jnp.ones(2) * 4.0, task.body_radius
        )[0]
    )(state.vel)
    assert np.isfinite(derivative).all() and np.linalg.norm(derivative) > 0
    cloud_derivative = jax.jacrev(
        lambda p: task.sensor.sample(bank, 0, p, jnp.eye(3), jnp.float32(0.0))[0].sum()
    )(state.pos[0])
    np.testing.assert_array_equal(cloud_derivative, jnp.zeros(3))
