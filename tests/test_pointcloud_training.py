"""Composed paper training, state gradients and exact CPU continuation."""

import importlib
import importlib.util

import jax
import jax.numpy as jnp
import numpy as np

from drone_playground.composition import validate_config
from tests.reference_configs import compose_reference as compose_config


def module():
    name = "drone_playground.learning.algorithms.pointcloud_bptt"
    assert importlib.util.find_spec(name), "Paper BPTT trainer must be implemented"
    return importlib.import_module(name)


def small_config():
    return compose_config(
        "paper_pointcloud",
        [
            "training.num_envs=2",
            "training.policy_updates=2",
            "training.num_evals=2",
            "training.device=cpu",
            "algorithm.horizon_length=4",
            "scene.obstacles_per_kind=1",
            "observation.sensor.azimuth_count=12",
            "observation.sensor.elevation_count=3",
            "objective.velocity_window=2",
        ],
    )


def test_paper_composition_selects_real_module_slots():
    module()
    from drone_playground.composition import build_environment

    cfg = small_config()
    validate_config(cfg)
    task = build_environment(cfg, "cpu", "train", 2)
    assert task.model.forward == "point_mass_lag"
    assert task.model.backward == "exponential"
    assert task.sensor.points_per_frame == 36
    assert task.controller.input_kind == "world_acceleration"
    assert task.action_size == 3
    assert task.dt == 0.1
    assert cfg["algorithm"]["name"] == "pointcloud_bptt"


def test_paper_one_update_changes_encoder_and_has_finite_metrics():
    trainer = module()
    from drone_playground.composition import build_environment

    cfg = small_config()
    task = build_environment(cfg, "cpu", "train", 2)
    state, network, optimizer = trainer.initialize(task, cfg)
    update = trainer.make_update(task, network, optimizer, cfg)
    changed, metrics = update(state)
    assert int(changed.updates) == 1
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert float(metrics["gradient_norm"]) > 0
    old = state.params["params"]["point_0"]["kernel"]
    new = changed.params["params"]["point_0"]["kernel"]
    assert np.linalg.norm(np.asarray(new - old)) > 0


def test_paper_cpu_resume_preserves_optimizer_rng_and_update(tmp_path):
    trainer = module()
    from drone_playground.composition import build_environment
    from drone_playground.runs.pointcloud import load_training_state, save_training_state

    cfg = small_config()
    task = build_environment(cfg, "cpu", "train", 2)
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
    from drone_playground.composition import build_environment

    cfg = small_config()
    task = build_environment(cfg, "cpu", "train", 2)
    bank = task.scene.sample(jax.random.PRNGKey(4), 2)
    state = task.initial_state(bank)
    derivative = jax.jacrev(
        lambda v: task.observer.proprioception(
            state.replace(vel=v), bank.goal, jnp.ones(2) * 4.0, task.body_radius
        )[0]
    )(state.vel)
    assert np.isfinite(derivative).all() and np.linalg.norm(derivative) > 0
    cloud_derivative = jax.jacrev(
        lambda p: task.sensor.sample(bank, 0, p, jnp.eye(3), jnp.float32(0.0))[0].sum()
    )(state.pos[0])
    np.testing.assert_array_equal(cloud_derivative, jnp.zeros(3))
