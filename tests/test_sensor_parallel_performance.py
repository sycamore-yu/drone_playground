"""Geometry and selective depth capture keep their original measurement contract."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

import drone_playground
from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.procedural_scene import ProceduralScene
from drone_playground.simulation.scene import _intersection


def test_bptt_update_with_one_terminal_world_uses_masked_depth_capture():
    """A single terminal world can be restarted inside a differentiable rollout."""
    from drone_playground.learning.trainer import Trainer
    from drone_playground.simulation.tasks import Event

    env = make_environment(
        task="navigation",
        scene="train/procedural",
        sensor="depth",
        sensor_config={"width": 16, "height": 12, "frequency_hz": 10},
        observation_config={"depth": {"pool": 1}},
        procedural={"max_obstacles": 8},
        num_envs=4,
        method_hz=10,
        device="cpu",
    )
    trainer = Trainer(env, kind="depth", algorithm="bptt", seed=4, config={"horizon": 2})
    state = trainer.initialize()
    task = state.env_state.task
    state = state.replace(
        env_state=state.env_state.replace(
            task=task.replace(event=task.event.at[0].set(Event.SUCCESS))
        )
    )
    updated, metrics = trainer.update(state)
    jax.block_until_ready((updated, metrics))
    assert bool(jnp.isfinite(metrics["loss"]))
    assert int(updated.env_state.scene_state.episode[0]) > 0
    env.sim.close()


def test_masked_reset_preserves_nonempty_nearby_depth_hits():
    """Sparse acquisition reproduces actual depth hits, not only empty frames."""
    env = make_environment(
        task="navigation",
        scene="train/procedural",
        sensor="depth",
        sensor_config={"width": 16, "height": 12, "frequency_hz": 30, "pitch_deg": 0},
        procedural={"max_obstacles": 8},
        num_envs=4,
        device="cpu",
    )
    initial = env.reset(jax.random.key(21))
    position = initial.physics.states.pos[:, 0]
    obstacle = position[:, None, :] + jnp.array([4.0, 0.0, 0.0])
    original = initial.scene_state
    centers = original.centers.at[:, 0].set(obstacle[:, 0])
    scene_state = original.replace(
        centers=centers,
        target=centers,
        sizes=original.sizes.at[:, 0].set(jnp.array([1.0, 1.0, 1.0])),
        active=jnp.zeros_like(original.active).at[:, 0].set(True),
        moving=jnp.zeros_like(original.moving),
        cylinder=jnp.zeros_like(original.cylinder),
    )
    selected = jnp.array([True, False, True, False])
    full = env.sensor.reset(initial.physics, scene_state=scene_state)
    partial = env.sensor.reset(initial.physics, scene_state=scene_state, mask=selected)
    assert np.asarray(full.measurement.mask[selected]).any()
    np.testing.assert_allclose(
        partial.measurement.values[selected], full.measurement.values[selected], atol=2e-5
    )
    np.testing.assert_array_equal(
        partial.measurement.mask[selected], full.measurement.mask[selected]
    )
    env.sim.close()


def test_masked_depth_reset_handles_partial_last_chunk():
    """Selection preserves all requested worlds when batch is not divisible by 32."""
    env = make_environment(
        task="navigation",
        scene="train/procedural",
        sensor="depth",
        sensor_config={"width": 8, "height": 4, "frequency_hz": 10},
        procedural={"max_obstacles": 8},
        num_envs=35,
        method_hz=10,
        device="cpu",
    )
    state = env.reset(jax.random.key(12))
    mask = jnp.arange(35) != 7
    expected = env.reset(jax.random.key(42), state, mask=jnp.ones(35, dtype=bool))
    actual = env.reset(jax.random.key(42), state, mask=mask)
    np.testing.assert_allclose(
        actual.observation.measurement.values[mask],
        expected.observation.measurement.values[mask],
        atol=2e-5,
    )
    np.testing.assert_array_equal(
        actual.observation.measurement.mask[mask], expected.observation.measurement.mask[mask]
    )
    np.testing.assert_allclose(
        actual.physics.states.pos[:, 0][mask], expected.physics.states.pos[:, 0][mask]
    )
    env.sim.close()


@pytest.mark.parametrize("case", ["oa_static_depth", "oa_dynamic_depth"])
def test_depth_10hz_experiment_changes_only_sampling_frequency(case):
    """The low-frequency preset changes only one sensor setting."""
    cfg_root = Path(drone_playground.__file__).parent / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(cfg_root)):
        reference = compose(config_name="config", overrides=[f"experiment={case}"])
        candidate = compose(config_name="config", overrides=[f"experiment={case}_10hz"])
    assert reference.sensor.config.frequency_hz == 30
    assert candidate.sensor.config.frequency_hz == 10
    candidate.sensor.config.frequency_hz = 30
    assert candidate == reference


@pytest.mark.parametrize("motion", ["static", "dynamic"])
def test_per_origin_raycast_with_multiple_observation_origins(motion):
    """Ray batches with several acquisition times preserve geometry semantics."""
    scene = ProceduralScene({"max_obstacles": 8, "obstacle_motion": motion})
    state = scene.sample(jax.random.key(4), 2)
    origins = jnp.array(
        [
            [[20.0, 0.0, 3.0], [30.0, -4.0, 3.0]],
            [[45.0, 3.0, 2.0], [65.0, 4.0, 3.0]],
        ]
    )
    directions = jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), (2, 2, 7, 3))
    times = jnp.array([[0.0, 2.0], [0.5, 1.5]])
    result = jax.jit(lambda: scene.raycast(origins, directions, times, 12.0, scene_state=state))()
    for world in range(2):
        selected = jax.tree.map(lambda x, i=world: x[i : i + 1], state)
        for origin in range(2):
            expected = scene.raycast(
                origins[world : world + 1, origin],
                directions[world : world + 1, origin],
                times[world : world + 1, origin],
                12.0,
                scene_state=selected,
            )
            np.testing.assert_allclose(result[world, origin], expected[0], atol=2e-5)


@pytest.mark.parametrize("range_limit", [10.0, 40.0])
@pytest.mark.parametrize("crowded", [False, True])
def test_nearby_obstacle_selection_matches_full_intersection(range_limit, crowded):
    """Conservative selection and overflow fallback match exhaustive ray tests."""
    scene = ProceduralScene({"max_obstacles": 208, "obstacle_motion": "dynamic"})
    state = scene.sample(jax.random.key(8), 3)
    position = jnp.array([[20.0, 0.0, 3.0], [50.0, 0.0, 3.0], [80.0, 0.0, 3.0]])
    if crowded:
        state = state.replace(
            centers=jnp.broadcast_to(
                position[:, None, :] + jnp.array([5.0, 0.0, 0.0]), state.centers.shape
            ),
            target=jnp.broadcast_to(
                position[:, None, :] + jnp.array([5.0, 0.0, 0.0]), state.centers.shape
            ),
            active=jnp.ones_like(state.active),
        )
    direction = jax.random.normal(jax.random.key(9), (3, 9, 3))
    direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
    times = jnp.array([0.0, 1.5, 2.5])

    def exhaustive():
        center = scene._centers_at(state, times)
        delta = position[:, None, None, :] - center[:, None, :, :]
        ray = direction[:, :, None, :]
        sizes = state.sizes[:, None, :, :]
        cylinders = _intersection("cylinder", delta, ray, sizes)
        boxes = _intersection("box", delta, ray, sizes)
        hit = jnp.where(state.cylinder[:, None, :], cylinders, boxes)
        return jnp.minimum(
            range_limit,
            jnp.min(jnp.where(state.active[:, None, :], hit, jnp.inf), axis=-1),
        )

    actual = jax.jit(
        lambda: scene.raycast(position, direction, times, range_limit, scene_state=state)
    )()
    np.testing.assert_allclose(actual, jax.jit(exhaustive)(), atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("motion", ["static", "dynamic"])
@pytest.mark.parametrize("selected", [(0,), (1, 3), (), (0, 1, 2, 3)])
def test_masked_depth_reset_matches_full_reset_and_preserves_live_worlds(motion, selected):
    """Only selected worlds get new depth data; all others keep their state."""
    env = make_environment(
        task="navigation",
        scene="train/procedural",
        sensor="depth",
        sensor_config={"width": 16, "height": 12, "frequency_hz": 30},
        procedural={"max_obstacles": 8, "obstacle_motion": motion},
        num_envs=4,
        method_hz=10,
        physics_hz=500,
        device="cpu",
    )
    initial = env.reset(jax.random.key(10))
    mask = jnp.isin(jnp.arange(4), jnp.array(selected, dtype=jnp.int32))
    full = env.reset(jax.random.key(20), initial, mask=jnp.ones(4, dtype=bool))
    partial = env.reset(jax.random.key(20), initial, mask=mask)
    for key in ("values", "mask", "points_body", "acquisition_time", "available_time"):
        actual = np.asarray(getattr(partial.observation.measurement, key))
        original = np.asarray(getattr(initial.observation.measurement, key))
        expected = np.asarray(getattr(full.observation.measurement, key))
        np.testing.assert_allclose(actual[np.asarray(mask)], expected[np.asarray(mask)], atol=3e-5)
        np.testing.assert_array_equal(actual[~np.asarray(mask)], original[~np.asarray(mask)])
    env.sim.close()
