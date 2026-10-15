"""Parallel fixed scenes preserve geometry, episode clocks and recovery state."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.learning.trainer import Trainer
from drone_playground.simulation import runner
from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.scene import SceneBatch
from drone_playground.simulation.sensors import SensorConfig, measure
from drone_playground.simulation.tasks import Event


def test_parallel_queries_match_individual_scenes_at_each_world_time():
    """Dispatch must preserve dynamic geometry and gradients at distinct times."""
    batch = SceneBatch(["S01", "D03"], 4)
    position = jnp.array([[25.0, 1, 3], [50.0, -2, 2], [30.0, 3, 4], [60.0, -4, 2]])
    times = jnp.array([0.0, 1.2, 4.6, 9.8])
    directions = jnp.array([[1.0, 0, 0], [0.0, 1, 0], [0.0, 0, -1]])
    expected = jnp.stack(
        [batch.scene_for_world(i).clearance(position[i], times[i]) for i in range(4)]
    )
    np.testing.assert_allclose(jax.jit(batch.clearance)(position, times), expected)
    gradient = jax.jit(jax.grad(lambda p: batch.clearance(p, times).sum()))(position)
    expected_gradient = jnp.stack(
        [jax.grad(batch.scene_for_world(i).clearance)(position[i], times[i]) for i in range(4)]
    )
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-6)
    for rays in (directions, jnp.broadcast_to(directions, (4, 3, 3)), directions[0]):
        expected = jnp.stack(
            [
                batch.scene_for_world(i).raycast(
                    position[i], rays[i] if rays.ndim == 3 else rays, times[i], 40.0
                )
                for i in range(4)
            ]
        )
        np.testing.assert_allclose(
            jax.jit(batch.raycast)(position, rays, times), expected, rtol=1e-6, atol=1e-5
        )
    point = jnp.array([2.0, 0.0, 3.0])
    assert batch.clearance(point) == min(scene.clearance(point) for scene in batch.scenes)
    with pytest.raises(IndexError):
        batch.scene_for_world(4)
    with pytest.raises(ValueError, match="positions"):
        batch.clearance(position[:2])


@pytest.mark.parametrize("names,count", [(["S01", "S01"], 2), (["empty"], 1), (["S01", "D03"], 3)])
def test_parallel_scene_assignment_rejects_invalid_groups(names, count):
    """Every named scene needs an equal nonempty group of physical worlds."""
    with pytest.raises(ValueError, match="unique Navigation"):
        SceneBatch(names, count)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"task": "tracking"}, "fixed Navigation"),
        ({"task": "navigation", "procedural": {}}, "fixed Navigation"),
        (
            {"task": "navigation", "task_config": {"reward_profile": "fan2025"}},
            "per-object reward",
        ),
    ],
)
def test_parallel_environment_rejects_unsupported_geometry(options, match):
    """Fail at construction before an unsupported reward or scene reaches JIT."""
    with pytest.raises(ValueError, match=match):
        make_environment(scene=["S01", "D01"], num_envs=2, **options)


@pytest.mark.parametrize("profile", ["d435i", "mid360"])
def test_parallel_sensor_measurements_use_actual_world_geometry(profile):
    """Both sensor modalities retain calibrated measurements under dispatch."""
    batch = SceneBatch(["S02", "D02"], 2)
    config = SensorConfig(profile=profile, width=8, height=6)
    positions = jnp.array([[50.0, 2, 3], [60.0, -1, 3]])
    quaternions = jnp.array([[0.0, 0, 0, 1], [0.0, 0, 0, 1]])
    times = jnp.array([0.3, 5.2])
    count = 16 if profile == "mid360" else None
    actual = jax.jit(lambda p, q, t: measure(batch, config, p, q, t, selected_count=count))(
        positions, quaternions, times
    )
    for i in range(2):
        expected = measure(
            batch.scene_for_world(i),
            config,
            positions[i],
            quaternions[i],
            times[i],
            selected_count=count,
        )
        for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(left[i], right, rtol=1e-5, atol=1e-5)


def test_parallel_ppo_preserves_live_worlds_and_restores_complete_state(tmp_path):
    """Three updates cross the old two-update boundary without an artificial reset."""
    env = make_environment(
        task="navigation",
        scene=["S01", "D01"],
        num_envs=2,
        action={"level": "acceleration", "heading": "target"},
        duration=0.12,
        task_config={"reward_profile": "goal_reaching", "finite_horizon": True},
    )
    trainer = Trainer(
        env,
        config={"horizon": 1, "ppo_epochs": 1, "critic_epochs": 1, "minibatches": 1, "gamma": 1.0},
    )
    initial = trainer.initialize()
    state = initial
    try:
        for _ in range(3):
            state, metrics = trainer.update(state)
            assert all(np.isfinite(value).all() for value in metrics.values())
        np.testing.assert_allclose(state.env_state.time, 0.06)
        assert int(state.interactions) == 6
        assert any(
            not np.array_equal(a, b)
            for a, b in zip(
                jax.tree.leaves(initial.params), jax.tree.leaves(state.params), strict=True
            )
        )
        path = tmp_path / "parallel.checkpoint"
        trainer.save_state(path, state, provenance={"test": "parallel fixed scenes"})
        restored, _ = trainer.load_state(path)
        expected = trainer.update(state)
        actual = trainer.update(restored)
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
            if isinstance(a, jax.Array) and jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
                a, b = jax.random.key_data(a), jax.random.key_data(b)
            np.testing.assert_array_equal(a, b)
        terminal = state.env_state.replace(
            task=state.env_state.task.replace(event=jnp.array([Event.TIMEOUT, Event.RUNNING]))
        )
        np.testing.assert_array_equal(terminal.terminated, [True, False])
        np.testing.assert_array_equal(terminal.truncated, [False, False])
        reset = env.reset(jax.random.key(31), terminal, terminal.done)
        assert reset.time[0] == 0 and reset.time[1] == state.env_state.time[1]
        np.testing.assert_array_equal(reset.physics.states.pos[1], terminal.physics.states.pos[1])
        after = state.env_state
        for _ in range(4):
            after = env.step(after, jnp.zeros((2, env.action_size)))
        np.testing.assert_array_equal(after.task.event, [Event.TIMEOUT, Event.TIMEOUT])
        np.testing.assert_array_equal(after.terminated, [True, True])
    finally:
        env.sim.close()


def test_parallel_episode_records_and_replay_identify_component_scene(tmp_path, monkeypatch):
    """A batch label must not replace actual scene identities in exported episodes."""
    env = make_environment(task="navigation", scene=["S01", "D03"], num_envs=2)
    try:
        state = env.reset(jax.random.key(7))
        state = state.replace(
            task=state.task.replace(event=jnp.array([Event.SUCCESS, Event.TIMEOUT]))
        )
        episodes = runner.episode_records(env, state, 7, "policy", 0.1)
        assert [row["scene"] for row in episodes] == ["S01", "D03"]
        assert [row["geometry_sha256"] for row in episodes] == [
            scene.geometry_hash for scene in env.scene.scenes
        ]
        seen = []

        def export(path, scene, *args, **kwargs):
            seen.append(scene.name)
            return path

        monkeypatch.setattr(runner, "export_replay", export)
        traces = {
            "time": np.array([[0.0, 0], [0.02, 0.02]]),
            "position": np.tile(np.asarray(state.physics.states.pos[:, 0])[None], (2, 1, 1)),
            "quaternion": np.tile(np.asarray(state.physics.states.quat[:, 0])[None], (2, 1, 1)),
            "sensor_points": np.zeros((2, 2, 0, 3)),
            "sensor_mask": np.zeros((2, 2, 0), dtype=bool),
        }
        runner.save_evaluation(
            tmp_path, env, episodes, traces, report={}, record_trajectories=False
        )
        assert seen == ["S01", "D03"]
    finally:
        env.sim.close()
