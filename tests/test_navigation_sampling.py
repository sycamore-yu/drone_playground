"""Training-only starts and deterministic scene blocks through real CPU updates."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from drone_playground import cli
from drone_playground.learning import checkpoint
from drone_playground.learning.ppo import collect_rollout
from drone_playground.learning.trainer import Trainer
from drone_playground.simulation.environment import make_environment
from drone_playground.simulation.tasks import Event

SCENES = ["S01", "S02", "S03", "D01", "D02", "D03", "S06", "D06"]


def assert_tree_equal(actual, expected):
    """Provide assert tree equal for the surrounding execution."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if isinstance(a, jax.Array) and jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)


def recipe(kind="depth", algorithm="shac"):
    """Select configuration fields that define the three-seed training recipe."""
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(cli.__file__).parent / "configs")
    ):
        return compose(
            config_name="config",
            overrides=[f"experiment=navigation_{kind}", f"learning={algorithm}"],
        )


@pytest.mark.parametrize("scene", SCENES)
def test_wide_starts_valid_reproducible_and_benchmark_unchanged(scene):
    """Verify wide starts valid reproducible and benchmark unchanged."""
    env = make_environment(
        action={"level": "acceleration", "heading": "target"},
        task="navigation",
        scene=scene,
        num_envs=64,
    )
    key = jax.random.PRNGKey(17)
    nominal = env.reset(key)
    assert_tree_equal(nominal, env.reset(key, randomize_position=False))
    wide = env.reset(key, randomize_position=True)
    assert_tree_equal(wide, env.reset(key, randomize_position=True))
    position = np.asarray(wide.physics.states.pos[:, 0])
    assert np.all(position >= [2, -18, 1])
    assert np.all(position <= [94, 18, 5.5])
    assert np.all(np.asarray(env.scene.clearance(position, 0.0)) - env.task.radius >= 0.15)
    # Coverage across the course, both sides and altitude, not only nominal jitter.
    assert np.all(np.histogram(position[:, 0], bins=[2, 25, 50, 75, 94])[0] > 0)
    assert position[:, 1].min() < -10 and position[:, 1].max() > 10
    assert position[:, 2].min() < 2 and position[:, 2].max() > 4.5
    offset = np.asarray(nominal.physics.states.pos[:, 0] - env.task.start)
    assert np.all(np.abs(offset) <= [0.25, 0.25, 0.100001])
    pos_key, vel_key, _ = jax.random.split(key, 3)
    original = env.task.start + jax.random.uniform(
        pos_key,
        (64, 3),
        minval=-jnp.array([0.25, 0.25, 0.1]),
        maxval=jnp.array([0.25, 0.25, 0.1]),
    )
    np.testing.assert_array_equal(nominal.physics.states.pos[:, 0], original)
    np.testing.assert_array_equal(
        wide.physics.states.vel[:, 0], jax.random.uniform(vel_key, (64, 3), minval=-0.1, maxval=0.1)
    )
    np.testing.assert_array_equal(wide.physics.states.quat, nominal.physics.states.quat)
    assert_tree_equal(nominal, env.reset(key))


@pytest.mark.parametrize("kind", ["depth", "lidar"])
def test_masked_random_reset_preserves_inactive_world_and_sensor(kind):
    """Verify masked random reset preserves inactive world and sensor."""
    env = make_environment(
        action={"level": "acceleration", "heading": "target"},
        task="navigation",
        scene="D01",
        num_envs=2,
        sensor=kind,
        sensor_config=(
            {"points_per_frame": 32}
            if kind == "lidar"
            else {"width": 64, "height": 48, "pitch_deg": 20, "max_range": 10}
        ),
        point_count=32,
    )
    state = env.step(env.reset(jax.random.PRNGKey(1)), jnp.zeros((2, 3)))
    reset = env.reset(
        jax.random.PRNGKey(2), state, jnp.array([True, False]), randomize_position=True
    )
    for a, b in zip(jax.tree.leaves(reset), jax.tree.leaves(state), strict=True):
        if isinstance(a, jax.Array) and jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        if np.ndim(a) and a.shape[0] == 2:
            np.testing.assert_array_equal(a[1], b[1])
    assert float(reset.time[0]) == 0
    assert float(reset.physics.states.pos[0, 0, 0]) > 2.25
    # Episode resets use the same sampler and clear only the finished histories.
    trainer = Trainer(
        env,
        loss=("zhang" if kind == "depth" else "liu"),
        kind=kind,
        config={"horizon": 1, "randomize_navigation_start": True},
    )
    initial = trainer.initialize()
    assert np.all(np.asarray(initial.env_state.physics.states.pos[:, 0, 0]) > 2.25)
    finished = state.replace(
        task=state.task.replace(event=jnp.array([Event.TIMEOUT, Event.RUNNING]))
    )
    history = jax.tree.map(jnp.ones_like, initial.objective_state)
    continued, _, _ = collect_rollout(
        trainer,
        initial.replace(
            env_state=finished, recurrent_memory=jnp.ones((2, 192)), objective_state=history
        ),
    )
    assert float(continued.env_state.time[0]) == 0
    assert float(continued.env_state.physics.states.pos[0, 0, 0]) > 2.25
    assert float(continued.env_state.time[1]) > float(state.time[1])
    for value in (continued.recurrent_memory, *jax.tree.leaves(continued.objective_state)):
        np.testing.assert_array_equal(value[0], 0)
    assert int(continued.objective_state.count[1]) == 2


@pytest.mark.parametrize("task,scene", [("tracking", "empty"), ("racing", "racing")])
def test_training_sampler_rejects_other_tasks(task, scene):
    """Verify training sampler rejects other tasks."""
    env = make_environment(task=task, scene=scene)
    with pytest.raises(ValueError, match=r"Navigation"):
        env.reset(jax.random.PRNGKey(0), randomize_position=True)
    with pytest.raises(ValueError, match=r"Navigation"):
        Trainer(env, config={"randomize_navigation_start": True})


@pytest.mark.parametrize("missing_option", ["randomize_navigation_start", "failure_cost"])
def test_missing_saved_training_option_is_rejected(tmp_path, missing_option):
    """A saved training contract must match exactly; missing options are not guessed."""
    trainer = Trainer(make_environment(), config={"horizon": 1})
    initial = trainer.initialize()
    old_config = dict(trainer.resolved_config)
    del old_config[missing_option]
    path = tmp_path / "old.zip"
    checkpoint.save_model(
        path,
        initial,
        config={"learning": old_config},
        provenance={},
        kind=trainer.kind,
        actor_spec=trainer.actor.specification,
    )
    with pytest.raises(ValueError, match="configuration"):
        trainer.load_state(path)
