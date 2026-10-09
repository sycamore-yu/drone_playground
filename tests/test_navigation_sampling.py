"""Training-only starts and deterministic scene blocks through real CPU updates."""

import json
import signal
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
from drone_playground.simulation.environment import Environment
from drone_playground.simulation.policy import read_checkpoint
from drone_playground.simulation.records import RunRecord
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
            overrides=[f"experiment=navigation_mixed_{kind}", f"learning={algorithm}"],
        )


@pytest.mark.parametrize("scene", SCENES)
def test_wide_starts_valid_reproducible_and_benchmark_unchanged(scene):
    """Verify wide starts valid reproducible and benchmark unchanged."""
    env = Environment(task="navigation", scene=scene, num_envs=64)
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
    env = Environment(
        task="navigation",
        scene="D01",
        num_envs=2,
        sensor=kind,
        sensor_config={"points_per_frame": 32} if kind == "lidar" else {},
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
    trainer = Trainer(env, kind=kind, config={"horizon": 1, "randomize_navigation_start": True})
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
    env = Environment(task=task, scene=scene)
    with pytest.raises(ValueError, match=r"Navigation"):
        env.reset(jax.random.PRNGKey(0), randomize_position=True)
    with pytest.raises(ValueError, match=r"Navigation"):
        Trainer(env, config={"randomize_navigation_start": True})


@pytest.mark.parametrize("kind,batch", [("depth", 128), ("lidar", 32)])
@pytest.mark.parametrize("algorithm", ["ppo", "apg", "shac"])
def test_shared_mixed_recipes(kind, batch, algorithm):
    """Verify shared mixed recipes."""
    config = recipe(kind, algorithm)
    assert list(config.learning.scenes) == SCENES
    assert config.learning.scene_updates == 20
    assert config.learning.options.randomize_navigation_start is True
    assert config.learning.options.horizon == 32
    assert config.learning.options.lr == 0.0003
    assert config.learning.options.temporal_gradient_alpha == 0.916290731874155
    assert config.simulation.num_envs == batch
    assert config.initial_checkpoint is None
    assert config.benchmark.scenes is None
    assert config.learning.required_consecutive_passes == 3


@pytest.mark.parametrize("kind", ["depth", "lidar"])
def test_cli_real_scene_switch_and_exact_midblock_boundary_resume(kind, monkeypatch, tmp_path):
    """Verify cli real scene switch and exact midblock boundary resume."""
    config = recipe(kind)
    config.mode = "train"
    config.simulation.device = "cpu"
    config.simulation.num_envs = 2
    config.simulation.method_hz = 50
    config.learning.scenes = ["S01", "D01"]
    config.learning.scene_updates = 2
    config.learning.options.horizon = 2
    config.learning.options.minibatches = 1
    config.learning.options.critic_epochs = 1
    config.learning.options.ppo_epochs = 1
    config.learning.checkpoint_interval = 1
    config.learning.evaluation_interval = 1
    config.learning.log_interval = 1
    config.benchmark.scenes = ["S01"]  # Training evaluation must still request all eight.
    if kind == "lidar":
        config.sensor.config.points_per_frame = 32
        config.sensor.point_count = 32
    handlers, snapshots, inputs, outputs, constructed, initialized = {}, {}, {}, {}, [], []
    original_update, original_init = Trainer.update, Trainer.initialize
    original_save, original_environment = checkpoint.save_state, cli.create_environment

    def set_signal(sig, handler):
        previous = handlers.get(sig, signal.SIG_DFL)
        handlers[sig] = handler
        return previous

    def create_environment(config, **kwargs):
        constructed.append(kwargs["scene"])
        return original_environment(config, **kwargs)

    def initialize(self):
        initialized.append(self.env.scene.name)
        return original_init(self)

    def update(self, state):
        inputs[int(state.updates)] = (self.env.scene.name, state)
        result, metrics = original_update(self, state)
        outputs[int(result.updates)] = result
        if result.updates == 5:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return result, metrics

    def save(path, state, **kwargs):
        original_save(path, state, **kwargs)
        if state.updates in (3, 4) and state.updates not in snapshots:
            snapshot = tmp_path / f"resume-{state.updates}.zip"
            snapshot.write_bytes(Path(path).read_bytes())
            snapshots[state.updates] = snapshot

    def evaluate(config, *args, **kwargs):
        assert config.mode == "checkpoint_eval"
        assert config.benchmark.scenes is None
        assert config.benchmark.episodes is None
        return {"passed": False}

    monkeypatch.setattr(cli.signal, "signal", set_signal)
    monkeypatch.setattr(cli, "gpu_usage", lambda: {})
    monkeypatch.setattr(cli, "create_environment", create_environment)
    monkeypatch.setattr(cli, "evaluate", evaluate)
    monkeypatch.setattr(Trainer, "initialize", initialize)
    monkeypatch.setattr(Trainer, "update", update)
    monkeypatch.setattr(checkpoint, "save_state", save)
    config.output = str(tmp_path / "continuous")
    record = RunRecord(config.output, config)
    cli.train(config, record)
    expected = outputs[5]
    assert constructed == ["S01", "D01"]
    assert initialized == ["S01"]  # No reseeding/initialization when entering another scene.
    assert [scene for scene, _ in inputs.values()] == ["S01", "S01", "D01", "D01", "S01"]
    for boundary in (2, 4):
        before, after = outputs[boundary], inputs[boundary][1]
        for field in (
            "params",
            "critic_params",
            "target_critic_params",
            "optimizer_state",
            "log_std",
            "updates",
            "interactions",
        ):
            assert_tree_equal(getattr(after, field), getattr(before, field))
        np.testing.assert_array_equal(after.rng, jax.random.split(before.rng)[0])
        for value in (after.recurrent_memory, *jax.tree.leaves(after.objective_state)):
            np.testing.assert_array_equal(value, 0)
        np.testing.assert_array_equal(after.env_state.time, 0)
        np.testing.assert_array_equal(after.env_state.task.event, Event.RUNNING)
    header = json.loads((record.directory / "run.json").read_text())
    assert set(header["training_sampling"]["geometry_sha256"]) == {"S01", "D01"}
    events = [
        json.loads(line)
        for line in (record.directory / "events/metrics.jsonl").read_text().splitlines()
    ]
    switches = [event for event in events if event["event"] == "training_scene_switch"]
    assert [event["update"] for event in switches] == [2, 4]
    assert all(event["truncated_worlds"] == event["reset_worlds"] == 2 for event in switches)
    for update_count in (3, 4):
        _, metadata = read_checkpoint(snapshots[update_count], "training")
        assert metadata["provenance"]["training_sampling"] == {
            "scene_index": 1,
            "updates_in_scene": update_count - 2,
        }
        constructed.clear()
        initialized.clear()
        config.resume = str(snapshots[update_count])
        config.output = str(tmp_path / f"continued-{update_count}")
        cli.train(config, RunRecord(config.output, config))
        assert constructed == ["D01", "S01"]
        assert initialized == ["D01"]
        assert_tree_equal(outputs[5], expected)
        assert outputs[5].updates == 5 and outputs[5].interactions == 20


@pytest.mark.parametrize(
    "scenes,block", [([], 2), (["empty"], 2), (["S01", "S01"], 2), (["S01"], 0), (["S01"], True)]
)
def test_invalid_scene_schedule_fails_before_environment(scenes, block, monkeypatch):
    """Verify invalid scene schedule fails before environment."""
    config = recipe()
    config.learning.scenes, config.learning.scene_updates = scenes, block
    monkeypatch.setattr(cli, "create_environment", lambda *a, **kw: pytest.fail("constructed env"))
    with pytest.raises(ValueError, match=r"learning.scene"):
        cli.train(config, None)


@pytest.mark.parametrize("missing_option", ["randomize_navigation_start", "failure_cost"])
def test_old_trainer_checkpoint_without_optional_setting_restores(tmp_path, missing_option):
    """Restore a legacy checkpoint using the unchanged default for an omitted option."""
    trainer = Trainer(Environment(), config={"horizon": 1})
    initial = trainer.initialize()
    old_config = dict(trainer.resolved_config)
    del old_config[missing_option]
    path = tmp_path / "old.zip"
    checkpoint.save_state(path, initial, config={"learning": old_config}, provenance={})
    restored, _ = trainer.load_state(path)
    assert_tree_equal(restored, initial)
