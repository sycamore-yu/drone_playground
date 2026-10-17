"""Bounded training and playback contracts with real Orbax persistence."""

import json
import signal
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import pytest
from flax import struct
from hydra import compose, initialize_config_dir

from drone_playground import cli
from drone_playground.learning import runner
from drone_playground.learning.checkpoint import load_model_state
from drone_playground.simulation.policy import read_checkpoint


@pytest.fixture
def config(tmp_path):
    """Use the public tracking recipe and a short, exactly divisible budget."""
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(cli.__file__).parent / "configs")
    ):
        result = compose(config_name="config", overrides=["experiment=tracking", "mode=train"])
    result.output = str(tmp_path)
    result.simulation.device = "cpu"
    result.simulation.num_envs = 4
    result.learning.options.horizon = 2
    result.learning.max_iterations = 4
    result.learning.checkpoint_interval = 2
    result.learning.log_interval = 1
    return result


@struct.dataclass
class SmallState:
    """Keep update counters and policy parameters in an actual serializable state."""

    updates: int = 0
    interactions: int = 0
    params: dict = struct.field(default_factory=lambda: {"params": jnp.zeros(1)})


class Record:
    """Capture the lifecycle without GPU queries or TensorBoard setup."""

    def __init__(self, directory):
        """Capture calls in memory."""
        self.directory = directory
        self.identity = {"status": "running"}
        self.events = []

    def event(self, event, **fields):
        """Append one event."""
        self.events.append({"event": event, **fields})

    def finish(self, status, **fields):
        """Store the final state."""
        self.identity.update(status=status, **fields)


@pytest.fixture
def training(monkeypatch, tmp_path):
    """Replace only expensive dynamics; keep the actual loop and Checkpoint writer."""
    harness = SimpleNamespace(stop_at=None, nonfinite_at=None, handlers={})

    class Trainer:
        actor = SimpleNamespace(
            specification={"kind": "state", "action_size": 4, "hidden_size": 192}
        )
        kind = "state"
        distance_curriculum = False

        def __init__(self, *args, **kwargs):
            self.resolved_config = {}
            self.config = {"horizon": 2}
            self.algorithm = kwargs["algorithm"]
            self.env = SimpleNamespace(num_envs=4, task=SimpleNamespace(name="tracking"))

        def initialize(self):
            return SmallState()

        def update(self, state):
            update = state.updates + 1
            if update == harness.stop_at:
                harness.handlers[signal.SIGTERM](signal.SIGTERM, None)
            return state.replace(
                updates=update,
                interactions=state.interactions + 8,
                params={"params": state.params["params"] + 1},
            ), {
                "loss": float("nan") if update == harness.nonfinite_at else 1.0,
                "episodes_completed": 0,
                "episode_return_sum": 0.0,
                "episode_length_sum": 0,
                "episode_successes": 0,
            }

    def set_signal(sig, handler):
        old = harness.handlers.get(sig, signal.SIG_DFL)
        harness.handlers[sig] = handler
        return old

    monkeypatch.setattr(runner, "Trainer", Trainer)
    monkeypatch.setattr(runner, "create_environment", lambda config: object())
    monkeypatch.setattr(runner, "create_method", lambda config: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(runner, "gpu_usage", lambda: {})
    monkeypatch.setattr(runner.signal, "signal", set_signal)
    harness.record = Record(tmp_path)
    return harness


def test_bounded_training_exports_latest_actor_without_evaluation(config, training):
    """The finite loop saves each interval and finishes without a quality claim."""
    runner.train(config, training.record)
    result = training.record.identity
    assert result["status"] == "budget_exhausted" and result["interactions"] == 32
    assert result["latest_checkpoint"] == "checkpoints/model_000004"
    path = training.record.directory / result["latest_checkpoint"]
    state, metadata = load_model_state(path, SmallState())
    assert state.updates == 4 and state.params["params"].item() == 4
    assert metadata["provenance"]["interactions"] == 32
    read_checkpoint(path, "inference")
    assert {x["event"] for x in training.record.events} == {"update"}
    assert not (training.record.directory / "report.json").exists()
    assert all(training.handlers[sig] == signal.SIG_DFL for sig in (signal.SIGINT, signal.SIGTERM))


def test_interruption_resumes_from_the_exact_saved_update(config, training):
    """Resume preserves counters and policy state without replaying an update."""
    training.stop_at = 2
    runner.train(config, training.record)
    assert training.record.identity["status"] == "interrupted"
    config.resume = str(training.record.directory / "checkpoints/model_000002")
    training.stop_at = None
    runner.train(config, training.record)
    state, _ = load_model_state(
        training.record.directory / "checkpoints/model_000004", SmallState()
    )
    assert state.updates == 4 and state.params["params"].item() == 4
    assert [x["update"] for x in training.record.events if x["event"] == "update"] == [1, 2, 3, 4]


def test_nonfinite_update_preserves_last_good_state(config, training):
    """A failed numerical update cannot overwrite the previous finite parameters."""
    training.nonfinite_at = 2
    with pytest.raises(FloatingPointError, match="Nonfinite update"):
        runner.train(config, training.record)
    state, _ = load_model_state(
        training.record.directory / "checkpoints/model_000001", SmallState()
    )
    assert state.params["params"].item() == 1
    assert training.record.identity["status"] == "failed"


def test_target_stops_entire_loop_and_exports_that_update(config, training, monkeypatch):
    """A reached target cannot execute another training update."""
    config.learning.best.source = "training"
    config.learning.best.confirmation_episodes = 0
    config.learning.early_stopping.target = 2.0

    def measure(self, state, *, final=False):
        return {
            "source": "training",
            "mean_reward": float(state.updates),
            "episodes": 1,
            "episode_records": [],
            "improved": True,
        }

    monkeypatch.setattr(runner.RewardSelection, "measure", measure)
    runner.train(config, training.record)
    result = training.record.identity
    assert result["status"] == "early_stopped" and result["updates"] == 2
    assert result["interactions"] == 16 and result["requested_iterations"] == 4
    state, metadata = load_model_state(
        training.record.directory / result["latest_checkpoint"], SmallState()
    )
    assert state.updates == 2 and state.params["params"].item() == 2
    assert metadata["provenance"]["stop_reason"]["reason"] == "target_reached"
    config.resume = str(training.record.directory / result["latest_checkpoint"])
    runner.train(config, training.record)
    assert training.record.identity["updates"] == 2
    assert [event["update"] for event in training.record.events if event["event"] == "update"] == [
        1,
        2,
    ]


def test_runtime_stop_finishes_before_another_update(config, training, monkeypatch):
    """A control file written during an update stops at its following boundary."""
    original = runner.Trainer.update

    def update(self, state):
        result = original(self, state)
        (training.record.directory / "training-control.json").write_text(json.dumps({"stop": True}))
        return result

    monkeypatch.setattr(runner.Trainer, "update", update)
    runner.train(config, training.record)
    assert training.record.identity["updates"] == 1
    assert training.record.identity["status"] == "early_stopped"
    assert training.record.identity["stop_reason"]["reason"] == "requested_stop"


def test_live_target_rechecks_current_policy_instead_of_stale_score(config, training, monkeypatch):
    """Enabling a target re-evaluates a policy that changed since the last probe."""
    config.learning.max_iterations = 8
    config.learning.best.source = "training"
    config.learning.best.confirmation_episodes = 0
    original = runner.Trainer.update
    probes = []

    def update(self, state):
        result = original(self, state)
        if result[0].updates == 3:
            (training.record.directory / "training-control.json").write_text(
                json.dumps({"early_stopping": {"target": 5.0}})
            )
        return result

    def measure(self, state, *, final=False):
        if not final and state.updates % 2:
            return None
        probes.append(state.updates)
        return {
            "source": "training",
            "mean_reward": 10.0 if state.updates == 2 else 0.0,
            "episodes": 1,
            "episode_records": [],
            "improved": False,
        }

    monkeypatch.setattr(runner.Trainer, "update", update)
    monkeypatch.setattr(runner.RewardSelection, "measure", measure)
    runner.train(config, training.record)
    assert probes[:2] == [2, 3]
    assert training.record.identity["status"] == "budget_exhausted"
    assert training.record.identity["updates"] == 8


@pytest.mark.parametrize("control", [[], {"stop": "yes"}, {"unknown": True}])
def test_invalid_runtime_controls_are_rejected(tmp_path, control):
    """A typo cannot be interpreted as a stop request."""
    path = tmp_path / "training-control.json"
    path.write_text(json.dumps(control))
    with pytest.raises(ValueError, match="Training control"):
        runner.training_control(path)


@pytest.mark.parametrize("budget", [None, True, 0, -1, 1.5])
def test_budget_must_be_positive_integer_iterations(config, training, budget):
    """Reject invalid update budgets without checking batch divisibility."""
    config.learning.max_iterations = budget
    with pytest.raises(ValueError, match="positive integer"):
        runner.train(config, training.record)


def test_fixed_oa_map_is_rejected_for_training(config, training):
    """The fixed benchmark suite cannot accidentally become a training schedule."""
    config.task.name = "navigation"
    config.scene.name = "S01"
    with pytest.raises(ValueError, match="fixed maps belong to Benchmark"):
        runner.train(config, training.record)


def test_zip_checkpoint_is_rejected(tmp_path):
    """Historical archives require their original code, with no current ZIP loader."""
    path = tmp_path / "policy.zip"
    path.write_bytes(b"PK")
    with pytest.raises(ValueError, match="Orbax model directory"):
        read_checkpoint(path, "inference")


def test_replay_launches_selected_record_and_restores_paths(config, monkeypatch, tmp_path):
    """Verify replay launches selected record and restores paths."""
    path = tmp_path / "selected.mj_unroll"
    path.touch()
    (tmp_path / "another.mj_unroll").touch()
    metadata = tmp_path / "rscope_meta.pkl"
    metadata.touch()
    paths = SimpleNamespace(BASE_PATH=Path("old"), META_PATH=Path("meta"), TEMP_PATH=Path("cache"))
    visited = []

    def viewer(*, ssh_enabled):
        assert ssh_enabled is False
        records = list(paths.BASE_PATH.glob("*.mj_unroll"))
        assert len(records) == 1 and records[0].resolve() == path
        assert metadata == paths.META_PATH
        assert paths.TEMP_PATH.parent == paths.BASE_PATH.parent
        visited.append(paths.BASE_PATH)

    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setitem(sys.modules, "rscope", SimpleNamespace(config=paths))
    monkeypatch.setitem(sys.modules, "rscope.main", SimpleNamespace(main=viewer))
    config.mode = "replay"
    config.replay_path = str(path)
    cli.main(config)
    assert visited and not visited[0].exists()
    assert Path("old") == paths.BASE_PATH and Path("meta") == paths.META_PATH
    assert Path("cache") == paths.TEMP_PATH
    assert not (tmp_path / "run.json").exists()


def test_replay_reports_headless_and_missing_inputs(monkeypatch, tmp_path):
    """Verify replay reports headless and missing inputs."""
    path = tmp_path / "episode.mj_unroll"
    with pytest.raises(ValueError, match=r"existing .mj_unroll"):
        cli.replay(path)
    path.touch()
    with pytest.raises(ValueError, match=r"metadata is missing"):
        cli.replay(path)
    (tmp_path / "rscope_meta.pkl").touch()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    with pytest.raises(RuntimeError, match=r"headless"):
        cli.replay(path)


def test_replay_upstream_entry_decodes_real_record(monkeypatch, tmp_path):
    """Verify replay upstream entry decodes real record."""
    import mujoco.viewer
    from rscope import rollout

    from drone_playground.simulation.replay import export_replay

    source = tmp_path / "empty.xml"
    source.write_text("<mujoco><worldbody/></mujoco>")
    path = export_replay(
        tmp_path / "record",
        source,
        [0.0, 0.1],
        [[0, 0, 1], [0.1, 0, 1]],
        [[0, 0, 0, 1], [0, 0, 0, 1]],
    )
    (path.parent / "unselected.mj_unroll").write_bytes(b"must not be loaded")
    monkeypatch.setattr(rollout, "rollouts", [])
    for name, value in [
        ("num_evals", 0),
        ("num_envs", 0),
        ("env_ctrl_dt", 0),
        ("change_rollout", False),
    ]:
        monkeypatch.setattr(rollout, name, value)
    launched = []

    def launch(model, data, **kwargs):
        assert model.nq == 7
        assert rollout.num_evals == 1
        assert rollout.rollouts[0].qpos[:, 0, :3].tolist() == [[0, 0, 1], [0.1, 0, 1]]
        launched.append(True)
        return nullcontext(
            SimpleNamespace(is_running=lambda: False, lock=nullcontext, cam=SimpleNamespace())
        )

    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    cli.replay(path)
    assert launched == [True]


def test_replay_overlay_lock_and_viewer_thread_cleanup(monkeypatch, tmp_path):
    """Verify replay overlay lock and viewer thread cleanup."""
    import threading

    import mujoco.viewer

    path = tmp_path / "episode.mj_unroll"
    path.touch()
    (tmp_path / "rscope_meta.pkl").touch()
    paths = SimpleNamespace(BASE_PATH=Path("old"), META_PATH=Path("meta"), TEMP_PATH=Path("cache"))
    mutex = threading.Lock()
    events = []

    class Worker:
        def join(self):
            events.append("joined")

    worker = Worker()

    def set_texts():
        assert mutex.acquire(blocking=False), "MuJoCo's internal overlay lock would deadlock"
        mutex.release()
        events.append("overlay")

    @contextmanager
    def launch(*args, **kwargs):
        events.append("launched")
        yield SimpleNamespace(lock=lambda: mutex, set_texts=set_texts, cam=SimpleNamespace())
        events.append("closed")

    def main(**kwargs):
        model = SimpleNamespace(body=lambda name: SimpleNamespace(id=1))
        with mujoco.viewer.launch_passive(model) as viewer:
            assert viewer.cam.trackbodyid == 1 and viewer.cam.distance == 6.0
            with viewer.lock():
                viewer.set_texts()

    monkeypatch.setenv("DISPLAY", ":test")
    monkeypatch.setitem(sys.modules, "rscope", SimpleNamespace(config=paths))
    monkeypatch.setitem(sys.modules, "rscope.main", SimpleNamespace(main=main))
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(cli.threading, "enumerate", lambda: [worker] if events else [])
    cli.replay(path)
    assert events == ["launched", "overlay", "closed", "joined"]
    assert mujoco.viewer.launch_passive is launch


@pytest.mark.parametrize("rollout_length", [2, 7, 16])
def test_iteration_budget_does_not_change_when_rollout_length_changes(
    config, training, rollout_length
):
    """The budget is an iteration count, not a derived batch-transition total."""
    config.learning.max_iterations = 3
    config.learning.options.horizon = rollout_length
    runner.train(config, training.record)
    assert training.record.identity["updates"] == 3
    assert training.record.identity["requested_iterations"] == 3
