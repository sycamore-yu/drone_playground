"""Bounded CLI integration with real archives and cheap execution stand-ins."""

import json
import signal
import sys
import zipfile
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import jax.numpy as jnp
import pytest
from flax import struct
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from drone_playground import cli
from drone_playground.learning.checkpoint import (
    load_inference,
    load_state,
    save_inference,
    save_state,
)
from drone_playground.simulation import evaluation
from drone_playground.simulation.policy import load_policy


@pytest.fixture
def config(tmp_path):
    """Provide config for the surrounding execution."""
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(cli.__file__).parent / "configs")
    ):
        result = compose(config_name="config", overrides=["experiment=tracking", "mode=train"])
    result.output = str(tmp_path)
    result.simulation.device = "cpu"
    result.learning.evaluation_interval = 1
    result.learning.log_interval = 1
    return result


@struct.dataclass
class SmallState:
    """Represent SmallState state for the surrounding execution."""

    updates: int = 0
    interactions: int = 0
    params: dict = struct.field(default_factory=lambda: {"params": jnp.zeros(1)})


class Record:
    """Represent Record state for the surrounding execution."""

    def __init__(self, directory):
        """Initialize Record state for the declared contract."""
        self.directory = directory
        self.identity = {"status": "running"}
        self.events = []

    def event(self, event, **fields):
        """Provide event for the surrounding execution."""
        self.events.append({"event": event, **fields})

    def finish(self, status, **fields):
        """Provide finish for the surrounding execution."""
        self.identity.update(status=status, **fields)


@pytest.fixture
def training(monkeypatch, tmp_path):
    """Provide training for the surrounding execution."""
    import drone_playground.learning.trainer as learning

    harness = SimpleNamespace(
        now=0.0,
        stop_at=2,
        failed_evaluations=set(),
        fail_benchmark=False,
        evaluations=[],
        handlers={},
    )

    class Trainer:
        actor = object()
        resolved_config: ClassVar[dict] = {}

        def __init__(self, *args, **kwargs):
            pass

        def initialize(self):
            return SmallState()

        def update(self, state):
            harness.now += 2
            return state.replace(updates=state.updates + 1, interactions=state.interactions + 10), {
                "loss": 1.0
            }

    def evaluate(config, directory, *, evaluation_index=0, **kwargs):
        directory.mkdir(parents=True, exist_ok=True)
        harness.now += 3
        harness.evaluations.append((config.mode, evaluation_index))
        if config.mode == "benchmark" and harness.fail_benchmark:
            raise RuntimeError("benchmark interrupted")
        if config.mode == "checkpoint_eval" and evaluation_index == harness.stop_at:
            harness.handlers[signal.SIGTERM](signal.SIGTERM, None)
        return {"passed": evaluation_index not in harness.failed_evaluations}

    def set_signal(sig, handler):
        old = harness.handlers.get(sig, signal.SIG_DFL)
        harness.handlers[sig] = handler
        return old

    monkeypatch.setattr(learning, "Trainer", Trainer)
    monkeypatch.setattr(cli, "create_environment", lambda config: object())
    monkeypatch.setattr(cli, "evaluate", evaluate)
    monkeypatch.setattr(cli, "gpu_usage", lambda: {})
    monkeypatch.setattr(cli.time, "monotonic", lambda: harness.now)
    monkeypatch.setattr(cli.signal, "signal", set_signal)
    harness.record = Record(tmp_path)
    harness.path = tmp_path / "checkpoints/latest.training.zip"
    return harness


def test_resume_preserves_c5_selection_seeds_and_cost(config, training):
    """Verify resume preserves c5 selection seeds and cost."""
    cli.train(config, training.record)
    state, metadata = load_state(training.path, SmallState())
    assert state.updates == 2
    progress = metadata["provenance"]
    assert progress["consecutive_passes"] == progress["evaluation_index"] == 2
    assert progress["training_wall_seconds"] == 10
    config.resume = str(training.path)
    training.stop_at = None
    cli.train(config, training.record)
    assert training.evaluations == [
        ("checkpoint_eval", 1),
        ("checkpoint_eval", 2),
        ("checkpoint_eval", 3),
        ("benchmark", 0),
    ]
    selection = json.loads((training.record.directory / "selection.json").read_text())
    assert (
        selection["update"] == selection["consecutive_passes"] == selection["evaluation_index"] == 3
    )
    updates = [event for event in training.record.events if event["event"] == "update"]
    assert updates[-1]["interactions"] == 30
    assert updates[-1]["session_interactions"] == 10
    assert updates[-1]["interactions_per_second"] == 5
    assert training.record.identity["training_wall_seconds"] == 18
    _, saved = load_state(training.path, SmallState())
    assert saved["provenance"]["training_wall_seconds"] == 18
    assert training.record.identity["status"] == "accepted"
    _, _, frozen = load_policy(selection["checkpoint"])
    assert frozen["provenance"]["consecutive_passes"] == 3


def test_checkpoint_evaluations_do_not_repeat_policy_serialization(config, training, monkeypatch):
    """Keep crash recovery and C5 provenance with two state writes per evaluation."""
    from collections import Counter

    from drone_playground.learning import checkpoint

    writes = Counter()
    original_state, original_policy = checkpoint.save_state, checkpoint.save_inference

    def state_writer(path, state, **kwargs):
        writes[("state", state.updates)] += 1
        return original_state(path, state, **kwargs)

    def policy_writer(path, params, **kwargs):
        writes[("policy", kwargs["provenance"]["updates"])] += 1
        return original_policy(path, params, **kwargs)

    monkeypatch.setattr(checkpoint, "save_state", state_writer)
    monkeypatch.setattr(checkpoint, "save_inference", policy_writer)
    config.learning.checkpoint_interval = 1
    training.stop_at = None
    cli.train(config, training.record)
    assert writes[("state", 1)] <= 2
    assert writes[("policy", 1)] == 1
    assert sum(value for (purpose, _), value in writes.items() if purpose == "state") <= 7
    assert sum(value for (purpose, _), value in writes.items() if purpose == "policy") <= 4
    state, metadata = load_state(training.path, SmallState())
    assert state.updates == 3
    assert metadata["provenance"]["consecutive_passes"] == 3
    assert metadata["provenance"]["training_wall_seconds"] == 18


def test_resume_completes_pending_evaluation_before_training(config, training):
    """Retry a saved unevaluated update and retain the partial output."""
    save_state(
        training.path,
        SmallState(updates=2, interactions=20),
        config={"experiment": OmegaConf.to_container(config, resolve=True)},
        provenance={"evaluation_index": 1, "consecutive_passes": 1, "training_wall_seconds": 10},
    )
    partial = training.record.directory / "checkpoint_eval/update-00000002"
    partial.mkdir(parents=True)
    (partial / "episodes.csv").write_bytes(b"recorded before restart\n")
    config.resume = str(training.path)
    cli.train(config, training.record)
    state, metadata = load_state(training.path, SmallState())
    assert state.updates == 2 and state.interactions == 20
    assert training.evaluations == [("checkpoint_eval", 2)]
    assert metadata["provenance"]["consecutive_passes"] == 2
    retry = next(e for e in training.record.events if e["event"] == "checkpoint_evaluation_retry")
    assert (Path(retry["previous_output"]) / "episodes.csv").read_bytes() == (
        b"recorded before restart\n"
    )


def test_failed_evaluation_keeps_index_for_resume(config, training, monkeypatch):
    """A failed evaluation must be completed at the same saved update and seed index."""
    evaluate = cli.evaluate
    indices = []

    def interrupted(config, directory, *, evaluation_index=0, **kwargs):
        indices.append(evaluation_index)
        if len(indices) == 1:
            raise RuntimeError("evaluation interrupted")
        return evaluate(config, directory, evaluation_index=evaluation_index, **kwargs)

    monkeypatch.setattr(cli, "evaluate", interrupted)
    with pytest.raises(RuntimeError, match=r"evaluation interrupted"):
        cli.train(config, training.record)
    _, metadata = load_state(training.path, SmallState())
    assert metadata["provenance"]["evaluation_index"] == 0
    config.resume = str(training.path)
    training.stop_at = 1
    cli.train(config, training.record)
    state, _ = load_state(training.path, SmallState())
    assert indices == [1, 1]
    assert state.updates == 1 and state.interactions == 10


def test_failed_evaluation_resets_saved_consecutive_count(config, training):
    """Verify failed evaluation resets saved consecutive count."""
    training.failed_evaluations = {2}
    cli.train(config, training.record)
    _, metadata = load_state(training.path, SmallState())
    assert metadata["provenance"]["consecutive_passes"] == 0
    assert metadata["provenance"]["evaluation_index"] == 2


def test_initial_checkpoint_preserves_parameters_with_fresh_training(config, training, tmp_path):
    """Verify initial checkpoint preserves parameters with fresh training."""
    initial = tmp_path / "initial.policy.zip"
    parameters = {"params": jnp.ones(1)}
    save_inference(initial, parameters, kind="state", config={}, provenance={"updates": 999})
    config.initial_checkpoint = str(initial)
    cli.train(config, training.record)
    state, metadata = load_state(training.path, SmallState())
    assert jnp.array_equal(state.params["params"], parameters["params"])
    assert state.updates == 2
    assert metadata["config"]["initialization"]["provenance"]["updates"] == 999
    assert metadata["provenance"]["consecutive_passes"] == 2


def test_resume_selected_checkpoint_does_not_train_again(config, training):
    """Verify resume selected checkpoint does not train again."""
    training.stop_at = None
    training.fail_benchmark = True
    with pytest.raises(RuntimeError, match=r"benchmark interrupted"):
        cli.train(config, training.record)
    config.resume = str(training.path)
    training.fail_benchmark = False
    cli.train(config, training.record)
    assert training.evaluations[-2:] == [("benchmark", 0), ("benchmark", 0)]
    assert training.record.identity["updates"] == 3
    assert training.record.identity["session_interactions"] == 0
    assert Path(training.record.identity["benchmark_directory"]).name.startswith("benchmark-")
    assert (training.record.directory / "benchmark").is_dir()


def test_legacy_resume_does_not_invent_history(config, training):
    """Verify legacy resume does not invent history."""
    save_state(
        training.path,
        SmallState(updates=10, interactions=100),
        config={"experiment": OmegaConf.to_container(config, resolve=True)},
        provenance={},
    )
    config.resume = str(training.path)
    training.stop_at = 1
    cli.train(config, training.record)
    _, metadata = load_state(training.path, SmallState())
    assert metadata["provenance"]["evaluation_index"] == 1
    assert metadata["provenance"]["consecutive_passes"] == 1
    assert metadata["provenance"]["training_wall_seconds"] is None
    assert training.record.identity["interactions"] == 110
    assert training.record.identity["session_interactions"] == 10


def test_unknown_selection_is_rejected_before_training(config, monkeypatch):
    """Verify unknown selection is rejected before training."""
    config.learning.selection = "best_score"
    monkeypatch.setattr(cli, "create_environment", lambda config: pytest.fail("constructed env"))
    with pytest.raises(ValueError, match=r"selection"):
        cli.train(config, Record(Path(config.output)))


@pytest.mark.parametrize(
    "field",
    ["dynamics", "drone", "physics_hz", "method_hz", "action_delay", "navigation_goal_observation"],
)
def test_frozen_execution_rejects_changed_physical_contract(config, monkeypatch, field):
    """Verify frozen execution rejects changed physical contract."""
    metadata = {"config": {"experiment": OmegaConf.to_container(config, resolve=True)}}
    config.mode = "benchmark"
    config.checkpoint = "frozen.zip"
    if field == "action_delay":
        config.method.action_delay = 0.1
    elif field == "navigation_goal_observation":
        OmegaConf.update(config, "simulation.navigation_goal_observation", True, force_add=True)
    else:
        config.simulation[field] = "changed" if field in {"drone", "dynamics"} else 123
    record = Record(Path(config.output))
    monkeypatch.setattr(cli, "RunRecord", lambda *args, **kwargs: record)
    monkeypatch.setattr(cli, "load_policy", lambda path: (object(), {}, metadata))
    monkeypatch.setattr(cli, "evaluate", lambda *args, **kwargs: pytest.fail("evaluated mismatch"))
    with pytest.raises(ValueError, match=field):
        cli.main(config)
    assert record.identity["status"] == "failed"


@pytest.mark.parametrize("loader", ["training", "learning_inference", "simulation_inference"])
@pytest.mark.parametrize("damage", ["format_version", "purpose", "sha256"])
def test_shared_archive_validation(tmp_path, loader, damage):
    """Verify shared archive validation."""
    path = tmp_path / "checkpoint.zip"
    if loader == "training":
        save_state(path, SmallState(), config={}, provenance={})
    else:
        save_inference(path, SmallState().params, kind="state", config={}, provenance={})

    def load():
        if loader == "training":
            return load_state(path, SmallState())
        return load_inference(path) if loader == "learning_inference" else load_policy(path)

    load()
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read("metadata.json"))
        payload = archive.read("variables.msgpack")
    metadata[damage] = "invalid"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata))
        archive.writestr("variables.msgpack", payload)
    with pytest.raises(
        ValueError, match=r"checksum" if damage == "sha256" else "version or purpose"
    ):
        load()


@pytest.mark.parametrize(
    "count, missing, failed_static, expected",
    [
        (4, None, None, True),
        (3, None, None, False),
        (5, "D03", None, False),
        (5, "S03", None, False),
        (4, None, "S02", False),
    ],
)
@pytest.mark.parametrize("method", ["ego", "super"])
def test_native_s6_uses_main_scene_counts_and_static_successes(
    config, monkeypatch, tmp_path, count, missing, failed_static, expected, method
):
    """Verify native s6 uses main scene counts and static successes."""
    config.mode = "benchmark"
    config.task.name = "navigation"
    config.method = {"name": method, "address": "unused", "timeout": 1, "replan_hz": 5}
    scenes = ["S01", "S02", "S03", "D01", "D02", "D03", "S06", "D06"]
    config.benchmark.scenes = [scene for scene in scenes if scene != missing]
    config.benchmark.episodes = count
    monkeypatch.setattr(evaluation, "RosPlanner", lambda *args, **kwargs: nullcontext(object()))
    monkeypatch.setattr(evaluation, "create_environment", lambda config, **kwargs: kwargs["scene"])
    seen = set()

    def rollout(env, planner, **kwargs):
        success = env.startswith("S") and env != failed_static and env not in seen
        event = "SUCCESS" if success else "COLLISION"
        seen.add(env)
        return [{"event": event}], {}, [], [{"status": "OK", "solve_ms": 1}]

    def save(directory, env, episodes, traces, **kwargs):
        directory.mkdir(parents=True)
        report = dict(kwargs["report"])
        (directory / "report.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(evaluation, "rollout_ros", rollout)
    monkeypatch.setattr(evaluation, "save_evaluation", save)
    report = cli.evaluate(config, tmp_path)
    assert report["criterion"] == "S6"
    assert report["passed"] is expected
    assert report["static_episodes"] == count * (2 if missing == "S03" else 3)
    assert report["dynamic_episodes"] == count * (2 if missing == "D03" else 3)
    assert report["reports"]["D01"]["successes"] == 0
    assert report["reports"]["D01"]["outcomes"] == {"COLLISION": count}
    assert report["reports"]["D01"]["passed"] is None
    diagnostic = json.loads((tmp_path / "D01/episode-0000/report.json").read_text())
    assert diagnostic["criterion"] == "diagnostic" and diagnostic["passed"] is None
    assert (tmp_path / "D01/episode-0000/decisions.json").is_file()


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
