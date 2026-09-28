"""Native planner composition, reference validation and physical tracking contracts."""

import numpy as np
import pytest

from drone_playground.composition import validate_config
from drone_playground.execution.controllers.trajectory import TrajectoryTracking
from drone_playground.integrations.native_planner import (
    DEFAULT_CONTAINER,
    NativePlanner,
    runtime_setup,
)
from tests.reference_configs import compose_reference as compose_config


def test_super_runtime_uses_its_own_ros_workspace():
    command = runtime_setup("super")
    assert "/planners/super/devel/setup.bash" in command
    assert "/planners/ego/devel/setup.bash" not in command


def test_cold_start_timeout_has_rpc_margin_and_is_recorded(tmp_path):
    planner = object.__new__(NativePlanner)
    planner.directory = tmp_path
    captured = {}

    def request(payload, timeout):
        captured.update(payload=payload, timeout=timeout)
        return {"ready": True, "launch_xml": "<launch/>"}

    planner.request = request
    planner.start({}, [1, 2, 3])
    assert captured["payload"]["startup_timeout_s"] == 90.0
    assert captured["timeout"] > captured["payload"]["startup_timeout_s"]


@pytest.mark.parametrize("task", ["static", "dynamic"])
@pytest.mark.parametrize("method", ["ego", "super"])
def test_native_recipe_composes_and_rejects_training(task, method):
    cfg = compose_config(f"p5_{task}_{method}")
    validate_config(cfg)
    assert cfg["method"]["container"] == DEFAULT_CONTAINER
    cfg["mode"] = "train"
    with pytest.raises(ValueError, match="eval/play"):
        validate_config(cfg)


def test_trajectory_tracking_hover_and_forward_direction():
    controller = TrajectoryTracking().bind([-1, -1, -3.2, 0], [1, 1, 3.2, 1])
    state = dict(pos=[0.0, 0.0, 2.0], vel=[0.0, 0.0, 0.0], quat=[0.0, 0.0, 0.0, 1.0])
    ref = dict(
        position=[0.0, 0.0, 2.0], velocity=[0.0, 0.0, 0.0], acceleration=[0.0, 0.0, 0.0], yaw=0.0
    )
    np.testing.assert_allclose(
        controller.command(state, ref, 0.027), [0.0, 0.0, 0.0, 0.027 * 9.81], atol=1e-7
    )
    ref["position"] = [1.0, 0.0, 2.0]
    assert controller.command(state, ref, 0.027)[1] > 0
    ref["acceleration"] = [np.nan, 0.0, 0.0]
    with pytest.raises(ValueError, match="Non-finite"):
        controller.command(state, ref, 0.027)


@pytest.mark.parametrize("stamp", [0.0, 2.0, np.nan])
def test_stale_future_and_nonfinite_native_reference_is_rejected(stamp):
    planner = object.__new__(NativePlanner)
    planner.request = lambda payload: {
        "reference": dict(
            position=[0.0, 0.0, 2.0],
            velocity=[0.0, 0.0, 0.0],
            acceleration=[0.0, 0.0, 0.0],
            yaw=0.0,
            time=stamp,
        )
    }
    assert planner.step({"time": 1.0})["reference"] is None


def test_selected_replay_retains_case_geometry_and_terminal_time(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from drone_playground.evaluation.navigation import export_navigation_replays
    from drone_playground.visualization import navigation_scene, rscope_io

    captured = []
    monkeypatch.setattr(navigation_scene, "active_indices", lambda bank, scenario: [])
    monkeypatch.setattr(
        navigation_scene,
        "obstacle_track",
        lambda bank, scenario, times: np.zeros((len(times), 0, 3)),
    )
    monkeypatch.setattr(navigation_scene, "create_replay_model", lambda env, scenario: scenario)

    def record(model, target, trace):
        captured.append(
            (model, trace["pos"][0, 0, 0], target.name, len(trace["time"]), trace["time"][-1, 0])
        )
        return target / "rollout.mj_unroll"

    monkeypatch.setattr(rscope_io, "export_rollout", record)
    env = SimpleNamespace(bank=SimpleNamespace(num_instances=18))
    active = np.ones((4, 6), bool)
    active[1:, 0] = False
    active[2:, 5] = False
    trace = {
        "pos": np.broadcast_to(np.arange(6)[None, :, None], (4, 6, 3)),
        "time": np.broadcast_to(np.array([0.02, 0.04, 0.06, 0.08])[:, None], (4, 6)),
        "active": active,
    }
    export_navigation_replays(env, {"medium": trace}, tmp_path, case_indices={"medium": [0, 5]})
    assert captured == [(6, 0, "case-000", 1, 0.02), (11, 5, "case-005", 2, 0.04)]
