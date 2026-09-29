"""Native planner control-task adapters retain their actual sensor and goal interface."""

import numpy as np
import pytest

from drone_playground.composition import compose_method, validate_config


@pytest.mark.parametrize("method", ["super", "ego_planner"])
@pytest.mark.parametrize("task", ["hovering", "tracking", "racing"])
def test_control_composition_is_explicit_and_sensor_is_preserved(method, task):
    cfg = compose_method("paper/" + method, task, ["observation@env.observation=state_reference"])
    validate_config(cfg)
    assert cfg["env"]["sensor"]["name"] != "none"
    assert cfg["env"]["execution"]["tracker"] is not None


def test_start_declares_interactive_goals_in_worker_protocol(tmp_path):
    from drone_playground.integrations.native_planner import NativePlanner

    worker = object.__new__(NativePlanner)
    worker.directory = tmp_path
    requests = []
    worker.request = lambda item, timeout: requests.append(item) or {"launch_xml": "<launch/>"}
    worker.start({}, np.ones(3), task_adapter={"interactive_goals": True})
    assert requests[0]["task_adapter"] == {"interactive_goals": True}


def test_fallback_only_episodes_cannot_pass_native_execution():
    from drone_playground.evaluation.native_control import native_execution_evidence

    good = dict(case=0, commands=100, trajectories=4, missing_command_steps=10)
    empty = dict(case=1, commands=0, trajectories=0, missing_command_steps=150)
    assert native_execution_evidence([good], 1)["passed"] is True
    result = native_execution_evidence([good, empty], 2)
    assert result["passed"] is False
    assert result["fallback_only_cases"] == [1]
    assert native_execution_evidence([], 1)["passed"] is False


def test_ego_patch_preserves_received_three_dimensional_goal(tmp_path):
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "src/planner/plan_manage/src/ego_replan_fsm.cpp"
    source.parent.mkdir(parents=True)
    original = """    bool success = false;
    end_pt_ << msg->poses[0].pose.position.x, msg->poses[0].pose.position.y, 1.0;
    success = planner_manager_->planGlobalTraj(odom_pos_, odom_vel_, Eigen::Vector3d::Zero(), end_pt_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
"""
    source.write_text(original)
    subprocess.run(
        ["git", "apply", str(root / "native_planners/patches/ego-3d-goals.patch")],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    assert source.read_text() == original.replace(
        "position.y, 1.0;", "position.y, msg->poses[0].pose.position.z;"
    )


def test_start_retains_compiled_native_identity(tmp_path):
    import json

    from drone_playground.integrations.native_planner import NativePlanner

    worker = object.__new__(NativePlanner)
    worker.directory = tmp_path
    identity = {"/native/binary": "a" * 64}
    worker.request = lambda item, timeout: {"launch_xml": "<launch/>", "runtime_sha256": identity}
    worker.start({}, np.ones(3))
    assert json.loads((tmp_path / "runtime-identity.json").read_text()) == identity


def test_ground_start_bootstrap_moves_only_vertical_target():
    from drone_playground.evaluation.native_control import bootstrap_position

    start = np.array([-1.5, 0.75, 0.01])
    np.testing.assert_array_equal(bootstrap_position("racing", start), [-1.5, 0.75, 0.8])
    np.testing.assert_array_equal(start, [-1.5, 0.75, 0.01])
    np.testing.assert_array_equal(bootstrap_position("hovering", start), start)
    high = np.array([1.0, 2.0, 1.5])
    np.testing.assert_array_equal(bootstrap_position("racing", high), high)
