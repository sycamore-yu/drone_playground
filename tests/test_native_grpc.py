"""Exercise the public Python client against an independently compiled C++ service."""

from pathlib import Path

import grpc
import numpy as np
import pytest

from drone_playground.native.client import NativeClient
from drone_playground.native.contracts import MotionCommand, Trajectory, Waypoint


@pytest.fixture
def server():
    binary = Path(__file__).parents[1] / "tmp/native-sdk/interop_server"
    if not binary.is_file():
        pytest.skip("Build the native C++ interoperability fixture; see native/README.md")
    return [str(binary), "{address}"]


def state(x=0):
    return dict(position=[x, 0, 1], velocity=[0, 0, 0], quaternion=[0, 0, 0, 1])


@pytest.mark.parametrize(
    "kind,output_type",
    [("trajectory", Trajectory), ("waypoint", Waypoint), ("motion", MotionCommand)],
)
def test_cpp_process_returns_typed_physical_output_and_resets_state(server, kind, output_type):
    with NativeClient(kind, command=server) as client:
        client.reset(goal=Waypoint([[5, 0, 1]], tolerance=0.5))
        result = client.step(time=1, state=state(2))
        assert result.status == "valid" and isinstance(result.output, output_type)
        assert result.plan_id == "1"
        if kind == "trajectory":
            np.testing.assert_allclose(result.output.sample(2)["position"], [4, 0, 1])
        elif kind == "waypoint":
            np.testing.assert_allclose(result.output.positions, [[5, 0, 1]])
        else:
            np.testing.assert_allclose(result.output.values, [1, 2, 3, 0])
        assert client.step(time=1.1, state=state()).plan_id == "2"
        client.reset()
        assert client.step(time=0, state=state()).plan_id == "1"
        process = client.process
    assert process.poll() is not None


def test_step_requires_reset_and_monotone_simulation_time(server):
    with NativeClient("trajectory", command=server) as client:
        with pytest.raises(RuntimeError, match="reset"):
            client.step(time=0, state=state())
        client.reset()
        client.step(time=1, state=state())
        with pytest.raises(ValueError, match="time"):
            client.step(time=0.9, state=state())


def test_rpc_deadline_prevents_using_an_uncertain_session(server):
    with NativeClient("trajectory", command=server, parameters={"delay_seconds": 0.2}) as client:
        client.reset()
        with pytest.raises(grpc.RpcError) as error:
            client.step(time=0, state=state(), timeout=0.01)
        assert error.value.code() == grpc.StatusCode.DEADLINE_EXCEEDED
        with pytest.raises(RuntimeError, match="fault"):
            client.step(time=0.1, state=state())


def test_no_plan_is_a_visible_outcome_not_a_synthetic_success(server):
    with NativeClient("no_plan", command=server) as client:
        client.reset()
        result = client.step(time=0, state=state())
        assert result.status == "no_plan" and result.output is None


def test_external_algorithm_uses_the_same_evaluation_adapter(server, tmp_path):
    from drone_playground.integrations.native_service import create_native_planner

    settings = dict(
        implementation="native_service", algorithm="trajectory", deployment=dict(command=server)
    )
    planner = create_native_planner(settings, tmp_path, 0, None)
    try:
        planner.start({}, [5, 0, 1])
        reply = planner.step(dict(time=0.0, **state()))
        assert reply["commands"] == 1 and reply["trajectories"] == 1
        np.testing.assert_allclose(reply["trajectory"].sample(0.5)["position"], [1, 0, 1])
        assert (tmp_path / "service-provenance.json").is_file()
    finally:
        planner.close()


def test_ros_bundle_remains_parseable_by_its_python38_runtime():
    import ast

    root = Path(__file__).parents[1]
    paths = [
        root / "native_planners/bridge/worker.py",
        root / "src/drone_playground/contracts.py",
        *(root / "src/drone_playground/native").rglob("*.py"),
    ]
    for path in paths:
        ast.parse(path.read_text(), filename=str(path), feature_version=(3, 8))


def test_ros_shutdown_reaps_a_child_in_a_separate_process_group(tmp_path):
    import importlib.util
    import subprocess
    import sys
    import time

    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location(
        "ros_worker_test", root / "native_planners/bridge/worker.py"
    )
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"], start_new_session=True); print(p.pid,flush=True); time.sleep(60)',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    child = int(parent.stdout.readline())
    worker.stop(parent)
    assert parent.poll() is not None
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        path = Path(f"/proc/{child}/stat")
        if not path.exists() or path.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("Shutdown orphaned the planner child")


def test_solver_budget_does_not_publish_a_late_action(server):
    with NativeClient("trajectory", command=server, parameters={"delay_seconds": 0.05}) as client:
        client.reset()
        decision = client.step(time=0, state=state(), timeout=2, solve_budget_seconds=0.01)
        assert decision.status == "budget_exhausted" and decision.output is None


def test_ego_fixed_map_covers_task_world_bounds(monkeypatch, tmp_path):
    import importlib.util
    import xml.etree.ElementTree as ET
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("ros_map_test", root / "native_planners/bridge/worker.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    fixture = ET.fromstring('''<launch><node pkg="ego_planner" type="ego_planner_node" name="ego">
        <param name="grid_map/map_size_x" value="$(arg map_size_x_)"/>
        <param name="grid_map/map_size_y" value="$(arg map_size_y_)"/>
        <param name="grid_map/virtual_ceil_height" value="4.9"/>
        </node></launch>''')
    monkeypatch.setattr(worker.ET, "parse", lambda _: ET.ElementTree(fixture))
    path = worker.launch_file("ego", {"intrinsics": dict(cx_px=60, cy_px=45, fx_px=65, fy_px=65)},
        [98, 0, 3], tmp_path, task_adapter=dict(world_low=[0, -20, 0.5], world_high=[100, 20, 6]))
    parsed = ET.fromstring(Path(path).read_text())
    values = {p.get("name"): p.get("value") for p in parsed.findall("node/param")}
    assert float(values["grid_map/map_size_x"]) / 2 > 100
    assert float(values["grid_map/map_size_y"]) / 2 > 20
    assert float(values["grid_map/virtual_ceil_height"]) == 5.9
