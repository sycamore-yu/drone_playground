"""Native acados Racing MPCC configuration, geometry and flight interface."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from drone_playground.simulation.factory import create_environment, create_method
from drone_playground.simulation.methods import PlannerController
from drone_playground.simulation.racing_mpcc import RacingMPCC, arc_length_course
from drone_playground.simulation.scene import Scene
from drone_playground.simulation.tasks import Task


def racing_config(*overrides):
    """Compose a real Racing controller configuration from Hydra."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[1] / "src/drone_playground/configs"),
    ):
        return compose(
            config_name="config",
            overrides=[
                "task=racing",
                "scene=racing",
                "sensor=state",
                "method=controller",
                "simulation.device=cpu",
                "simulation.num_envs=1",
                *overrides,
            ],
        )


def test_arc_length_preserves_gate_ordered_racing_course():
    """The controller uses the genuine Racing task spline."""
    task = Task("racing", Scene("racing"))
    arc, points = arc_length_course(task)
    assert arc[0] == 0 and arc[-1] > 4
    assert np.all(np.diff(arc) > 0)
    np.testing.assert_allclose(points[0], task.start, atol=1e-5)
    np.testing.assert_allclose(points[-1], task.goal, atol=1e-4)
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(points, axis=0), axis=1), np.diff(arc), rtol=0.08, atol=3e-3
    )


def test_progress_projection_cannot_jump_to_future_gate():
    """A revisited gate cannot skip the intervening ordered course."""
    task = Task("racing", Scene("racing"))
    controller = RacingMPCC.__new__(RacingMPCC)
    controller.arc, controller.course = arc_length_course(task)
    controller.progress = 0.0
    controller.frequency = 50.0
    controller.maximum_speed = 3.0
    assert 0 <= controller._project_progress(controller.course[-1]) < 0.2


def test_optional_course_binding_reuses_planner_controller_interface():
    """No separate Racing controller dispatch path is necessary."""
    events = []

    class CourseAwareController:
        execution = "host"
        output_level = "attitude_thrust"

        def set_course(self, task):
            events.append(("course", task.name))

        def initialize_memory(self, batch, seed):
            events.append(("memory", batch, seed))
            return ()

    method = PlannerController(CourseAwareController())
    memory = method.initialize(SimpleNamespace(task=SimpleNamespace(name="racing"), num_envs=1), 8)
    assert events == [("course", "racing"), ("memory", 1, 8)]
    assert memory == {"controller": (), "trajectory": None, "last_plan": -float("inf")}


def test_mpcc_hydra_group_matches_physical_controller_contract():
    """The native MPCC is discoverable without any legacy RPG adapter."""
    config = racing_config("controller=racing_mpcc")
    assert config.method.name == "controller"
    assert config.controller._target_.endswith("RacingMPCC")
    assert config.simulation.dynamics == "first_principles"


def test_real_acados_mpcc_short_flight_when_solver_installed(tmp_path):
    """Exercise the actual solver and Crazyflow for multiple decisions."""
    import os

    from drone_playground.simulation.runner import rollout_host

    source = os.getenv("ACADOS_SOURCE_DIR")
    if not source or not Path(source, "lib/libacados.so").is_file():
        pytest.skip("acados is needed for compiled MPC integration")
    config = racing_config(
        "controller=racing_mpcc",
        "controller.horizon=5",
        "controller.prediction_seconds=0.15",
        "task.duration=0.14",
        f"controller.workdir={tmp_path}/acados",
    )
    env = create_environment(config)
    method = create_method(config)
    try:
        episodes, trace, _, _ = rollout_host(env, method, seed=9)
        assert episodes[0]["event"] == "TIMEOUT"
        assert episodes[0]["method_failure"] == ""
        assert len(trace["time"]) > 3
        assert method.controller.last_status == 0
        assert np.isfinite(trace["position"]).all()
    finally:
        method.close()
