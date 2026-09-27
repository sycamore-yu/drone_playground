"""Native planner composition, reference validation and physical tracking contracts."""
import numpy as np
import pytest

from drone_playground.composition import compose_config, validate_config
from drone_playground.controllers.trajectory import TrajectoryTracking
from drone_playground.policies.native_planner import NativePlanner


@pytest.mark.parametrize("task", ["static", "dynamic"])
@pytest.mark.parametrize("method", ["ego", "super"])
def test_native_recipe_composes_and_rejects_training(task, method):
    cfg = compose_config(f"p5_{task}_{method}")
    validate_config(cfg)
    cfg["mode"] = "train"
    with pytest.raises(ValueError, match="evaluate/simulate"):
        validate_config(cfg)

def test_trajectory_tracking_hover_and_forward_direction():
    controller = TrajectoryTracking().bind([-1, -1, -3.2, 0], [1, 1, 3.2, 1])
    state = dict(pos=[0., 0., 2.], vel=[0., 0., 0.], quat=[0., 0., 0., 1.])
    ref = dict(position=[0., 0., 2.], velocity=[0., 0., 0.],
               acceleration=[0., 0., 0.], yaw=0.)
    np.testing.assert_allclose(controller.command(state, ref, 0.027),
                               [0., 0., 0., 0.027 * 9.81], atol=1e-7)
    ref["position"] = [1., 0., 2.]
    assert controller.command(state, ref, 0.027)[1] > 0
    ref["acceleration"] = [np.nan, 0., 0.]
    with pytest.raises(ValueError, match="Non-finite"):
        controller.command(state, ref, 0.027)

@pytest.mark.parametrize("stamp", [0.0, 2.0, np.nan])
def test_stale_future_and_nonfinite_native_reference_is_rejected(stamp):
    planner = object.__new__(NativePlanner)
    planner.request = lambda payload: {"reference": dict(
        position=[0., 0., 2.], velocity=[0., 0., 0.], acceleration=[0., 0., 0.],
        yaw=0., time=stamp)}
    assert planner.step({"time": 1.0})["reference"] is None


def test_selected_replay_retains_case_geometry_and_terminal_time(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from drone_playground.evaluation.navigation import export_navigation_replays
    from drone_playground.runs import navigation_scene, rscope_io

    captured = []
    monkeypatch.setattr(navigation_scene, "active_indices", lambda bank, scenario: [])
    monkeypatch.setattr(navigation_scene, "obstacle_track",
                        lambda bank, scenario, times: np.zeros((len(times), 0, 3)))
    monkeypatch.setattr(navigation_scene, "create_replay_model", lambda env, scenario: scenario)

    def record(model, target, trace):
        captured.append((model, trace["pos"][0, 0, 0], target.name,
                         len(trace["time"]), trace["time"][-1, 0]))
        return target / "rollout.mj_unroll"

    monkeypatch.setattr(rscope_io, "export_rollout", record)
    env = SimpleNamespace(bank=SimpleNamespace(num_instances=18))
    active = np.ones((4, 6), bool)
    active[1:, 0] = False
    active[2:, 5] = False
    trace = {"pos": np.broadcast_to(np.arange(6)[None, :, None], (4, 6, 3)),
             "time": np.broadcast_to(np.array([.02, .04, .06, .08])[:, None], (4, 6)),
             "active": active}
    export_navigation_replays(env, {"medium": trace}, tmp_path,
                              case_indices={"medium": [0, 5]})
    assert captured == [(6, 0, "case-000", 1, .02), (11, 5, "case-005", 2, .04)]
