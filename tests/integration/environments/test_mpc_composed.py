"""Real optimizer outputs advance the composed environment and reset solver memory."""

import jax
import numpy as np
import pytest

from drone_playground.composition import compose_experiment
from drone_playground.control.controllers.mpc.factory import build_controller
from drone_playground.environments.environment import build_environment


@pytest.mark.parametrize("kind", ["attitude_mpc", "sampling_mpc"])
def test_real_mpc_step_with_composed_dynamics_and_reset(kind, tmp_path):
    overrides = ["runtime.device=cpu", "runtime.action_delay_ms=null"]
    if kind == "sampling_mpc":
        overrides += ["method.decision.samples=16", "method.decision.prediction_device=cpu"]
    cfg = compose_experiment("control/" + kind, "hovering", overrides)
    env = build_environment(cfg, "cpu")
    ctrl = None
    try:
        state = env.reset(jax.random.PRNGKey(3))
        ctrl = build_controller(cfg, env, state, tmp_path)
        ctrl.reset(3)
        first = ctrl.step(env.controller_observation(state), 0)
        assert np.isfinite(first).all()
        end = jax.jit(env.step_physical)(state, first)
        assert np.isfinite(np.asarray(end.obs)).all()
        ctrl.after_step(first, env.controller_observation(end), float(end.reward), bool(end.done))
        ctrl.reset(3)
        repeated = ctrl.step(env.controller_observation(state), 0)
        np.testing.assert_allclose(first, repeated, rtol=1e-4, atol=1e-5)
    finally:
        if ctrl is not None:
            ctrl.close()
        env.close()


@pytest.mark.parametrize("kind", ["attitude_mpc", "sampling_mpc"])
def test_live_trajectory_reaches_mpc_and_physics_without_short_horizon_extrapolation(
    kind, tmp_path
):
    from drone_playground.control.external_tracking import ExternalTracking
    from drone_playground.references import Trajectory
    from drone_playground.runtime.decision import output_reply

    choices = [
        "runtime.device=cpu",
        "runtime.action_delay_ms=null",
        "controller@env.controller=" + kind,
    ]
    if kind == "sampling_mpc":
        choices += ["env.controller.samples=16"]
    config = compose_experiment("control/ppo", "hovering", choices)
    env = build_environment(config, "cpu")
    tracker = None
    try:
        state = env.reset(jax.random.key(1))
        tracker = ExternalTracking(env, state, tmp_path)
        coefficients = np.zeros((1, 4, 2))
        coefficients[0, :3, 0] = env.controller_observation(state)["pos"]
        curve = Trajectory(0.0, [3.0], coefficients)
        action = tracker.command(output_reply(curve, 0.0, "plan", 3.0), state, 0)
        assert tracker.consumed == 1
        result = jax.jit(env.step_physical)(state, action)
        assert np.isfinite(np.asarray(result.obs)).all()
        short = Trajectory(0.0, [0.1], coefficients)
        tracker.command(output_reply(short, 0.0, "short", 0.1), state, 0)
        assert tracker.short_horizon == 1 and tracker.consumed == 1
    finally:
        if tracker is not None:
            tracker.close()
        env.close()
