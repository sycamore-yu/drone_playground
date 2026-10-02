"""Sampling MPC against a real tracking environment."""

import jax
import numpy as np

from drone_playground.actions.commands import Trajectory
from drone_playground.actions.controllers.mpc.sampling import SamplingMPC
from drone_playground.environments.tasks.references import race_reference
from drone_playground.environments.tasks.tracking.rigid_body import TrackingEnv


def test_actual_sampling_decision_and_warm_start():
    env = TrackingEnv(
        task="tracking",
        reference="random",
        dynamics="first_principles",
        device="cpu",
        reference_count=2,
    )
    ref, _ = race_reference(np.array([-1.5, 1, 0.07]))
    controller = SamplingMPC(
        drone="cf21B_500",
        reference=ref,
        frequency=50,
        samples=32,
        horizon=8,
        device="cpu",
    )
    try:
        state = env.reset(jax.random.PRNGKey(3))
        action = controller.compute_from_data(state.pipeline_state.sim_data, 0)
        assert action.shape == (4,)
        assert np.isfinite(action).all()
        assert controller.last_diagnostics["samples"] > 0
        second = controller.compute_from_data(state.pipeline_state.sim_data, 1)
        assert np.isfinite(second).all()

        coefficients = np.zeros((1, 4, 2))
        coefficients[0, :3, 0] = env.controller_observation(state)["pos"]
        coefficients[0, 0, 1] = 1.0
        coefficients[0, 3, 0] = 0.3
        command = controller.compute_control(
            env.controller_observation(state),
            0,
            trajectory=Trajectory(0, [2], coefficients),
        )
        assert np.isfinite(command).all()
        assert command[2] == 0.3
    finally:
        controller.close()
        env.close()
