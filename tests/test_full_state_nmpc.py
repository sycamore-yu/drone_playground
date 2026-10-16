"""Full-state quadrotor NMPC uses the shared physical tracking-controller contract."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from drone_playground.simulation.factory import create_environment, create_method
from drone_playground.simulation.full_state_nmpc import FullStateNMPC, reference_orientation
from drone_playground.simulation.methods import Trajectory


def compose_controller(*overrides):
    """Resolve the same Hydra inputs provided by the public Command Builder."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[1] / "src/drone_playground/configs"),
    ):
        return compose(
            config_name="config",
            overrides=[
                "task=tracking",
                "scene=empty",
                "sensor=state",
                "method=controller",
                "controller=full_state_nmpc",
                "simulation.device=cpu",
                "simulation.num_envs=1",
                *overrides,
            ],
        )


def test_reference_attitude_matches_acceleration_and_yaw():
    """Flatness-derived feedforward is physically meaningful and normalized."""
    level, collective = reference_orientation(np.zeros(3), yaw=0.0, mass=0.03)
    np.testing.assert_allclose(level, [1.0, 0.0, 0.0, 0.0], atol=1e-7)
    assert collective == pytest.approx(0.03 * 9.81)
    upward, thrust = reference_orientation(np.array([2.0, 0.0, 0.0]), 0.0, 0.03)
    assert np.linalg.norm(upward) == pytest.approx(1.0)
    assert upward[2] > 0.0
    assert thrust > collective
    with pytest.raises(ValueError, match="finite"):
        reference_orientation(np.array([np.nan, 0.0, 0.0]), 0.0, 0.03)


def test_controller_config_requires_full_physical_body_rate_executor():
    """Hydra routes RPS-inspired thrust/rate output to Crazyflow BodyRate control."""
    config = compose_controller()
    assert config.controller._target_.endswith("FullStateNMPC")
    assert FullStateNMPC.output_level == "body_rate"
    assert config.simulation.dynamics == "first_principles"
    env = create_environment(config)
    assert env.control_level == "body_rate"


def test_acados_nmpc_runs_short_real_crazyflow_tracking_when_installed(tmp_path):
    """Closed-loop positional tracking must use an actual compiled acados solve."""
    import os

    from drone_playground.simulation.runner import rollout_host

    source = os.getenv("ACADOS_SOURCE_DIR")
    if not source or not Path(source, "lib/libacados.so").is_file():
        pytest.skip("Install acados to run the native NMPC flight test")
    cfg = compose_controller(
        "controller.horizon=5",
        "controller.prediction_seconds=0.15",
        f"controller.workdir={tmp_path}/acados",
        "task.duration=0.12",
    )
    env = create_environment(cfg)
    method = create_method(cfg)
    try:
        episodes, traces, _, _ = rollout_host(env, method, seed=17)
        assert len(episodes) == 1
        assert episodes[0]["method_failure"] == ""
        assert len(traces["time"]) > 2
        assert method.controller.last_status == 0
        assert np.isfinite(traces["position"]).all()
    finally:
        method.close()


@pytest.mark.parametrize("task,scene", [("racing", "racing"), ("navigation", "S01")])
def test_controller_supported_by_racing_and_navigation_configurations(task, scene):
    """The tracker accepts timestamped target references, not race-specific progress."""
    config = compose_controller(f"task={task}", f"scene={scene}")
    env = create_environment(config)
    assert env.task.name == task
    assert env.control_level == "body_rate"


def test_nmpc_consumes_native_planner_trajectory_without_a_ros_worker(tmp_path):
    """EGO/SUPER's shared Trajectory output can feed the NMPC/body-rate executor."""
    import os

    source = os.getenv("ACADOS_SOURCE_DIR")
    if not source or not Path(source, "lib/libacados.so").is_file():
        pytest.skip("acados required for physical controller/trajectory test")
    config = compose_controller(
        "task=navigation",
        "scene=S01",
        "controller.horizon=5",
        "controller.prediction_seconds=0.15",
        f"controller.workdir={tmp_path}/acados",
    )
    env = create_environment(config)
    state = env.reset(jax.random.PRNGKey(93))
    method = create_method(config)
    memory = method.initialize(env, 93)
    try:
        start = np.asarray(state.physics.states.pos[0, 0])
        timestamps = np.linspace(0, 0.3, 11)
        path = Trajectory(
            times=timestamps,
            positions=start[None] + timestamps[:, None] * np.array([0.2, 0.0, 0.0]),
            velocities=np.tile([0.2, 0.0, 0.0], (len(timestamps), 1)),
            accelerations=np.zeros((len(timestamps), 3)),
        )
        reference = tuple(
            jnp.asarray(np.stack([path.sample(t)[i] for t in method.controller.reference_offsets]))[
                None
            ]
            for i in range(3)
        )
        control, _ = method.controller(state.physics, reference, memory["controller"])
        assert control.level == "body_rate" and control.frame == "body"
        assert control.value.shape == (1, 4)
        next_state = env.step_setpoint(state, control)
        assert np.isfinite(np.asarray(next_state.physics.states.pos)).all()
    finally:
        method.close()
