"""Trajectory-tracking quadrotor NMPC using acados and a 10-state rigid body model.

Based on the state/control and reference conventions of RPS98/mpc (BSD-3-Clause):
  x = [world position, body-to-world quaternion wxyz, world velocity]
  u = [collective thrust in N, body angular rates in rad/s]

This is a native, independently written Crazyflow adapter rather than a copy
of the upstream solver. The Crazyflow actuator and flight dynamics remain the
closed-loop plant; the OCP is a nominal rigid-body predictive approximation.
"""

from __future__ import annotations

import os
import site
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import load_params
from scipy.spatial.transform import Rotation

from drone_playground.simulation.methods import Setpoint


def reference_orientation(acceleration, yaw: float, mass: float) -> tuple[np.ndarray, float]:
    """Convert world acceleration and heading into body-to-world wxyz and thrust.

    This uses the differential-flatness thrust-axis construction: a planner
    provides a(t); NMPC derives the desired tilt and collective thrust itself.
    """
    acceleration = np.asarray(acceleration, dtype=float)
    if acceleration.shape != (3,) or not np.isfinite([*acceleration, yaw, mass]).all() or mass <= 0:
        raise ValueError("NMPC reference must have finite acceleration, yaw and mass")
    force = acceleration + np.array([0.0, 0.0, 9.81])
    magnitude = float(np.linalg.norm(force))
    if magnitude < 1e-6:
        raise ValueError("NMPC reference cannot specify zero net specific force")
    z_axis = force / magnitude
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y_axis = np.cross(z_axis, heading)
    if np.linalg.norm(y_axis) < 1e-6:
        y_axis = np.cross(z_axis, np.array([0.0, 1.0, 0.0]))
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    quaternion_xyzw = Rotation.from_matrix(np.column_stack((x_axis, y_axis, z_axis))).as_quat()
    return np.roll(quaternion_xyzw, 1), mass * magnitude


class FullStateNMPC:
    """A general time-parameterized trajectory tracker, independent of Task kind.

    RPS98/mpc supplies the full-state and body-rate OCP architecture. The
    reference type and physical executor follow Drone Playground conventions.
    This controller does not predict obstacles or optimize progress; a Planner
    must provide an obstacle-free time-valid trajectory for Navigation.
    """

    execution = "host"
    output_level = "body_rate"

    def __init__(
        self,
        drone: str = "cf21B_500",
        frequency: float = 50.0,
        horizon: int = 10,
        prediction_seconds: float = 0.3,
        position_weight: float = 70.0,
        velocity_weight: float = 12.0,
        attitude_weight: float = 5.0,
        rate_weight: float = 0.4,
        thrust_weight: float = 3.0,
        rate_limit: float = 5.0,
        yaw: float = 0.0,
        workdir: str = ".cache/acados-full-state-nmpc",
        acados_source: str | None = None,
        python_path: str = ".cache/acados-python",
    ) -> None:
        """Construct the reference-tracking OCP and compile its acados solver."""
        if type(horizon) is not int or horizon < 2:
            raise ValueError("NMPC horizon must have at least two shooting stages")
        scalars = (
            frequency,
            prediction_seconds,
            position_weight,
            velocity_weight,
            attitude_weight,
            rate_weight,
            thrust_weight,
            rate_limit,
            yaw,
        )
        if (
            not np.isfinite(scalars).all()
            or frequency <= 0
            or prediction_seconds <= 0
            or rate_limit <= 0
            or min(
                position_weight,
                velocity_weight,
                attitude_weight,
                rate_weight,
                thrust_weight,
            )
            < 0
        ):
            raise ValueError("NMPC frequency, weights and physical limits must be finite")
        source = acados_source or os.environ.get("ACADOS_SOURCE_DIR")
        if not source or not (Path(source) / "lib/libacados.so").is_file():
            raise FileNotFoundError("NMPC requires acados; run pixi run setup-acados")
        os.environ["ACADOS_SOURCE_DIR"] = str(Path(source).resolve())
        site.addsitedir(str(Path(python_path).resolve()))
        try:
            from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
        except ImportError as error:
            raise ImportError("Install the optional acados Python interface") from error
        import casadi as ca

        hardware = load_params("first_principles", drone)
        self.mass = float(np.asarray(hardware["mass"]).item())
        self.hover = self.mass * 9.81
        motor_thrust = float(np.asarray(hardware["thrust_max"]).item())
        self.thrust_upper = min(1.6 * self.hover, 4 * motor_thrust)
        self.thrust_lower = 0.4 * self.hover
        if self.thrust_upper <= self.thrust_lower:
            raise ValueError("NMPC nominal actuator thrust bounds are invalid")
        self.yaw = float(yaw)
        self.horizon = horizon
        self.reference_offsets = np.arange(horizon + 1) * prediction_seconds / horizon
        self.last_status = None

        state = ca.SX.sym("x", 10)
        control = ca.SX.sym("u", 4)
        desired_quaternion = ca.SX.sym("q_reference", 4)
        quaternion = state[3:7] / ca.sqrt(ca.sumsqr(state[3:7]) + 1e-12)
        qw, qx, qy, qz = [quaternion[i] for i in range(4)]
        wx, wy, wz = [control[i] for i in range(1, 4)]
        quaternion_derivative = 0.5 * ca.vertcat(
            -qx * wx - qy * wy - qz * wz,
            qw * wx + qy * wz - qz * wy,
            qw * wy + qz * wx - qx * wz,
            qw * wz + qx * wy - qy * wx,
        )
        thrust_axis = ca.vertcat(
            2 * (qx * qz + qw * qy),
            2 * (qy * qz - qw * qx),
            1 - 2 * (qx * qx + qy * qy),
        )
        body_acceleration = control[0] / self.mass * thrust_axis - ca.vertcat(0, 0, 9.81)
        model = AcadosModel()
        model.name = "full_state_nmpc"
        model.x = state
        model.u = control
        model.p = desired_quaternion
        model.f_expl_expr = ca.vertcat(state[7:10], quaternion_derivative, body_acceleration)

        # Orientation error is a 3D quaternion residual around q_reference;
        # quaternion components themselves must not be optimized as 4D angles.
        rw, rx, ry, rz = [desired_quaternion[i] for i in range(4)]
        error_xyz = ca.vertcat(
            rw * qx - rx * qw - ry * qz + rz * qy,
            rw * qy + rx * qz - ry * qw - rz * qx,
            rw * qz - rx * qy + ry * qx - rz * qw,
        )
        running = ca.vertcat(
            state[:3], 2 * error_xyz, state[7:10], control[0] / self.hover, control[1:4]
        )
        terminal = ca.vertcat(state[:3], 2 * error_xyz, state[7:10])
        model.cost_y_expr = running
        model.cost_y_expr_e = terminal

        ocp = AcadosOcp()
        ocp.model = model
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        running_weights = [position_weight] * 3 + [attitude_weight] * 3
        running_weights += [velocity_weight] * 3 + [thrust_weight] + [rate_weight] * 3
        ocp.cost.W = np.diag(running_weights)
        ocp.cost.W_e = np.diag(
            [2 * position_weight] * 3 + [attitude_weight] * 3 + [velocity_weight] * 3
        )
        ocp.cost.yref = np.r_[np.zeros(9), 1.0, np.zeros(3)]
        ocp.cost.yref_e = np.zeros(9)
        ocp.constraints.idxbu = np.arange(4)
        ocp.constraints.lbu = np.r_[self.thrust_lower, [-rate_limit] * 3]
        ocp.constraints.ubu = np.r_[self.thrust_upper, [rate_limit] * 3]
        ocp.constraints.x0 = np.r_[np.zeros(3), [1.0, 0.0, 0.0, 0.0], np.zeros(3)]
        ocp.parameter_values = np.array([1.0, 0.0, 0.0, 0.0])
        options = ocp.solver_options
        options.N_horizon = horizon
        options.tf = prediction_seconds
        options.qp_solver = "FULL_CONDENSING_HPIPM"
        options.hessian_approx = "GAUSS_NEWTON"
        options.integrator_type = "ERK"
        options.nlp_solver_type = "SQP_RTI"
        options.qp_solver_warm_start = 1
        options.qp_solver_iter_max = 30
        options.tol = 1e-5
        directory = Path(workdir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        ocp.code_export_directory = str(directory / "c_generated_code")
        self.solver = AcadosOcpSolver(ocp, json_file=str(directory / "ocp.json"), verbose=False)

    def initialize_memory(self, batch: int, seed: int = 0):
        """Reset the solver's per-episode warm start, without resetting any simulator."""
        if batch != 1:
            raise ValueError("Full-State NMPC executes one physical world per solve")
        self.solver.reset(reset_qp_solver_mem=1)
        self.last_status = None
        return ()

    def __call__(self, physics, reference, memory):
        """Track a common world-frame (position, velocity, acceleration) reference."""
        positions, velocities, accelerations = (np.asarray(x)[0] for x in reference)
        if (
            positions.shape != (self.horizon + 1, 3)
            or velocities.shape != positions.shape
            or accelerations.shape != positions.shape
            or not all(np.isfinite(x).all() for x in (positions, velocities, accelerations))
        ):
            raise ValueError("Full-State NMPC requires finite N+1 position/velocity/acceleration")
        body = physics.states
        position, quat_xyzw, velocity = (
            np.asarray(getattr(body, name)[0, 0]) for name in ("pos", "quat", "vel")
        )
        x0 = np.r_[position, np.roll(quat_xyzw, 1), velocity]
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        for i in range(self.horizon + 1):
            quaternion, thrust = reference_orientation(accelerations[i], self.yaw, self.mass)
            target = np.r_[positions[i], np.zeros(3), velocities[i]]
            if i < self.horizon:
                target = np.r_[
                    target,
                    np.clip(thrust, self.thrust_lower, self.thrust_upper) / self.hover,
                    np.zeros(3),
                ]
            self.solver.set(i, "p", quaternion)
            self.solver.set(i, "yref", target)
        self.last_status = int(self.solver.solve())
        solution = np.asarray(self.solver.get(0, "u"), dtype=float)
        if self.last_status != 0 or solution.shape != (4,) or not np.isfinite(solution).all():
            raise RuntimeError(f"Full-State NMPC acados solve failed: status {self.last_status}")
        # acados uses [thrust, wx, wy, wz], while Crazyflow stages [wx, wy, wz, thrust].
        command = jnp.asarray(np.r_[solution[1:4], solution[0]], dtype=jnp.float32)[None]
        return Setpoint(command, level="body_rate", frame="body"), memory

    def close(self):
        """Release references to the generated solver."""
        self.solver = None
