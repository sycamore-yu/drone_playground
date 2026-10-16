"""Arc-length contouring MPC for gate racing using acados and Crazyflow dynamics.

The method follows the standard MPCC contour/lag/progress objective. The course
is the Task's gate-ordered spline, reparameterized by arc length; local stage
tangents are updated on each feedback solve. No third-party controller source
is vendored into this project.
"""

from __future__ import annotations

import os
import site
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import load_fn_params, load_params
from crazyflow.dynamics.so_rpy import dynamics, symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation

from drone_playground.simulation.methods import Setpoint


def arc_length_course(task, *, spacing: float = 0.04) -> tuple[np.ndarray, np.ndarray]:
    """Sample the gate-ordered task path uniformly in Euclidean arc length."""
    if task.name != "racing":
        raise ValueError("MPCC requires a Racing Task with an ordered gate course")
    if not 0 < spacing <= 0.5:
        raise ValueError("Course spacing must be in (0, 0.5] metres")
    knots = np.asarray(task.knots)
    spline = PPoly(np.asarray(task.coefficients), knots)
    dense = spline(np.linspace(float(knots[0]), float(knots[-1]), 2001))
    segment_lengths = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    keep = np.concatenate(([True], np.diff(arc) > 1e-9))
    if arc[-1] < spacing or int(keep.sum()) < 3:
        raise ValueError("Racing course needs a nondegenerate spatial path")
    uniform = np.linspace(0, float(arc[-1]), max(3, int(np.ceil(arc[-1] / spacing)) + 1))
    positions = np.column_stack(
        [np.interp(uniform, arc[keep], dense[keep, dim]) for dim in range(3)]
    )
    return uniform, positions


class RacingMPCC:
    """Acados nonlinear MPC with explicit progress and contouring error state.

    This uses a piecewise linearized *local* arc-length reference each solve;
    unlike a globally parameterized contouring OCP, tangent points are treated
    as stage parameters. The flight plant remains the configured Crazyflow
    dynamics; the predictive model uses Crazyflow's nominal `so_rpy` equations.
    """

    execution = "host"
    output_level = "attitude_thrust"

    def __init__(
        self,
        drone: str = "cf21B_500",
        frequency: float = 50.0,
        horizon: int = 12,
        prediction_seconds: float = 0.36,
        nominal_progress_speed: float = 1.0,
        maximum_progress_speed: float = 1.8,
        contour_weight: float = 150.0,
        lag_weight: float = 8.0,
        progress_weight: float = 0.8,
        control_weight: float = 2.0,
        tilt_limit: float = 0.35,
        workdir: str = ".cache/racing-mpcc",
        acados_source: str | None = None,
        python_path: str = ".cache/acados-python",
    ) -> None:
        """Build a finite-horizon OCP without starting a simulator or ROS node."""
        values = (
            frequency,
            prediction_seconds,
            nominal_progress_speed,
            maximum_progress_speed,
            contour_weight,
            lag_weight,
            progress_weight,
            control_weight,
            tilt_limit,
        )
        if type(horizon) is not int or horizon < 2 or not np.isfinite(values).all():
            raise ValueError("MPCC requires a positive finite horizon and weights")
        if (
            min(frequency, prediction_seconds, nominal_progress_speed, maximum_progress_speed) <= 0
            or min(contour_weight, lag_weight, progress_weight, control_weight) < 0
            or not 0 < tilt_limit < np.pi / 2
            or nominal_progress_speed > maximum_progress_speed
        ):
            raise ValueError("Invalid MPCC objective, horizon, or actuator limits")
        source = acados_source or os.getenv("ACADOS_SOURCE_DIR")
        if not source or not (Path(source) / "lib/libacados.so").is_file():
            raise FileNotFoundError(
                "Racing MPCC needs an acados installation; run pixi run setup-acados "
                "and set ACADOS_SOURCE_DIR"
            )
        os.environ["ACADOS_SOURCE_DIR"] = str(Path(source).resolve())
        site.addsitedir(str(Path(python_path).resolve()))
        try:
            from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
        except ImportError as error:
            raise ImportError("Install the acados Python interface with setup-acados") from error
        import casadi as ca

        self.horizon = horizon
        self.frequency = float(frequency)
        self.reference_offsets = np.arange(horizon + 1) * prediction_seconds / horizon
        self.dt = prediction_seconds / horizon
        self.nominal_speed = nominal_progress_speed
        self.maximum_speed = maximum_progress_speed
        self.tilt_limit = float(tilt_limit)
        self.progress = 0.0
        self.arc = np.empty(0)
        self.course = np.empty((0, 3))
        self.last_status = None

        params = load_fn_params(dynamics, drone)
        hardware = load_params("first_principles", drone)
        self.hover = float(np.asarray(params["mass"]).item() * 9.81)
        self.maximum_thrust = float(4 * np.asarray(hardware["thrust_max"]).item())
        if not 0 < self.hover < self.maximum_thrust:
            raise ValueError("Nominal hover thrust is outside the available actuator range")
        rhs, state, command, _ = symbolic_dynamics_euler(
            **{
                key: params[key]
                for key in (
                    "mass",
                    "gravity_vec",
                    "acc_coef",
                    "cmd_f_coef",
                    "rpy_coef",
                    "rpy_rates_coef",
                    "cmd_rpy_coef",
                )
            }
        )
        progress = ca.MX.sym("arc_progress")
        forward_speed = ca.MX.sym("arc_speed")
        model = AcadosModel()
        model.name = "racing_contouring"
        model.x = ca.vertcat(state, progress)
        model.u = ca.vertcat(command, forward_speed)
        model.f_expl_expr = ca.vertcat(rhs, forward_speed)
        model.p = ca.MX.sym("course_stage", 7)
        position = model.x[:3]
        tangent = model.p[3:6]
        nominal_arc = model.p[6]
        closest = model.p[:3] + tangent * (progress - nominal_arc)
        error = position - closest
        lag = ca.dot(tangent, error)
        contour = error - tangent * lag
        ocp = AcadosOcp()
        ocp.model = model
        # Weighted least squares yields a positive-semidefinite GN QP Hessian.
        # Driving progress speed toward its upper limit is the bounded progress
        # incentive; contour and lag errors keep that progress on the gate path.
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        # Penalize attitude and normalized collective thrust to prevent a
        # contour-only objective from demanding inverted attitudes at gates.
        model.cost_y_expr = ca.vertcat(
            contour,
            lag,
            model.x[3:6],
            command[:3],
            command[3] / self.hover,
            forward_speed,
        )
        model.cost_y_expr_e = ca.vertcat(contour, lag)
        ocp.cost.W = np.diag(
            [contour_weight] * 3
            + [lag_weight]
            + [12.0, 12.0, 5.0]
            + [4 * control_weight, 4 * control_weight, 2 * control_weight]
            + [5 * control_weight]
            + [max(progress_weight, 1e-7)]
        )
        ocp.cost.W_e = np.diag([contour_weight * 2] * 3 + [lag_weight * 2])
        ocp.cost.yref = np.r_[np.zeros(10), 1.0, maximum_progress_speed]
        ocp.cost.yref_e = np.zeros(4)
        ocp.solver_options.N_horizon = horizon
        ocp.solver_options.tf = prediction_seconds
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver_warm_start = 1
        ocp.solver_options.qp_solver_iter_max = 30
        ocp.solver_options.tol = 1e-5
        ocp.constraints.x0 = np.zeros(13)
        ocp.constraints.idxbu = np.arange(5)
        ocp.constraints.lbu = np.array([-tilt_limit, -tilt_limit, -np.pi, 0.4 * self.hover, 0.0])
        ocp.constraints.ubu = np.array(
            [
                tilt_limit,
                tilt_limit,
                np.pi,
                min(self.maximum_thrust, 1.6 * self.hover),
                maximum_progress_speed,
            ]
        )
        ocp.constraints.idxbx = np.array([2, 3, 4])
        ocp.constraints.lbx = np.array([0.12, -tilt_limit, -tilt_limit])
        ocp.constraints.ubx = np.array([2.0, tilt_limit, tilt_limit])
        ocp.parameter_values = np.r_[np.zeros(3), [1.0, 0.0, 0.0], 0.0]
        root = Path(workdir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        ocp.code_export_directory = str(root / "c_generated_code")
        self.solver = AcadosOcpSolver(ocp, json_file=str(root / "ocp.json"), verbose=False)

    def set_course(self, task) -> None:
        """Bind the Racing Task's published gate order before episode execution."""
        self.arc, self.course = arc_length_course(task)

    def initialize_memory(self, batch: int, seed: int = 0):
        """Reset progress and acados warm start for an independent flight."""
        if batch != 1:
            raise ValueError("acados MPC runs one physical world per solve")
        if not len(self.arc):
            raise ValueError("Bind a Racing Task before creating MPC memory")
        self.progress = 0.0
        self.last_status = None
        self.solver.reset(reset_qp_solver_mem=1)
        return ()

    def _sample_course(self, progress: float) -> np.ndarray:
        """Return position, normalized tangent, and arc station at a progress value."""
        station = float(np.clip(progress, 0, self.arc[-1]))
        index = int(np.clip(np.searchsorted(self.arc, station) - 1, 0, len(self.arc) - 2))
        start, end = self.arc[index : index + 2]
        alpha = np.clip((station - start) / (end - start), 0.0, 1.0)
        direction = (self.course[index + 1] - self.course[index]) / (end - start)
        direction /= max(np.linalg.norm(direction), 1e-10)
        point = (1 - alpha) * self.course[index] + alpha * self.course[index + 1]
        return np.r_[point, direction, station]

    def _project_progress(self, position: np.ndarray) -> float:
        """Update progress locally so a self-crossing track cannot skip future gates."""
        # A quadrotor cannot advance multiple metres along the ordered route
        # in one controller tick. A global nearest-point projection would
        # jump to later laps or repeated gates at track self-intersections.
        back = max(0.0, self.progress - 0.12)
        forward = min(
            float(self.arc[-1]),
            self.progress + 0.06 + 2.0 * self.maximum_speed / self.frequency,
        )
        available = np.flatnonzero((self.arc >= back) & (self.arc <= forward))
        if not len(available):
            raise ValueError("No Racing course samples in the local projection window")
        nearest = available[np.argmin(np.linalg.norm(self.course[available] - position, axis=1))]
        self.progress = max(self.progress, float(self.arc[nearest]))
        return self.progress

    def __call__(self, physics, reference, memory):
        """Solve MPCC and return a Crazyflow attitude/thrust setpoint."""
        del reference
        body = physics.states
        pos, quat, vel, omega = (
            np.asarray(getattr(body, name)[0, 0]) for name in ("pos", "quat", "vel", "ang_vel")
        )
        self._project_progress(pos)
        x0 = np.r_[
            pos,
            Rotation.from_quat(quat).as_euler("xyz"),
            vel,
            ang_vel2rpy_rates(quat, omega),
            self.progress,
        ]
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        for stage in range(self.horizon + 1):
            planned_arc = self.progress + stage * self.dt * self.nominal_speed
            self.solver.set(stage, "p", self._sample_course(planned_arc))
        self.last_status = int(self.solver.solve())
        command = np.asarray(self.solver.get(0, "u"), dtype=float)
        if self.last_status != 0 or not np.isfinite(command).all():
            raise RuntimeError(f"acados MPCC solve failed: status {self.last_status}")
        value = np.clip(
            command[:4],
            [-self.tilt_limit, -self.tilt_limit, -np.pi, 0.4 * self.hover],
            [self.tilt_limit, self.tilt_limit, np.pi, min(self.maximum_thrust, 1.6 * self.hover)],
        )
        return Setpoint(jnp.asarray(value, dtype=jnp.float32)[None]), memory

    def close(self) -> None:
        """Release solver references and generated solver memory."""
        self.solver = None
