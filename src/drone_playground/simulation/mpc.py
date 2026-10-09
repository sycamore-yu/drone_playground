"""LSY attitude MPC and Crazyflow elite-mean sampling MPC with supplied references."""

import os
import site
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import load_fn_params, load_params
from crazyflow.dynamics.so_rpy import dynamics, symbolic_dynamics_euler
from crazyflow.dynamics.utils.rotation import ang_vel2rpy_rates
from crazyflow.sim import Sim
from scipy.spatial.transform import Rotation

from drone_playground.simulation.methods import Setpoint


class SamplingMPC:
    """Pinned Crazyflow sampling controller, without the example's task or hidden state.

    Reference: crazyflow 36f584d, examples/control/sampling.py. Candidate costs,
    elite arithmetic mean, shifted warm start and thrust observer are retained.
    """

    execution = "host"
    output_level = "attitude_thrust"

    def __init__(
        self,
        drone="cf21B_500",
        frequency=50.0,
        samples=2000,
        horizon=25,
        prediction_seconds=1.0,
        device="cpu",
        yaw=0.0,
    ):
        """Configure candidate rollout dimensions and nominal predictive parameters."""
        if min(samples, horizon, frequency, prediction_seconds) <= 0:
            raise ValueError("Sampling MPC dimensions and frequencies must be positive")
        self.samples, self.horizon, self.frequency = samples, horizon, frequency
        self.predict_dt, self.yaw = prediction_seconds / horizon, yaw
        prediction_hz = round(1 / self.predict_dt)
        if not np.isclose(prediction_hz * self.predict_dt, 1):
            raise ValueError("MPC prediction timestep requires an integer simulation frequency")
        self.reference_offsets = (np.arange(horizon) + 1) * self.predict_dt
        self.sim = Sim(
            n_worlds=samples,
            drone=drone,
            dynamics="so_rpy_rotor_drag",
            control="attitude",
            freq=prediction_hz,
            attitude_freq=prediction_hz,
            device=device,
        )
        base, step = self.sim.data, self.sim.build_step_fn()
        hardware = load_params("first_principles", drone)
        self.hover = float(np.asarray(hardware["mass"]).item() * 9.81)
        self.thrust_time = 1 / float(np.asarray(base.params.thrust_dyn_coef).reshape(-1)[0])
        self.neutral = jnp.array([0.0, 0.0, yaw, self.hover])
        angle = np.deg2rad(60)
        lower = jnp.array([-angle, -angle, yaw, 0])
        upper = jnp.array([angle, angle, yaw, 4 * float(np.asarray(hardware["thrust_max"]).item())])
        sigma = jnp.array([0.1, 0.1, 0.0, 0.08])

        def optimize(body, memory, goals, velocities):
            key, sample_key = jax.random.split(memory["key"])
            candidates = (
                jnp.clip(
                    memory["mean"][None]
                    + jax.random.normal(sample_key, (samples, horizon, 4)) * sigma,
                    lower,
                    upper,
                )
                .at[0]
                .set(memory["mean"])
            )
            states = base.states.replace(
                pos=jnp.broadcast_to(body.pos[0], base.states.pos.shape),
                quat=jnp.broadcast_to(body.quat[0], base.states.quat.shape),
                vel=jnp.broadcast_to(body.vel[0], base.states.vel.shape),
                ang_vel=jnp.broadcast_to(body.ang_vel[0], base.states.ang_vel.shape),
                rotor_vel=jnp.full_like(base.states.rotor_vel, memory["thrust"]),
            )

            def predict(data, row):
                command, goal, velocity = row
                data = data.replace(
                    controls=data.controls.replace(
                        attitude=data.controls.attitude.replace(staged_cmd=command[:, None])
                    )
                )
                data = step(data, 1)
                p, v = data.states.pos[:, 0], data.states.vel[:, 0]
                cost = 50 * jnp.sum((p - goal) ** 2, -1) + jnp.sum((v - velocity) ** 2, -1)
                cost += 5 * jnp.sum(command[:, :2] ** 2, -1) + 5 * (command[:, 3] - self.hover) ** 2
                cost += 100 * (command[:, 2] - yaw) ** 2
                return data, cost

            _, costs = jax.lax.scan(
                predict,
                base.replace(states=states),
                (candidates.transpose(1, 0, 2), goals, velocities),
            )
            elite = jnp.argsort(costs.sum(0))[: max(1, int(samples * 0.01))]
            updated = candidates[elite].mean(0)
            times = jnp.arange(horizon) * self.predict_dt
            shifted = jax.vmap(
                lambda control, neutral: jnp.interp(
                    times + 1 / frequency, times, control, right=neutral
                ),
                in_axes=(1, 0),
                out_axes=1,
            )(updated, self.neutral)
            thrust = (
                memory["thrust"] + (updated[0, 3] - memory["thrust"]) / self.thrust_time / frequency
            )
            return updated[0], {"key": key, "mean": shifted, "thrust": thrust}

        self._optimize = jax.jit(optimize)

    def initialize_memory(self, batch, seed=0):
        """Initialize independent candidate history for the single-world host runner."""
        if batch != 1:
            raise ValueError("Host MPC evaluates one independent world at a time")
        return {
            "key": jax.random.PRNGKey(seed),
            "mean": jnp.tile(self.neutral, (self.horizon, 1)),
            "thrust": jnp.asarray(self.hover),
        }

    def __call__(self, physics, reference, memory):
        """Optimize a complete supplied horizon and return the first physical input."""
        positions, velocities, _ = reference
        if positions.shape != (1, self.horizon, 3):
            raise ValueError("Sampling MPC requires the complete future reference horizon")
        command, memory = self._optimize(physics.states, memory, positions[0], velocities[0])
        if not np.isfinite(np.asarray(command)).all():
            raise FloatingPointError("Sampling MPC produced a nonfinite command")
        return Setpoint(command[None]), memory

    def close(self):
        """Release the predictive simulator's native resources."""
        self.sim.close()


class LSYAttitudeMPC:
    """The old LSY nonlinear OCP and acados solver with an external reference horizon."""

    execution = "host"
    output_level = "attitude_thrust"

    def __init__(
        self,
        drone="cf21B_500",
        frequency=50.0,
        horizon=25,
        workdir=".cache/acados-generated",
        acados_source=None,
        python_path=".cache/acados-python",
        yaw=0.0,
    ):
        """Build the pinned LSY problem in a private generated-code directory."""
        source = acados_source or os.environ.get("ACADOS_SOURCE_DIR")
        if source is None or not (Path(source) / "lib/libacados.so").is_file():
            raise FileNotFoundError(
                "LSY MPC needs acados v0.5.1; set ACADOS_SOURCE_DIR and run setup-acados"
            )
        os.environ["ACADOS_SOURCE_DIR"] = str(Path(source).resolve())
        site.addsitedir(str(Path(python_path).resolve()))
        try:
            from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
        except ImportError as error:
            raise ImportError(
                "Install the optional MPC Python interface with pixi run setup-acados"
            ) from error
        self.horizon, self.yaw = int(horizon), float(yaw)
        self.reference_offsets = np.arange(horizon + 1) / frequency
        params = load_fn_params(dynamics, drone)
        hardware = load_params("first_principles", drone)
        self.hover = float(np.asarray(params["mass"]).item() * 9.81)
        rhs, state, control, _ = symbolic_dynamics_euler(
            **{
                k: params[k]
                for k in (
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
        model = AcadosModel()
        model.name, model.x, model.u, model.f_expl_expr = "lsy_attitude", state, control, rhs
        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = horizon
        ocp.cost.cost_type = ocp.cost.cost_type_e = "LINEAR_LS"
        q = np.array([50, 50, 400, 1, 1, 1, 10, 10, 10, 5, 5, 5], float)
        ocp.cost.W, ocp.cost.W_e = np.diag(np.r_[q, [1, 1, 1, 50]]), np.diag(q)
        ocp.cost.Vx = np.vstack([np.eye(12), np.zeros((4, 12))])
        ocp.cost.Vu = np.vstack([np.zeros((12, 4)), np.eye(4)])
        ocp.cost.Vx_e = np.eye(12)
        ocp.cost.yref, ocp.cost.yref_e = np.zeros(16), np.zeros(12)
        ocp.constraints.lbx, ocp.constraints.ubx = np.full(3, -0.5), np.full(3, 0.5)
        ocp.constraints.idxbx = np.array([3, 4, 5])
        # Preserve the predecessor's two factors of four; source adaptation is documented.
        ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, 16 * float(hardware["thrust_min"])])
        ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, 16 * float(hardware["thrust_max"])])
        ocp.constraints.idxbu, ocp.constraints.x0 = np.arange(4), np.zeros(12)
        options = ocp.solver_options
        options.qp_solver, options.hessian_approx = "FULL_CONDENSING_HPIPM", "GAUSS_NEWTON"
        options.integrator_type, options.nlp_solver_type = "ERK", "SQP"
        options.tol, options.qp_solver_cond_N, options.qp_solver_warm_start = 1e-6, horizon, 1
        options.qp_solver_iter_max, options.nlp_solver_max_iter = 20, 50
        options.tf = horizon / frequency
        directory = Path(workdir).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        ocp.code_export_directory = str(directory)
        self.solver = AcadosOcpSolver(ocp, json_file=str(directory / "ocp.json"), verbose=False)
        self.last_status = None

    def initialize_memory(self, batch, seed=0):
        """Reset acados warm-start state between episodes."""
        if batch != 1:
            raise ValueError("Host MPC evaluates one independent world at a time")
        self.solver.reset(reset_qp_solver_mem=1)
        return ()

    def __call__(self, physics, reference, memory):
        """Solve the original nonlinear program for the supplied reference horizon."""
        body = physics.states
        pos, quat, vel, omega = (
            np.asarray(getattr(body, n)[0, 0]) for n in ("pos", "quat", "vel", "ang_vel")
        )
        x0 = np.r_[
            pos, Rotation.from_quat(quat).as_euler("xyz"), vel, ang_vel2rpy_rates(quat, omega)
        ]
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)
        positions, velocities, _ = (np.asarray(x)[0] for x in reference)
        if len(positions) != self.horizon + 1:
            raise ValueError("LSY MPC requires all running and terminal reference stages")
        for i in range(self.horizon + 1):
            target = np.zeros(16 if i < self.horizon else 12)
            target[:3], target[5], target[6:9] = positions[i], self.yaw, velocities[i]
            if i < self.horizon:
                target[15] = self.hover
            self.solver.set(i, "yref", target)
        self.last_status = int(self.solver.solve())
        command = np.asarray(self.solver.get(0, "u"))
        if self.last_status != 0 or not np.isfinite(command).all():
            raise RuntimeError(f"LSY MPC solve failed with status {self.last_status}")
        return Setpoint(jnp.asarray(command)[None]), memory

    def close(self):
        """Release the generated solver instance."""
        self.solver = None
