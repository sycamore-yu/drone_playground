"""Run the pinned LSY AttitudeMPC with private generated acados code."""

from __future__ import annotations

import os
import site
import sys
import time
from pathlib import Path

import numpy as np
from ml_collections import ConfigDict


def acados_directory() -> Path:
    default = Path.cwd() / "tmp/p3p4/optimization/acados"
    root = Path(os.environ.get("ACADOS_SOURCE_DIR", default)).resolve()
    if not (root / "lib/libacados.so").is_file():
        raise FileNotFoundError(
            f"acados v0.5.1 needs a local build: pixi run setup-acados ({root})"
        )
    os.environ["ACADOS_SOURCE_DIR"] = str(root)
    os.environ["ACADOS_INSTALL_DIR"] = str(root)
    site.addsitedir(str(root.parent / "python-deps"))
    interface = str(root / "interfaces/acados_template")
    if interface not in sys.path:
        sys.path.insert(0, interface)
    return root


class LSYAttitudeMPC:
    """Pinned optimization problem, with separately declared execution adaptations."""

    def __init__(self, obs, info, config, *, workdir: Path):
        self.source = acados_directory()
        from drone_playground.control.controllers.mpc.lsy_upstream.attitude_mpc import AttitudeMPC

        config = ConfigDict(config.to_dict() if hasattr(config, "to_dict") else config)
        config.acados_directory = str(Path(workdir).resolve())
        self.native = AttitudeMPC(obs, info, config)
        self.last_diagnostics = {}
        self.delay_predictor = None

    def enable_delay_compensation(self, milliseconds, initial_action):
        """Opt in after setting the task's reference; never read the sampled delay."""
        import casadi as ca

        from drone_playground.control.controllers.mpc.delay_prediction import IssuedCommandPredictor

        if self.delay_predictor is not None:
            raise ValueError("Delay compensation is already configured")
        model = self.native._ocp.model
        rhs = ca.Function("issued_command_rhs", [model.x, model.u], [model.f_expl_expr])
        self.delay_predictor = IssuedCommandPredictor(rhs, milliseconds / 1000, initial_action)
        axis = np.arange(len(self.native._waypoints_pos), dtype=float)
        ahead = axis + self.delay_predictor.delay_seconds / self.native._dt
        for field in ("_waypoints_pos", "_waypoints_vel", "_waypoints_yaw"):
            value = getattr(self.native, field)
            shifted = (
                np.interp(ahead, axis, value)
                if value.ndim == 1
                else np.stack(
                    [np.interp(ahead, axis, value[:, i]) for i in range(value.shape[1])],
                    axis=1,
                )
            )
            setattr(self.native, field, shifted)

    def compute_control(self, obs, info=None):
        tic = time.perf_counter()
        now = self.native._tick * self.native._dt
        if self.delay_predictor is not None:
            obs = self.delay_predictor.predict(obs, now)
        action = np.asarray(self.native.compute_control(dict(obs), info), dtype=np.float64)
        if self.delay_predictor is not None:
            self.delay_predictor.record(now, action)
        self.last_diagnostics = {
            "status": int(self.native.last_status),
            "decision_seconds": time.perf_counter() - tic,
            "solver_seconds": float(self.native._acados_ocp_solver.get_stats("time_tot")),
            "sqp_iterations": int(self.native._acados_ocp_solver.get_stats("sqp_iter")),
            "finite_action": bool(np.isfinite(action).all()),
        }
        if not np.isfinite(action).all():
            raise FloatingPointError("LSY MPC produced a non-finite action")
        return action

    def step_callback(self, *args, **kwargs):
        return self.native.step_callback(*args, **kwargs)

    def compute_trajectory(self, obs, trajectory, time, *, yaw=None):
        """Supply every acados stage from the live plan, retaining the upstream solver."""
        from drone_playground.references import reference_horizon

        if self.delay_predictor is not None:
            raise ValueError("Delay-compensated MPC currently requires a fixed task reference")
        offsets = np.linspace(0, self.native._T_HORIZON, self.native._N + 1)
        reference = reference_horizon(trajectory, time, offsets, yaw=yaw)
        self.native._waypoints_pos = reference["position"]
        self.native._waypoints_vel = reference["velocity"]
        self.native._waypoints_yaw = reference["yaw"]
        self.native._tick, self.native._tick_max = 0, 1
        self.native._finished = False
        return self.compute_control(obs)

    def episode_callback(self):
        self.native.episode_callback()
        self.native._finished = False
        if self.delay_predictor is not None:
            self.delay_predictor.reset()

    def reset(self, seed=0):
        """Reset both controller history and solver warm-start state per episode."""
        del seed
        self.episode_callback()
        self.native._acados_ocp_solver.reset(reset_qp_solver_mem=1)

    def step(self, observation, tick):
        """Return one physical control using the native controller clock."""
        del tick
        return self.compute_control(observation, {})

    def after_step(self, command, observation, reward, done):
        """Advance the native controller lifecycle after physical execution."""
        return self.step_callback(command, observation, reward, done, False, {})

    def close(self):
        self.native = None
