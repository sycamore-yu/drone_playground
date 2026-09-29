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
    default = Path(__file__).resolve().parents[4] / "tmp/p3p4/optimization/acados"
    root = Path(os.environ.get("ACADOS_SOURCE_DIR", default)).resolve()
    if not (root / "lib/libacados.so").is_file():
        raise FileNotFoundError(
            f"acados v0.5.1 needs a local build: bash scripts/tools/setup_acados.sh ({root})"
        )
    os.environ["ACADOS_SOURCE_DIR"] = str(root)
    os.environ["ACADOS_INSTALL_DIR"] = str(root)
    site.addsitedir(str(root.parent / "python-deps"))
    interface = str(root / "interfaces/acados_template")
    if interface not in sys.path:
        sys.path.insert(0, interface)
    return root


class LSYAttitudeMPC:
    """The upstream controller unchanged at its optimization/control interface."""

    def __init__(self, obs, info, config, *, workdir: Path):
        self.source = acados_directory()
        from .lsy_upstream.attitude_mpc import AttitudeMPC

        config = ConfigDict(config.to_dict() if hasattr(config, "to_dict") else config)
        config.acados_directory = str(Path(workdir).resolve())
        self.native = AttitudeMPC(obs, info, config)
        self.last_diagnostics = {}

    def compute_control(self, obs, info=None):
        tic = time.perf_counter()
        action = np.asarray(self.native.compute_control(dict(obs), info), dtype=np.float64)
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
        from drone_playground.execution.reference import reference_horizon

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

    def close(self):
        self.native = None
