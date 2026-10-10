"""Compare frozen Racing APG/SHAC on identical training-diagnostic resets.

The script does not read or write a frozen benchmark, and never selects a
checkpoint. It records actual applied low-level controls and gate events.

Example:
  python tools/diagnose_racing.py \
    --apg results/scratch_v02/racing_apg_s0/checkpoints/step-000400/policy.zip \
    --shac results/scratch_v02/racing_shac_s0/checkpoints/step-001000/policy.zip \
    --output tmp/racing_apg_vs_shac.json
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
from omegaconf import OmegaConf

from drone_playground.simulation.evaluation import create_environment, create_method
from drone_playground.simulation.policy import load_policy, read_checkpoint
from drone_playground.simulation.records import atomic_json
from drone_playground.simulation.runner import rollout


def diagnose(path: Path, seed: int, duration: float) -> dict:
    """Trace one identically initialized policy on the physical Racing course."""
    _, identity = read_checkpoint(path, "inference")
    config = OmegaConf.create(identity["config"]["experiment"])
    if config.task.name != "racing" or config.sensor.name != "state":
        raise ValueError("Expected a frozen State Racing policy checkpoint")
    config.mode = "checkpoint_eval"
    # The diagnostic may run without NVIDIA and does not change physics.
    config.simulation.device = "cpu"
    config.task.duration = duration
    env = create_environment(config, num_envs=1)
    method = create_method(config)
    try:
        actor, parameters, _ = load_policy(path)
        episodes, trajectories = rollout(
            env,
            seed=seed,
            method=method,
            actor=actor,
            parameters=parameters,
            chunk_steps=25,
            record=True,
        )
        row = episodes[0]
        times = np.asarray(trajectories["time"][:, 0])
        positions = np.asarray(trajectories["position"][:, 0])
        commands = np.asarray(trajectories["command"][:, 0])
        gates = np.asarray(trajectories["gates"][:, 0])
        valid = np.flatnonzero(np.r_[True, np.diff(times) > 0])
        times, positions, commands, gates = (
            arr[valid] for arr in (times, positions, commands, gates)
        )
        gate_change = np.flatnonzero(np.diff(gates) > 0)
        first_gate = np.asarray(env.task.gate_positions[env.task.gate_order[0]])
        return {
            "checkpoint": str(path),
            "seed": seed,
            "duration_s": duration,
            "event": row["event"],
            "flight_seconds": row["flight_seconds"],
            "gates_passed": row["gates_passed"],
            "gate_crossings_seconds": times[gate_change + 1].tolist(),
            "first_gate_min_distance_m": float(
                np.linalg.norm(positions - first_gate, axis=-1).min()
            ),
            "final_position_m": positions[-1].tolist(),
            "min_clearance_m": row["min_clearance"],
            "control_output_shape": list(commands.shape),
            "applied_control_mean": commands.mean(axis=0).tolist(),
            "applied_control_abs_max": np.abs(commands).max(axis=0).tolist(),
            "applied_control_delta_abs_mean": (
                np.abs(np.diff(commands, axis=0)).mean(axis=0).tolist() if len(commands) > 1 else []
            ),
            "applied_control_all_finite": bool(np.isfinite(commands).all()),
        }
    finally:
        method.close()


def main():
    """Record one comparable roll-out without contaminating the test partition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apg", required=True, type=Path)
    parser.add_argument("--shac", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1_001_000)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1_000_000 <= args.seed < 1_500_000:
        raise ValueError("Only original training-diagnostic reset seeds are allowed")
    if not 0 < args.duration <= 60:
        raise ValueError("Diagnostic Racing duration must be in (0, 60] seconds")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite earlier trace: {args.output}")
    report = {
        "seed_partition": "training_diagnostic",
        "apg": diagnose(args.apg, args.seed, args.duration),
        "shac": diagnose(args.shac, args.seed, args.duration),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
