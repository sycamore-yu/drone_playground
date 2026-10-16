"""Run the native replacement gate; this is not a runtime sensor backend.

Usage: JAX_PLATFORMS=cpu python tools/check_sensor_migration.py --output report.json
Exit 1 means the candidate did not satisfy the existing MuJoCo range contract.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco_lidar.core_jax import MjLidarJax


def check() -> dict:
    """Compare explicit boundary cases against MuJoCo's C ray implementation."""
    cases = [
        ("sphere", "1", [0, 0, 0], [1, 0, 0]),
        ("box", "1 1 1", [0, 0, 0], [1, 0, 0]),
        ("cylinder", "1 2", [0, 0, 0], [0, 0, 1]),
        ("capsule", "1 2", [0, 0, 0], [0, 0, 1]),
        ("plane", "0 0 .1", [0, 0, -1], [0, 0, 1]),
        ("box", "1 1 1", [-2, 1, 0], [1, 0, 0]),
        ("box", "1 1 1", [-2, 0, 0], [1, 0, 0]),
    ]
    rows = []
    for kind, size, origin, direction in cases:
        model = mujoco.MjModel.from_xml_string(
            f'<mujoco><worldbody><geom type="{kind}" size="{size}"/></worldbody></mujoco>'
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        p, d = np.array(origin, dtype=float), np.array(direction, dtype=float)
        reference = float(mujoco.mj_ray(model, data, p, d, None, 1, -1, np.zeros(1, np.int32)))
        actual = float(
            MjLidarJax(model).render(
                jnp.array(data.geom_xpos),
                jnp.array(data.geom_xmat),
                jnp.array(p),
                jnp.array(d[None]),
            )[0]
        )
        rows.append(
            {
                "kind": kind,
                "origin": origin,
                "direction": direction,
                "mujoco": reference,
                "mujoco_lidar": actual,
                "match": bool(np.isclose(actual, reference, atol=1e-5)),
            }
        )
    return {
        "mujoco_version": version("mujoco"),
        "candidate_version": version("mujoco-lidar"),
        "passed": all(row["match"] for row in rows),
        "cases": rows,
    }


def main() -> int:
    """Write every comparison, including failures, and return the actual gate status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
