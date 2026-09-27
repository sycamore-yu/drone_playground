#!/usr/bin/env python3
"""Compare reconstructed ideal measurements with preserved native input packets."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--difficulty", default="easy")
    parser.add_argument("--case", type=int, default=0)
    parser.add_argument("--tick", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for run_id in args.run_id:
        run = ROOT / "experiments" / run_id
        source = run / "native" / args.difficulty / str(args.case) / f"sensor-packet-{args.tick:04d}.json"
        reconstruction = run / "sensor-reconstruction" / args.difficulty / f"case-{args.case:03d}" / f"tick-{args.tick:04d}.npz"
        packet = json.loads(source.read_text())
        with np.load(reconstruction, allow_pickle=False) as data:
            if "depth" in packet:
                original = np.frombuffer(base64.b64decode(packet["depth"]), dtype="<f4").reshape(packet["height"], packet["width"])
                recovered = data["depth"]
            else:
                original = np.frombuffer(base64.b64decode(packet["points"]), dtype="<f4").reshape(-1, 4)[:, :3]
                recovered = data["points_world"][data["valid"]]
            equal_shape = recovered.shape == original.shape
            error = float(np.abs(recovered - original).max()) if equal_shape else None
            rows.append(dict(run_id=run_id, tick=args.tick, time_s=packet["time"],
                             original_shape=list(original.shape), reconstructed_shape=list(recovered.shape),
                             max_absolute_error_m=error, tolerance_m=2e-5,
                             passed=equal_shape and error <= 2e-5,
                             source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             reconstruction=str(reconstruction.relative_to(ROOT)),
                             reconstruction_sha256=hashlib.sha256(reconstruction.read_bytes()).hexdigest()))
    result = dict(passed=all(x["passed"] for x in rows), checks=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
