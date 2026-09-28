"""Read phase records through the shipped RScope Viewer, without changing them."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def hashes(directory):
    return {
        p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reader", type=Path, required=True, help="Verified RScope Viewer python/bridge.py"
    )
    parser.add_argument("--directory", type=Path, action="append", default=[])
    parser.add_argument("--matrix", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "rscope_verification_reader", args.reader.resolve()
    )
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    directories = list(args.directory)
    for matrix in args.matrix:
        for row in json.loads(matrix.read_text())["rows"]:
            if row.get("status") != "completed":
                continue
            run = root / "experiments" / row["run_id"]
            directories.extend(
                run / f"independent-{split}" / "rollouts" for split in ("dev", "heldout")
            )
    checked = []
    for directory in dict.fromkeys(p.resolve() for p in directories):
        before = hashes(directory)
        files = sorted(directory.glob("*.mj_unroll"))
        if not files:
            raise FileNotFoundError(f"Missing replay file in {directory}")
        for file in files:
            replay = bridge.Replay(file)
            mocap_id = int(replay.model.body_mocapid[replay.focus])
            if mocap_id < 0:
                raise ValueError("Expected Crazyflow mocap-backed drone")
            error = 0.0
            for case in range(replay.episodes):
                decoded = replay.episode(case)
                actual = np.asarray(decoded["actual"])
                expected = replay.record["mocap_pos"][:, case, mocap_id]
                error = max(error, float(np.max(np.abs(actual - expected))))
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9)
                if all(
                    k in replay.record["metrics"]
                    for k in ("reference_x", "reference_y", "reference_z")
                ):
                    expected_ref = np.stack(
                        [
                            replay.record["metrics"][k][:, case]
                            for k in ("reference_x", "reference_y", "reference_z")
                        ],
                        axis=-1,
                    )
                    np.testing.assert_array_equal(decoded["reference"], expected_ref)
            entry = dict(
                file=str(file.relative_to(root)),
                frames=replay.frames,
                saved_cases=replay.episodes,
                model_bodies=replay.model.nbody,
                max_position_error=error,
                metadata_sha256=replay.source_hash,
                unchanged_files=before,
            )
            checked.append(entry)
            print(
                json.dumps({k: v for k, v in entry.items() if k != "unchanged_files"}), flush=True
            )
        if hashes(directory) != before:
            raise RuntimeError(f"Replay changed source directory: {directory}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                passed=True,
                reader_sha256=hashlib.sha256(args.reader.read_bytes()).hexdigest(),
                file_count=len(checked),
                saved_cases=sum(x["saved_cases"] for x in checked),
                records=checked,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
