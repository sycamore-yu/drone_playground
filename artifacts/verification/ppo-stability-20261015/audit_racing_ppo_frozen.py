"""Check the fixed Racing PPO cohort, loaded models and actual portable replays."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = Path(__file__).resolve().parent


def module(name, path):
    """Reuse the project's collector and native-viewer round-trip checks."""
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


acceptance = module("acceptance", ROOT / "tools/report_acceptance.py")
replay = module("replay_checks", ROOT / "tests/test_replay.py")
results, recipes, seen_cases = [], [], set()
for seed in (0, 1, 2):
    training = ROOT / f"results/racing_common_reward_p1f0/ppo_reference_options_s{seed}"
    frozen = ROOT / f"results/racing_common_reward_p1f0/frozen4m/ppo_s{seed}"
    run = acceptance.collect_run(training, ROOT, benchmark_run=frozen)
    verdict = acceptance.seed_result(run, ("racing", "state", None))
    assert verdict["status"] == "passed", verdict
    assert run["benchmark"]["checkpoint_link"]["passed"]
    recipes.append(acceptance.recipe(run["config"]))
    directory = frozen / "eval/001"
    report = json.loads((directory / "report.json").read_text())
    assert report["seed_base"] == 4_000_000
    scene = report["reports"]["racing"]
    assert scene["episodes"] == scene["successes"] == 100
    cases = set(run["benchmark"]["seeds"])
    assert cases and not cases & seen_cases
    seen_cases |= cases
    decoded = []
    with np.load(directory / "trajectories.npz") as raw:
        for relative in scene["replays"]:
            path = directory / relative
            index = int(path.parent.name.rsplit("-", 1)[1])
            times = raw["racing__0000__time"][:, index]
            keep = np.r_[True, np.diff(times) > 0]
            with pytest.MonkeyPatch.context() as patches:
                record, model, data, meta = replay.decode(path, patches)
                np.testing.assert_array_equal(record.time[:, 0], times[keep])
                np.testing.assert_array_equal(
                    record.qpos[:, 0, :3], raw["racing__0000__position"][keep, index]
                )
                assert meta["replay"]["geometry_hash"] == scene["geometry"]["sha256"]
                for asset, expected in meta["replay"]["asset_sha256"].items():
                    assert acceptance.digest(path.parent / asset) == expected
                for frame in (0, len(record.time) // 2, len(record.time) - 1):
                    replay.restore(record, model, data, frame)
                    np.testing.assert_allclose(
                        data.xpos[model.body("drone").id], record.qpos[frame, 0, :3], atol=1e-8
                    )
                decoded.append(
                    {
                        "path": str(path),
                        "sha256": acceptance.digest(path),
                        "frames": len(record.time),
                        "record_matches_raw_trajectory": True,
                        "native_viewer_loader_and_mujoco_restore": True,
                        "asset_hashes_verified": True,
                    }
                )
    assert len(decoded) == 2
    results.append(
        {
            "seed": seed,
            "verdict": verdict,
            "selected_payload_sha256": run["checkpoint"]["payload_sha256"],
            "loaded_checkpoint_link": run["benchmark"]["checkpoint_link"],
            "frozen_run": str(frozen),
            "frozen_csv_sha256": acceptance.digest(directory / "episodes.csv"),
            "frozen_report_sha256": acceptance.digest(directory / "report.json"),
            "trajectories_sha256": acceptance.digest(directory / "trajectories.npz"),
            "successes": 100,
            "episodes": 100,
            "descriptive_wilson_95_interval": [100 / (100 + 1.96**2), 1.0],
            "replays": decoded,
        }
    )
assert recipes[0] == recipes[1] == recipes[2]
output = {
    "scope": "Three independently initialized Racing PPO seeds, frozen4million cohort. "
    "C1/C3/C5/C6 and source links verified. The matrix's cross-algorithm Actor check "
    "still awaits the other algorithms. No Navigation stability claim.",
    "script_sha256": acceptance.digest(Path(__file__)),
    "same_recipe": True,
    "uncertainty": "Per-seed Wilson intervals describe sampled episode success only; "
    "three training seeds do not establish performance over all possible training seeds.",
    "runs": results,
}
(EVIDENCE / "racing-ppo-frozen-audit.json").write_text(json.dumps(output, indent=2) + "\n")
print([(row["seed"], row["successes"], len(row["replays"])) for row in results])
