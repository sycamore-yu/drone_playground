"""Correct the one-time collector scope error using already checked episode evidence."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("collector", ROOT / "tools/report_acceptance.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
source = DIRECTORY / "acceptance-single-full.json"
snapshot = json.loads(source.read_text())
corrected = []
for run in snapshot["runs"]:
    if run["C5"]["passed"] is not False:
        continue
    assert run["task"] == "navigation"
    assert run["selection"]["rule"] == collector.SINGLE_RULE
    for evaluation in run["checkpoint_evaluations"]:
        report = json.loads((Path(evaluation["directory"]) / "report.json").read_text())
        assert set(report["reports"]) == set(collector.NAV_SCENES[:6])
        assert evaluation["partition"]["reasons"] == [
            "S06: no verified CSV initialization seeds",
            "D06: no verified CSV initialization seeds",
        ]
        for scene in ("S06", "D06"):
            assert evaluation["scenes"][scene]["episodes"] == 0
            del evaluation["scenes"][scene]
        assert all(v["quality"]["passed"] is True for v in evaluation["scenes"].values())
        evaluation["partition"] = collector.check(
            True,
            "Reused checked primary-scene identities; "
            "optional extensions removed from validation scope",
        )
    issues = []
    events = collector.read_events(Path(run["path"]) / "metrics.jsonl", issues)
    assert not issues, issues
    collector.check_single_full(run, run["C5"]["screening"], events)
    assert run["C5"]["passed"] is True, run["path"]
    corrected.append(run["path"])
assert len(corrected) == 5
result = {key: snapshot[key] for key in ("schema_version", "manifest", "resolved_root")}
result["runs"] = [run for run in snapshot["runs"] if not run.get("reused_evidence")]
result.update(
    generated_utc=datetime.now(UTC).isoformat(),
    correction={
        "source": str(source),
        "runs": corrected,
        "reason": "Full candidate validation requires six primary Navigation scenes; "
        "extensions remain mandatory in frozen reports.",
        "reused": "All previously checked CSV rows, configuration checksums and model identity "
        "evidence. No model loading, hashing or episode replay.",
    },
)
(DIRECTORY / "acceptance-primary-validation.json").write_text(json.dumps(result, indent=2) + "\n")
print({"corrected_selection_checks": len(corrected)})
