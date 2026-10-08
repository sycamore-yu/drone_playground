import copy

import pytest

from drone_playground.benchmarks import validate_control_report

TRACKING_SPEC = {
    "name": "tracking",
    "version": 1,
    "episodes": 100,
    "success": {
        "minimum_completion_rate": 0.95,
        "maximum_completed_rmse_m": 0.25,
    },
}
RACING_SPEC = {
    "name": "racing",
    "version": 1,
    "episodes": 100,
    "success": {"minimum_completion_rate": 0.90},
}


def validate(value, task):
    specification = TRACKING_SPEC if task == "tracking" else RACING_SPEC
    return validate_control_report(value, task, specification=specification)


def report(completed=95, error=0.1):
    return dict(
        role="eval",
        parameters_frozen=True,
        parameter_sha256="test",
        num_trials=100,
        episodes=[
            dict(seed=30000 + i, completed=i < completed, failed=i >= completed, rmse_m=error)
            for i in range(100)
        ],
    )


def test_tracking_uses_95_percent_and_keeps_failures():
    accepted = validate(report(), "tracking")
    assert accepted["passed"] and accepted["protocol"] == "benchmark-tracking-v1"
    result = validate(report(94), "tracking")
    assert not result["passed"] and result["failed"] == 6
    assert validate(report(94), "racing")["passed"]
    assert not validate(report(100, 0.26), "tracking")["passed"]


@pytest.mark.parametrize("change", ["duplicates", "missing", "dev", "unfrozen"])
def test_rejects_invalid_benchmark_evidence(change):
    value = copy.deepcopy(report())
    if change == "duplicates":
        value["episodes"][1]["seed"] = value["episodes"][0]["seed"]
    elif change == "missing":
        value["episodes"].pop()
    elif change == "dev":
        value["role"] = "dev"
    else:
        value["parameters_frozen"] = False
    with pytest.raises(ValueError):
        validate(value, "tracking")


def test_native_solver_needs_runtime_identity_and_does_not_require_training_seeds():
    value = report()
    value["parameter_identity_kind"] = "resolved optimization configuration"
    with pytest.raises(ValueError, match=r"runtime identity"):
        validate(value, "tracking")
    value["runtime_identity"] = {"libacados.so": "a" * 64}
    result = validate(value, "tracking")
    assert result["passed"] and result["caveat"] == "Frozen solver; no learning seeds required"
