"""Targets, plateaus, and runtime rule edits stop complete training updates."""

import pytest

from drone_playground.learning.selection import EarlyStopping


def options(**changes):
    """Start with disabled optional stopping."""
    return {"metric": "mean_reward", "target": None, "patience": 0, "min_delta": 0.0, **changes}


def measurement(score, **row):
    """Represent one complete fixed evaluation."""
    return {"mean_reward": score, "episode_records": [{"success": True, **row}]}


def test_plateau_retains_delta_and_history_across_resume():
    """Small fluctuations and ties do not reset evaluation patience."""
    settings = options(patience=2, min_delta=0.1)
    stopping = EarlyStopping(settings, "evaluation")
    stopping.observe(measurement(1.0))
    stopping.observe(measurement(1.1))
    assert stopping.reason() is None
    restored = EarlyStopping(settings, "evaluation", stopping.state)
    restored.observe(measurement(1.01))
    assert restored.reason()["reason"] == "no_improvement"
    assert restored.reason()["bad_evaluations"] == 2
    restored.observe(measurement(1.2))
    assert restored.reason() is None


def test_live_target_uses_existing_comparable_history():
    """A target change is effective on the next safe boundary."""
    stopping = EarlyStopping(options(), "evaluation")
    stopping.observe(measurement(2.0))
    assert stopping.reason() is None
    stopping.configure({"target": 2.0})
    assert stopping.reason()["reason"] == "target_reached"


def test_quality_targets_require_successful_finite_episodes():
    """A low error on a failed flight cannot satisfy Tracking's target."""
    stopping = EarlyStopping(options(metric="position_rmse", target=0.25), "evaluation")
    stopping.observe(measurement(0.0, position_rmse=0.1, success=False))
    assert stopping.reason() is None
    stopping.observe(measurement(0.0, position_rmse=0.2))
    assert stopping.reason()["reason"] == "target_reached"
    rate = EarlyStopping(options(metric="success_rate", target=1.0), "evaluation")
    rate.observe({"mean_reward": 0.0, "episode_records": [{"success": True}, {"success": False}]})
    assert rate.reason() is None
    rate.observe(measurement(0.0))
    assert rate.reason()["value"] == 1.0


@pytest.mark.parametrize(
    ("changes", "source"),
    [
        ({"patience": True}, "evaluation"),
        ({"target": float("nan")}, "evaluation"),
        ({"min_delta": -1.0}, "evaluation"),
        ({"patience": 2}, None),
        ({"metric": "success_rate", "target": 1.1}, "evaluation"),
        ({"metric": "success_rate", "target": 1.0}, "training"),
        ({"metric": "position_rmse", "target": -0.1}, "evaluation"),
    ],
)
def test_invalid_rules_do_not_silently_stop_training(changes, source):
    """Reject malformed or incomparable controls before applying them."""
    with pytest.raises(ValueError):
        EarlyStopping(options(**changes), source)
