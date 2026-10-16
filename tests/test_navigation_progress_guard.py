"""Opt-in PPO progression evidence: accepted navigation outcomes remain primary."""

from copy import deepcopy

import pytest

from drone_playground.learning.progress_guard import (
    ProgressGuardConfig,
    evaluate_progress_guard,
    screen_evidence,
)

SCENES = ("S01", "S02", "S03", "D01", "D02", "D03")


def report(progress, *, successes=0, collision=8, oob=0, lower_oob=0):
    """Build a completed fixed-cohort navigation screen with all eight episodes per scene."""
    assert 0 <= successes <= 8
    assert successes + collision + oob == 8
    return {
        "complete": True,
        "protocol": "diagnostic",
        "reports": {
            scene: {
                "episodes": 8,
                "successes": successes,
                "outcomes": {
                    "SUCCESS": successes,
                    "COLLISION": collision,
                    "OUT_OF_BOUNDS": oob,
                },
                "progress_diagnostics": {
                    "mean_normalized_final_progress": progress,
                    "mean_normalized_best_progress": progress,
                    "mean_final_distance_metres": (1 - progress) * 96,
                    "mean_path_length_metres": progress * 120,
                    "minimum_height_metres": 0.5 if lower_oob else 2.0,
                    "minimum_lower_boundary_margin_metres": 0.0 if lower_oob else 1.5,
                    "lower_height_out_of_bounds": lower_oob,
                },
            }
            for scene in SCENES
        },
    }


def test_safe_progress_prevents_zero_success_patience():
    """A genuinely improving zero-success policy gets budget despite 0/48."""
    settings = ProgressGuardConfig(min_progress_delta=0.03, max_unsafe_increase=0.0)
    state = None
    decisions = []
    for update, progress in ((120, 0.1), (240, 0.2), (360, 0.35)):
        state, decision = evaluate_progress_guard(
            report(progress),
            SCENES,
            update=update,
            previous=state,
            best_screen_score=0.0,
            settings=settings,
            min_updates=120,
            patience=2,
        )
        decisions.append(decision)
    assert all(not d["patience_exhausted"] for d in decisions)
    assert decisions[1]["progress_improved"]
    assert decisions[2]["progress_improved"]
    assert state["best_successes"] == 0
    assert state["best_progress"] == pytest.approx(0.35)
    assert state["best_progress_checkpoint_update"] == 360
    assert state["stagnant_screens"] == 0


def test_no_progress_stops_only_after_declared_minimum():
    """Progress-aware stopping remains finite rather than continuing indefinitely."""
    settings = ProgressGuardConfig(min_progress_delta=0.02, max_unsafe_increase=0.0)
    state = None
    decisions = []
    for update in (120, 240, 360, 480):
        state, decision = evaluate_progress_guard(
            report(0.05),
            SCENES,
            update=update,
            previous=state,
            best_screen_score=0.0,
            settings=settings,
            min_updates=300,
            patience=2,
        )
        decisions.append(decision)
    assert not any(d["patience_exhausted"] for d in decisions[:3])
    assert decisions[-1]["patience_exhausted"]


def test_unsafe_progress_must_not_reset_patience():
    """Closer-to-goal but increased below-ground failures are not safety progress."""
    settings = ProgressGuardConfig(min_progress_delta=0.02, max_unsafe_increase=0.0)
    state, _ = evaluate_progress_guard(
        report(0.2),
        SCENES,
        update=100,
        previous=None,
        best_screen_score=0.0,
        settings=settings,
        min_updates=100,
        patience=1,
    )
    unsafe = report(0.5, successes=0, collision=4, oob=4, lower_oob=4)
    state, result = evaluate_progress_guard(
        unsafe,
        SCENES,
        update=200,
        previous=state,
        best_screen_score=0.0,
        settings=settings,
        min_updates=100,
        patience=1,
    )
    assert result["patience_exhausted"]
    assert not result["progress_improved"]
    assert state["best_progress"] == pytest.approx(0.2)


def test_policy_regression_keeps_earlier_best_and_uses_two_screens():
    """A loss of an achieved scenario is an event, not checkpoint replacement."""
    settings = ProgressGuardConfig(min_progress_delta=0.02, max_unsafe_increase=0.0)
    state, _ = evaluate_progress_guard(
        report(0.8, successes=4, collision=4),
        SCENES,
        update=2880,
        previous=None,
        best_screen_score=0.125,
        settings=settings,
        min_updates=120,
        patience=3,
    )
    assert state["best_successes"] == 24
    assert state["best_checkpoint_update"] == 2880
    state, result = evaluate_progress_guard(
        report(0.82, collision=0, oob=8, lower_oob=8),
        SCENES,
        update=3120,
        previous=state,
        best_screen_score=0.0,
        settings=settings,
        min_updates=120,
        patience=3,
    )
    assert result["regression_streak"] == 1
    assert not result["policy_regression"]
    state, result = evaluate_progress_guard(
        report(0.83, collision=0, oob=8, lower_oob=8),
        SCENES,
        update=3360,
        previous=state,
        best_screen_score=0.0,
        settings=settings,
        min_updates=120,
        patience=3,
    )
    assert result["policy_regression"]
    assert state["best_checkpoint_update"] == 2880
    assert state["best_successes"] == 24
    assert state["best_progress"] == pytest.approx(0.8)
    assert state["best_progress_checkpoint_update"] == 2880


def test_screen_require_complete_consistent_per_scene_48_denominator():
    """Partial probes, 25-episode validation and invalid metrics cannot steer patience."""
    for invalid in (
        {**report(0.3), "complete": False},
        {**report(0.3), "protocol": "learning"},
        {**report(0.3), "reports": {"S01": report(0.3)["reports"]["S01"]}},
    ):
        with pytest.raises(ValueError):
            screen_evidence(invalid, SCENES)
    wrong = report(0.3)
    wrong["reports"]["D03"]["episodes"] = 25
    with pytest.raises(ValueError):
        screen_evidence(wrong, SCENES)
    missing = report(0.3)
    del missing["reports"]["S01"]["progress_diagnostics"]
    with pytest.raises(ValueError):
        screen_evidence(missing, SCENES)
    too_high = report(0.3)
    too_high["reports"]["S01"]["progress_diagnostics"]["mean_normalized_final_progress"] = float(
        "nan"
    )
    with pytest.raises(ValueError):
        screen_evidence(too_high, SCENES)
    with pytest.raises(ValueError):
        ProgressGuardConfig(min_progress_delta=0.0, max_unsafe_increase=0.0)
    with pytest.raises(ValueError):
        ProgressGuardConfig(min_progress_delta=0.1, max_unsafe_increase=-0.1)
    assert screen_evidence(report(0.2), SCENES)["episodes"] == 48


def test_persisted_guard_survives_restart_without_mutating_input():
    """Loaded screening states produce the same result, retaining prior best identities."""
    settings = ProgressGuardConfig(min_progress_delta=0.03, max_unsafe_increase=0.0)
    initial, _ = evaluate_progress_guard(
        report(0.2, successes=3, collision=5),
        SCENES,
        update=400,
        previous=None,
        best_screen_score=0.12,
        settings=settings,
        min_updates=100,
        patience=3,
    )
    before = deepcopy(initial)
    new_state, result = evaluate_progress_guard(
        report(0.15),
        SCENES,
        update=500,
        previous=initial,
        best_screen_score=0.0,
        settings=settings,
        min_updates=100,
        patience=3,
    )
    assert initial == before
    assert result["regression_streak"] == 1
    assert new_state["best_checkpoint_update"] == 400
