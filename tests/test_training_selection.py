"""Test bounded light-screen stopping, without tuning to frozen test data."""

import pytest

from drone_playground.learning.selection import (
    PatienceConfig,
    screening_score,
    update_patience,
)


def report(scenes, successes, *, rmse=None):
    """Create a complete eight-episode training-only diagnosis."""
    return {
        "complete": True,
        "reports": {
            scene: {"episodes": 8, "successes": number, "successful_position_rmse": rmse}
            for scene, number in zip(scenes, successes, strict=True)
        },
    }


def test_navigation_score_prioritizes_unsolved_scenes():
    """One zero-success scene blocks progress even when others are solved."""
    scenes = ["S01", "S02", "S03", "D01", "D02", "D03"]
    initial = screening_score(report(scenes, [8, 8, 0, 8, 8, 8]), "navigation", scenes)
    improved = screening_score(report(scenes, [8, 8, 1, 8, 8, 8]), "navigation", scenes)
    assert initial < 0.25
    assert improved - initial > 0.12


def test_patience_stops_after_six_unchanged_post_800_screens():
    """Six screens strictly after minimum updates end the losing recipe."""
    settings = PatienceConfig(min_updates=800, patience=6, min_delta=0.02)
    best, stagnant = None, 0
    for update in (160, 320, 480, 640, 800, 960, 1120, 1280, 1440, 1600):
        best, stagnant, improved, stop = update_patience(
            0.125, update, best, stagnant, settings=settings
        )
        assert stop is (update == 1600)
    assert best == 0.125
    assert stagnant == 6
    assert not improved


def test_meaningful_progress_resets_patience():
    """A late improvement prevents an early stop; plateau resumes its count."""
    settings = PatienceConfig()
    best, stagnant = 0.125, 5
    best, stagnant, improved, stop = update_patience(0.251, 1520, best, stagnant, settings=settings)
    assert (best, stagnant, improved, stop) == (0.251, 0, True, False)
    best, stagnant, improved, stop = update_patience(0.261, 1600, best, stagnant, settings=settings)
    assert (best, stagnant, improved, stop) == (0.251, 1, False, False)


@pytest.mark.parametrize(
    "min_updates,patience,min_delta", [(0, 6, 0.02), (800, 0, 0.02), (800, 6, 0), (800, 6, 2.0)]
)
def test_invalid_patience_configuration(min_updates, patience, min_delta):
    """Invalid bounds must not silently alter the training contract."""
    with pytest.raises(ValueError):
        PatienceConfig(min_updates=min_updates, patience=patience, min_delta=min_delta)


def test_incomplete_screen_cannot_affect_stopping():
    """Incomplete/incorrectly-sized records never become progress evidence."""
    with pytest.raises(ValueError):
        screening_score(report(["S01"], [5]), "navigation", ["S01", "S02"])
    with pytest.raises(ValueError):
        screening_score(
            {"complete": True, "reports": {"S01": {"episodes": 25, "successes": 23}}},
            "navigation",
            ["S01"],
        )
