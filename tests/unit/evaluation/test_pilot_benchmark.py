import pytest

from drone_playground.benchmarks import (
    checkpoint_eval_score,
    checkpoint_eval_seeds,
    pilot_scene_rates,
)


def test_pilot_counts_failures_and_high_error_as_unsuccessful():
    report = {
        "episodes": [
            dict(completed=True, failed=False, rmse_m=0.1),
            dict(completed=True, failed=False, rmse_m=0.3),
            dict(completed=False, failed=True, rmse_m=0.01),
        ]
    }
    assert checkpoint_eval_score("figure8", report, "release-pilot-v1") == (1 / 3,)


def test_pilot_uses_worst_scene_not_average_success():
    report = {
        "cells": {
            "easy": {"episodes": [dict(subtype="forest", arrived=True)] * 9},
            "hard": {"episodes": [dict(subtype="wall", arrived=False)]},
        }
    }
    assert pilot_scene_rates("navigation", report) == {"easy/forest": 1, "hard/wall": 0}
    assert checkpoint_eval_score("navigation", report, "release-pilot-v1") == (0,)


def test_checkpoint_eval_seed_start_is_runtime_configuration():
    assert checkpoint_eval_seeds({"checkpoint_eval_seed_start": 50000}, 3) == [50000, 50001, 50002]
    with pytest.raises(ValueError, match=r"seed start"):
        checkpoint_eval_seeds({"checkpoint_eval_seed_start": -1}, 1)
