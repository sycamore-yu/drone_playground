from drone_playground.learning.train import development_score, pilot_scene_rates


def test_pilot_counts_failures_and_high_error_as_unsuccessful():
    report = {
        "episodes": [
            dict(completed=True, failed=False, rmse_m=0.1),
            dict(completed=True, failed=False, rmse_m=0.3),
            dict(completed=False, failed=True, rmse_m=0.01),
        ]
    }
    assert development_score("figure8", report, "release-pilot-v1") == (1 / 3,)


def test_pilot_uses_worst_scene_not_average_success():
    report = {
        "cells": {
            "easy": {"episodes": [dict(subtype="forest", arrived=True)] * 9},
            "hard": {"episodes": [dict(subtype="wall", arrived=False)]},
        }
    }
    assert pilot_scene_rates("navigation", report) == {"easy/forest": 1, "hard/wall": 0}
    assert development_score("navigation", report, "release-pilot-v1") == (0,)
