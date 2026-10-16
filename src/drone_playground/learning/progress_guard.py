"""Opt-in, evidence-based Navigation PPO patience and policy-regression detection.

This training-only module never consumes independent frozen benchmark results.
Existing screen-only training runs retain their original stopping contract.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressGuardConfig:
    """New runs explicitly select a progress resolution and safety tolerance."""

    min_progress_delta: float
    max_unsafe_increase: float
    regression_patience: int = 2
    min_score_delta: float = 0.005

    def __post_init__(self):
        """Disallow silent thresholds or a meaningless patience contract."""
        if not math.isfinite(self.min_progress_delta) or not 0 < self.min_progress_delta < 1:
            raise ValueError("min_progress_delta must be finite in (0,1)")
        if not math.isfinite(self.max_unsafe_increase) or not 0 <= self.max_unsafe_increase <= 1:
            raise ValueError("max_unsafe_increase must be finite in [0,1]")
        if not math.isfinite(self.min_score_delta) or not 0 < self.min_score_delta < 1:
            raise ValueError("min_score_delta must be finite in (0,1)")
        if type(self.regression_patience) is not int or self.regression_patience < 2:
            raise ValueError("regression_patience must be at least two complete screens")


def screen_evidence(report: dict, scenes: tuple[str, ...] | list[str]) -> dict:
    """Aggregate only a *complete* matched 6x8 training diagnostic screen."""
    if (
        report.get("protocol") != "diagnostic"
        or report.get("complete") is not True
        or len(scenes) != 6
        or len(set(scenes)) != 6
        or set(report.get("reports", {})) != set(scenes)
    ):
        raise ValueError("Progress guard requires six complete diagnostic scenes")
    progress, best_progress, distances, path_lengths, lower_margins = [], [], [], [], []
    successes = unsafe = lower_exits = 0
    for scene in scenes:
        row = report["reports"][scene]
        n = row.get("episodes")
        s = row.get("successes")
        events = row.get("outcomes")
        telemetry = row.get("progress_diagnostics")
        if (
            n != 8
            or type(s) is not int
            or not 0 <= s <= 8
            or not isinstance(events, dict)
            or sum(events.values()) != 8
            or events.get("SUCCESS", 0) != s
            or not isinstance(telemetry, dict)
        ):
            raise ValueError(f"Incomplete/contradictory screen evidence for {scene}")
        required = (
            "mean_normalized_final_progress",
            "mean_normalized_best_progress",
            "mean_final_distance_metres",
            "mean_path_length_metres",
            "minimum_height_metres",
            "minimum_lower_boundary_margin_metres",
            "lower_height_out_of_bounds",
        )
        if any(key not in telemetry for key in required):
            raise ValueError(f"Missing progress evidence for {scene}")
        values = [telemetry[key] for key in required[:-1]]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError(f"Nonfinite progress evidence for {scene}")
        exits = telemetry["lower_height_out_of_bounds"]
        if (
            type(exits) is not int
            or not 0 <= exits <= events.get("OUT_OF_BOUNDS", 0)
            or not 0 <= telemetry["mean_normalized_best_progress"] <= 1
            or telemetry["mean_normalized_final_progress"] > 1
            or telemetry["mean_final_distance_metres"] < 0
            or telemetry["mean_path_length_metres"] < 0
        ):
            raise ValueError(f"Invalid progress/terminal evidence for {scene}")
        successes += s
        unsafe += sum(
            events.get(event, 0)
            for event in ("COLLISION", "OUT_OF_BOUNDS", "NUMERICAL_FAILURE", "METHOD_FAILURE")
        )
        lower_exits += exits
        progress.append(telemetry["mean_normalized_final_progress"])
        best_progress.append(telemetry["mean_normalized_best_progress"])
        distances.append(telemetry["mean_final_distance_metres"])
        path_lengths.append(telemetry["mean_path_length_metres"])
        lower_margins.append(telemetry["minimum_lower_boundary_margin_metres"])
    return {
        "episodes": 48,
        "successes": successes,
        "mean_final_progress": sum(progress) / 6,
        "worst_scene_final_progress": min(progress),
        "mean_best_progress": sum(best_progress) / 6,
        "mean_final_distance_metres": sum(distances) / 6,
        "mean_path_length_metres": sum(path_lengths) / 6,
        "minimum_lower_boundary_margin_metres": min(lower_margins),
        "unsafe_fraction": unsafe / 48,
        "lower_height_out_of_bounds": lower_exits,
    }


def evaluate_progress_guard(
    report: dict,
    scenes: tuple[str, ...] | list[str],
    *,
    update: int,
    previous: dict | None,
    best_screen_score: float,
    settings: ProgressGuardConfig,
    min_updates: int,
    patience: int,
) -> tuple[dict, dict]:
    """Update only training-screen decisions, never model weights or formal acceptance."""
    current = screen_evidence(report, scenes)
    if (
        type(update) is not int
        or update < 0
        or type(min_updates) is not int
        or min_updates < 1
        or type(patience) is not int
        or patience < 1
        or not math.isfinite(best_screen_score)
    ):
        raise ValueError("Invalid progress guard update, budget, patience or success score")
    previous = dict(previous) if previous is not None else {}
    old_score = previous.get("best_score")
    improved_score = old_score is None or best_screen_score >= old_score + settings.min_score_delta
    best_successes = previous.get("best_successes", 0)
    regression_streak = (
        previous.get("regression_streak", 0) + 1
        if best_successes > 0 and current["successes"] < best_successes
        else 0
    )
    old_progress = previous.get("best_progress")
    safer_or_equal = old_progress is None or (
        current["unsafe_fraction"]
        <= previous["best_progress_unsafe_fraction"] + settings.max_unsafe_increase
        and current["lower_height_out_of_bounds"]
        <= previous["best_progress_lower_height_exits"]
        + int(settings.max_unsafe_increase * current["episodes"])
    )
    improving_progress = (
        old_progress is None
        or current["mean_final_progress"] >= old_progress + settings.min_progress_delta
    ) and safer_or_equal
    # A 0/48 progress improvement can protect a learning policy from a
    # success-only patience prune. It cannot conceal a post-success regression.
    rescued_by_progress = (
        not improved_score
        and best_successes == 0
        and current["successes"] == 0
        and improving_progress
    )
    stagnant = (
        0
        if improved_score or rescued_by_progress or update < min_updates
        else previous.get("stagnant_screens", 0) + 1
    )
    if improved_score or old_score is None:
        best_score = best_screen_score
        best_checkpoint_update = update
        best_successes = current["successes"]
    else:
        best_score = old_score
        best_checkpoint_update = previous["best_checkpoint_update"]
    if improving_progress and (old_progress is None or rescued_by_progress or improved_score):
        best_progress = current["mean_final_progress"]
        best_progress_checkpoint_update = update
        best_progress_unsafe = current["unsafe_fraction"]
        best_progress_lower = current["lower_height_out_of_bounds"]
    else:
        best_progress = old_progress
        best_progress_checkpoint_update = previous["best_progress_checkpoint_update"]
        best_progress_unsafe = previous["best_progress_unsafe_fraction"]
        best_progress_lower = previous["best_progress_lower_height_exits"]
    result = {
        "best_score": best_score,
        "best_successes": best_successes,
        "best_checkpoint_update": best_checkpoint_update,
        "best_progress": best_progress,
        "best_progress_checkpoint_update": best_progress_checkpoint_update,
        "best_progress_unsafe_fraction": best_progress_unsafe,
        "best_progress_lower_height_exits": best_progress_lower,
        "stagnant_screens": stagnant,
        "regression_streak": regression_streak,
    }
    decision = {
        **current,
        "score_improved": improved_score,
        "progress_improved": rescued_by_progress,
        "stagnant_screens": stagnant,
        "regression_streak": regression_streak,
        "policy_regression": regression_streak >= settings.regression_patience,
        "patience_exhausted": update >= min_updates and stagnant >= patience,
    }
    return result, decision
