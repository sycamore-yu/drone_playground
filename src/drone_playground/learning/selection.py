"""Training-only progress measurement for bounded early stopping.

Checkpoint selection uses the independent full validation protocol; the
frozen benchmark never enters this module or influences checkpoint ranking.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PatienceConfig:
    """Count insignificant screening evaluations after minimum training updates."""

    min_updates: int = 800
    patience: int = 6
    min_delta: float = 0.02

    def __post_init__(self):
        """Refuse impossible or underspecified stopping settings."""
        if type(self.min_updates) is not int or self.min_updates < 1:
            raise ValueError("min_updates must be a positive integer")
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError("patience must be a positive integer")
        if (
            not isinstance(self.min_delta, (float, int))
            or not math.isfinite(self.min_delta)
            or not 0 < self.min_delta < 1
        ):
            raise ValueError("min_delta must be finite and between zero and one")


def screening_score(report: dict, task: str, scenes: list[str]) -> float:
    """Score complete *light* screens; emphasize the weakest navigation scene.

    The per-scene denominator is fixed at 8. Do not compare this quantity to
    C2/C3/C4 thresholds or any frozen benchmark result.
    """
    if not report.get("complete") or not scenes:
        raise ValueError("Patience requires a complete screening report")
    rates = []
    for scene in scenes:
        row = report.get("reports", {}).get(scene)
        if not isinstance(row, dict) or row.get("episodes") != 8:
            raise ValueError(f"Missing eight-episode screening for {scene}")
        successes = row.get("successes")
        if type(successes) is not int or not 0 <= successes <= 8:
            raise ValueError(f"Invalid screening successes for {scene}")
        rates.append(successes / 8)
    if task == "navigation":
        return min(rates) + 0.25 * sum(rates) / len(rates)
    score = rates[0]
    if task == "tracking":
        rmse = report["reports"][scenes[0]].get("successful_position_rmse")
        if rmse is not None and math.isfinite(rmse) and rmse >= 0:
            score += 0.25 * (1 - min(1.0, rmse / 0.35))
    return score


def update_patience(
    score: float,
    update: int,
    best: float | None,
    stagnant: int,
    *,
    settings: PatienceConfig,
) -> tuple[float, int, bool, bool]:
    """Return updated best, stagnant screens, meaningful change and stop flag."""
    if not math.isfinite(score):
        raise ValueError("Nonfinite screening score")
    improved = best is None or score >= best + settings.min_delta
    best = score if improved else best
    # Patience only starts counting after the declared minimum training budget.
    stagnant = 0 if improved or update < settings.min_updates else stagnant + 1
    return (
        best,
        stagnant,
        improved,
        (update >= settings.min_updates and stagnant >= settings.patience),
    )
