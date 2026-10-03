"""Navigation trial and difficulty aggregation, independent of the policy executor."""

import numpy as np

from drone_playground.environments.tasks.navigation.events import (
    OUTCOME_COLLISION,
    OUTCOME_NAMES,
    OUTCOME_TIMEOUT,
)


def summarize_cell(trace: dict, labels: list[dict], dt: float, duration: float) -> dict:
    """Aggregate one (unit, difficulty) cell, counting every episode exactly once."""
    done = np.asarray(trace["done"]).astype(bool)
    outcome = np.asarray(trace["outcome"]).astype(np.int32)
    steps_total, batch = done.shape
    any_done = done.any(axis=0)
    first = np.argmax(done, axis=0)
    rows = np.arange(batch)
    steps = np.where(any_done, first + 1, steps_total)
    final_outcome = np.where(any_done, outcome[first, rows], OUTCOME_TIMEOUT)

    arrived = final_outcome == 1
    collision = final_outcome == OUTCOME_COLLISION
    out_of_bounds = final_outcome == 3
    numerical = final_outcome == 4
    timeout = ~any_done

    times = steps * dt
    constrained = float(np.where(arrived, times, duration).mean())
    success_times = times[arrived]
    clearance = np.asarray(trace["metrics"]["clearance"])
    goal_distance = np.asarray(trace["metrics"]["goal_distance"])
    returns = np.where(np.asarray(trace["active"], dtype=bool), trace["reward"], 0).sum(axis=0)

    episodes = []
    for index in range(batch):
        episodes.append(
            {
                "case": index,
                **labels[index],
                "outcome": OUTCOME_NAMES[int(final_outcome[index])],
                "arrived": bool(arrived[index]),
                "collision": bool(collision[index]),
                "out_of_bounds": bool(out_of_bounds[index]),
                "numerical_failure": bool(numerical[index]),
                "timeout": bool(timeout[index]),
                "steps": int(steps[index]),
                "arrival_time_s": float(times[index]) if arrived[index] else None,
                "min_clearance_m": float(clearance[: steps[index], index].min()),
                "final_goal_distance_m": float(goal_distance[steps[index] - 1, index]),
                "return": float(returns[index]),
            }
        )
    return {
        "num_trials": int(batch),
        "arrived": int(arrived.sum()),
        "collision": int(collision.sum()),
        "out_of_bounds": int(out_of_bounds.sum()),
        "numerical_failure": int(numerical.sum()),
        "timeout": int(timeout.sum()),
        "success_rate": float(arrived.mean()),
        "collision_rate": float(collision.mean()),
        "failure_rate": float((collision | out_of_bounds | numerical).mean()),
        "timeout_rate": float(timeout.mean()),
        "constrained_time_mean_s": constrained,
        "success_time_mean_s": float(success_times.mean()) if arrived.any() else None,
        "success_time_samples": int(arrived.sum()),
        "min_clearance_m": float(clearance.min()),
        "return_mean": float(returns.mean()),
        "episodes": episodes,
    }


def combine_cells(cells: dict[str, dict]) -> dict:
    """Macro-average over difficulties, the protocol's primary selection quantity."""
    totals = sum(cell["num_trials"] for cell in cells.values())
    macro_success = float(np.mean([cell["success_rate"] for cell in cells.values()]))
    macro_collision = float(np.mean([cell["collision_rate"] for cell in cells.values()]))
    macro_constrained = float(np.mean([cell["constrained_time_mean_s"] for cell in cells.values()]))
    valid = [cell["success_time_mean_s"] for cell in cells.values() if cell["success_time_mean_s"]]
    return {
        "num_trials": totals,
        "arrived": sum(cell["arrived"] for cell in cells.values()),
        "collision": sum(cell["collision"] for cell in cells.values()),
        "out_of_bounds": sum(cell["out_of_bounds"] for cell in cells.values()),
        "numerical_failure": sum(cell["numerical_failure"] for cell in cells.values()),
        "timeout": sum(cell["timeout"] for cell in cells.values()),
        "success_rate": macro_success,
        "collision_rate": macro_collision,
        "failure_rate": float(np.mean([cell["failure_rate"] for cell in cells.values()])),
        "timeout_rate": float(np.mean([cell["timeout_rate"] for cell in cells.values()])),
        "constrained_time_mean_s": macro_constrained,
        "success_time_mean_s": float(np.mean(valid)) if valid else None,
        "return_mean": float(np.mean([cell["return_mean"] for cell in cells.values()])),
        "cells": cells,
    }
