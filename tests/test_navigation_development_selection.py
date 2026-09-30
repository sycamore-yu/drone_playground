"""Selection must preserve the worst scene and include all perturbations."""

import copy

import pytest


def report(counts):
    scenes = ["S01", "S02", "S03", "S06", "D01", "D02", "D03", "D06"]
    rows = [dict(scene_id=scene, seed=50000 + 8 * repeat + i,
                 arrived=repeat < count)
            for repeat in range(8) for i, (scene, count) in enumerate(zip(scenes, counts, strict=True))]
    return dict(episodes=rows, num_trials=64,
                initial_conditions=dict(seeds=[row["seed"] for row in rows]))


def test_worst_scene_precedes_total_and_total_breaks_zero_worst_ties():
    from drone_playground.evaluation.pointcloud_navigation import navigation_development_selection

    low_worst = navigation_development_selection(report([0, 8, 8, 8, 8, 8, 8, 8]))
    higher_worst = navigation_development_selection(report([1] * 8))
    assert higher_worst["score"] > low_worst["score"]
    assert higher_worst["pilot_objective"] > low_worst["pilot_objective"]
    empty = navigation_development_selection(report([0] * 8))
    assert low_worst["score"] > empty["score"]
    assert low_worst["worst_scene_success_rate"] == 0
    assert low_worst["overall_success_rate"] == 56 / 64
    assert navigation_development_selection(report([8] * 8))["pilot_objective"] == 1


@pytest.mark.parametrize("problem", ["missing", "duplicate_seed", "missing_reset", "wrong_reset"])
def test_selection_rejects_incomplete_or_repeated_perturbations(problem):
    from drone_playground.evaluation.pointcloud_navigation import navigation_development_selection

    value = copy.deepcopy(report([8] * 8))
    if problem == "missing":
        value["episodes"].pop()
    elif problem == "duplicate_seed":
        value["episodes"][0]["seed"] = value["episodes"][1]["seed"]
    elif problem == "missing_reset":
        value["initial_conditions"] = None
    else:
        value["initial_conditions"]["seeds"][0] += 100
    with pytest.raises(ValueError):
        navigation_development_selection(value)
