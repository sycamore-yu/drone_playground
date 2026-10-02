"""Selection must preserve the worst scene and include all perturbations."""

import copy

import pytest

NAVIGATION_PROTOCOL = {
    "scenes": {
        "static": ["S01", "S02", "S03", "S06"],
        "dynamic": ["D01", "D02", "D03", "D06"],
    },
    "primary_scenes": ["S01", "S02", "S03", "D01", "D02", "D03"],
    "episodes_per_scene": {"checkpoint_eval": 8},
    "success_threshold": {
        "navigation-checkpoint_eval-v2": {"scope": "all"},
        "navigation-checkpoint_eval-primary-v1": {
            "scope": "primary",
            "per_scene": 0.9,
        },
    },
}


def report(counts):
    scenes = ["S01", "S02", "S03", "S06", "D01", "D02", "D03", "D06"]
    rows = [dict(scene_id=scene, seed=50000 + 8 * repeat + i,
                 arrived=repeat < count)
            for repeat in range(8) for i, (scene, count) in enumerate(zip(scenes, counts, strict=True))]
    return dict(episodes=rows, num_trials=64,
                initial_conditions=dict(seeds=[row["seed"] for row in rows]))


def selection(value, criterion="navigation-checkpoint_eval-v2"):
    from drone_playground.benchmarks import navigation_checkpoint_eval_selection

    return navigation_checkpoint_eval_selection(
        value,
        criterion,
        protocol=NAVIGATION_PROTOCOL,
    )


def test_worst_scene_precedes_total_and_total_breaks_zero_worst_ties():
    low_worst = selection(report([0, 8, 8, 8, 8, 8, 8, 8]))
    higher_worst = selection(report([1] * 8))
    assert higher_worst["score"] > low_worst["score"]
    assert higher_worst["pilot_objective"] > low_worst["pilot_objective"]
    empty = selection(report([0] * 8))
    assert low_worst["score"] > empty["score"]
    assert low_worst["worst_scene_success_rate"] == 0
    assert low_worst["overall_success_rate"] == 56 / 64
    assert selection(report([8] * 8))["pilot_objective"] == 1


@pytest.mark.parametrize("problem", ["missing", "duplicate_seed", "missing_reset", "wrong_reset"])
def test_selection_rejects_incomplete_or_repeated_perturbations(problem):
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
        selection(value)


def test_primary_selection_does_not_use_extensions_to_choose_policy():
    first = report([8, 8, 8, 0, 8, 8, 8, 0])
    full = report([8] * 8)
    criterion = "navigation-checkpoint_eval-primary-v1"
    a = selection(first, criterion)
    b = selection(full, criterion)
    assert a['score'] == b['score'] == [8, 48]
    assert a['pilot_objective'] == b['pilot_objective'] == 1.
    assert a['primary_checkpoint_eval_passed'] and a['overall_success_rate'] == .75
    weak = selection(report([8, 8, 8, 8, 8, 7, 8, 8]), criterion)
    assert not weak['primary_checkpoint_eval_passed']
    assert weak['score'] < a['score']


def test_primary_selection_retains_worst_scene_priority():
    metric = 'navigation-checkpoint_eval-primary-v1'
    a = selection(report([0, 8, 8, 8, 8, 8, 8, 8]), metric)
    b = selection(report([1] * 8), metric)
    assert b['score'] > a['score'] and b['pilot_objective'] > a['pilot_objective']


def test_selection_consumes_explicit_scene_roles_and_configured_episode_count():
    from drone_playground.benchmarks import navigation_checkpoint_eval_selection

    scenes = ["forest", "warehouse", "forest06"]
    rows = [dict(scene_id=scene, seed=100+i, arrived=scene != "warehouse")
            for i, scene in enumerate(scenes * 3)]
    value = dict(episodes=rows, num_trials=len(rows),
                 initial_conditions=dict(seeds=[row["seed"] for row in rows]))
    specification = dict(scenes={"navigation": scenes}, primary_scenes=["forest", "forest06"],
                         episodes_per_scene={"checkpoint_eval": 3},
                         success_threshold={"navigation-checkpoint_eval-primary-v1":
                                            dict(scope="primary", per_scene=0.9)})
    result = navigation_checkpoint_eval_selection(value, "navigation-checkpoint_eval-primary-v1",
                                                  protocol=specification)
    assert result["score"] == [3, 6]
    assert result["pilot_objective"] == 1
    assert result["overall_success_rate"] == 2/3
