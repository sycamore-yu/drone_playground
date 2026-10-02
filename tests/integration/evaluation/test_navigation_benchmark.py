"""Primary-scene qualification keeps extension failures and legacy judgments."""

import copy

import pytest

from drone_playground.benchmarks import validate_navigation_report


def report(repeats=25):
    scenes = ['S01', 'S02', 'S03', 'S06', 'D01', 'D02', 'D03', 'D06']
    rows = [dict(scene_id=scene, seed=80000 + 8 * repeat + i,
                 arrived=True, outcome='arrived', initial_position_m=[repeat * .01, i * .01, 1.])
            for repeat in range(repeats) for i, scene in enumerate(scenes)]
    return dict(role='eval', parameters_frozen=True, parameter_sha256='test',
                num_trials=len(rows), episodes=rows,
                initial_conditions=dict(seeds=[r['seed'] for r in rows],
                                        position_m=[r['initial_position_m'] for r in rows]))


def fail(value, scene, count):
    for row in [r for r in value['episodes'] if r['scene_id'] == scene][:count]:
        row.update(arrived=False, outcome='collision')


def test_extension_failures_are_reported_and_legacy_judgment_is_unchanged():
    value = report()
    fail(value, 'S06', 25)
    fail(value, 'D06', 25)
    original = copy.deepcopy(value)
    assert not validate_navigation_report(value)['passed']
    result = validate_navigation_report(value, criterion='navigation-primary-v1')
    assert result['passed'] and result['protocol'] == 'benchmark-navigation-primary-v1'
    for task, prefix in [('static', 'S'), ('dynamic', 'D')]:
        cell = result['tasks'][task]
        assert cell['num_trials'] == 100 and cell['success_rate'] == .75
        assert cell['primary_num_trials'] == 75 and cell['extension_num_trials'] == 25
        assert cell['primary_scene_success_rates'] == dict.fromkeys([prefix + x for x in ['01', '02', '03']], 1.)
        assert cell['extension_scene_success_rates'] == {prefix + '06': 0.}
    assert value == original


def test_one_weak_primary_scene_fails_even_when_task_aggregate_passes():
    value = report()
    fail(value, 'D02', 3)
    assert validate_navigation_report(value)['passed']  # 97/100 aggregate.
    result = validate_navigation_report(value, criterion='navigation-primary-v1')
    assert result['tasks']['static']['passed']
    assert not result['tasks']['dynamic']['passed']
    assert result['tasks']['dynamic']['primary_scene_success_rates']['D02'] == .88


def test_primary_threshold_and_single_native_task():
    value = report(repeats=50)
    fail(value, 'S03', 5)
    value['episodes'] = [r for r in value['episodes'] if r['scene_id'].startswith('S')]
    value['num_trials'] = len(value['episodes'])
    value['initial_conditions'] = dict(seeds=[r['seed'] for r in value['episodes']],
                                     position_m=[r['initial_position_m'] for r in value['episodes']])
    assert validate_navigation_report(value, tasks=('static',), criterion='navigation-primary-v1')['passed']
    fail(value, 'S03', 6)
    assert not validate_navigation_report(value, tasks=('static',), criterion='navigation-primary-v1')['passed']


@pytest.mark.parametrize('problem', ['missing_extension', 'duplicate_seed', 'wrong_reset'])
def test_revised_criterion_preserves_evidence_checks(problem):
    value = report()
    if problem == 'missing_extension':
        value['episodes'] = [r for r in value['episodes'] if r['scene_id'] != 'D06']
        value['num_trials'] = len(value['episodes'])
    elif problem == 'duplicate_seed':
        value['episodes'][0]['seed'] = value['episodes'][1]['seed']
    else:
        value['initial_conditions']['position_m'][0] = [99., 99., 99.]
    with pytest.raises(ValueError):
        validate_navigation_report(value, criterion='navigation-primary-v1')


def test_unknown_criterion_is_rejected():
    with pytest.raises(ValueError, match=r'criterion'):
        validate_navigation_report(report(), criterion='unknown')


def test_benchmark_config_is_explicit_and_retired_release_alias_is_rejected():
    from drone_playground.composition import compose_experiment, validate_config

    config = compose_experiment("navigation/differentiable_pointcloud")
    validate_config(config)
    assert config["evaluation"]["protocol"] == "benchmarks/navigation.yaml"
    assert config["evaluation"]["episodes"] == 25
    assert config["evaluation"]["seed_start"] == 80000
    assert config["training"]["checkpoint_eval_episodes"] == 8
    assert config["training"]["checkpoint_eval_seed_start"] == 50000
    assert config["training"]["checkpoint_eval_metric"] == "navigation-checkpoint_eval-v2"

    with pytest.raises(ValueError, match="evaluation must resolve to a mapping"):
        compose_experiment(
            "navigation/differentiable_pointcloud",
            overrides=["evaluation=navigation_release"],
        )

    native = compose_experiment(
        "papers/super",
        overrides=[
            "+evaluation.protocol=benchmarks/navigation.yaml",
            "evaluation.episodes=null",
            "evaluation.seed_start=null",
        ],
    )
    validate_config(native)
    assert native["evaluation"]["episodes"] == 25
    assert native["evaluation"]["seed_start"] == 80000
