import numpy as np
import pytest

from drone_playground.composition import compose_method, validate_config
from drone_playground.environments.tasks.pointcloud_navigation import PointCloudNavigationTask
from drone_playground.evaluation.navigation_resets import navigation_resets


@pytest.fixture(scope='module')
def setup():
    config = compose_method('learning/pointcloud_navigation', overrides=[
        'evaluation=navigation_release', 'evaluation.episodes=25'])
    validate_config(config)
    return PointCloudNavigationTask(config), config['evaluation']['initial_conditions']


def test_frozen_resets_are_safe_reproducible_and_have_actual_variation(setup):
    task, spec = setup
    ids = np.arange(task.bank.num_instances)
    seeds = np.arange(30000,30000+len(ids))
    first = navigation_resets(task.bank, ids, seeds, spec, task.body_radius)
    second = navigation_resets(task.bank, ids, seeds, spec, task.body_radius)
    changed = navigation_resets(task.bank, ids, seeds+1000, spec, task.body_radius)
    np.testing.assert_array_equal(first['position'],second['position'])
    assert not np.array_equal(first['position'], changed['position'])
    assert np.min(first['record']['clearance_m']) >= .15
    assert np.all(np.abs(first['position']-task.bank.start) <= np.asarray(spec['position_half_width_m'])+1e-6)
    np.testing.assert_allclose(np.linalg.norm(first['quaternion'],axis=-1),1.,atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(first['rotation']),1.,atol=1e-6)
    # Batching/order must not change a case's independent seed interpretation.
    reverse = navigation_resets(task.bank, ids[::-1], seeds[::-1], spec, task.body_radius)
    np.testing.assert_array_equal(first['position'],reverse['position'][::-1])


def test_evaluator_starts_from_frozen_physical_state_and_records_values(setup):
    import copy

    import jax.numpy as jnp

    from drone_playground.evaluation.pointcloud_navigation import PointCloudNavigationEvaluator

    class ZeroPolicy:
        hidden_size = 1
        def apply(self, params, points, valid, proprio, memory):
            return jnp.zeros((len(proprio),3)), memory

    original, spec = setup
    task = copy.copy(original)
    task.duration, task.episode_length = .1, 1
    evaluator = PointCloudNavigationEvaluator(task, ZeroPolicy(), repeats=1, initial_conditions=spec)
    report, trace = evaluator.run({})
    np.testing.assert_allclose(trace['observation_pos'][0], report['initial_conditions']['position_m'])
    assert report['initial_conditions']['seeds'] == report['reset_seeds']
    assert len({tuple(row['initial_position_m']) for row in report['episodes']}) == 8
    for row in report['episodes']:
        assert 13 <= row['delay_ticks'] <= 25
    assert 'initial pose/velocity' in report['evaluation_scope']


def test_depth_method_accepts_same_navigation_protocol():
    config = compose_method('learning/depth_navigation', overrides=[
        'evaluation=navigation_release', 'evaluation.episodes=25'])
    validate_config(config)


def test_navigation_release_counts_both_tasks_and_keeps_all_failures():
    import copy

    from drone_playground.evaluation.navigation_resets import validate_navigation_report

    ids = ['S01','S02','S03','S06','D01','D02','D03','D06']*25
    rows = [dict(scene_id=name,seed=30000+i,arrived=True,outcome='arrived',
                 initial_position_m=[i*.001,0,1]) for i,name in enumerate(ids)]
    report = dict(split='heldout',parameters_frozen=True,parameter_sha256='test',num_trials=200,
                  episodes=rows,initial_conditions=dict(seeds=[row['seed'] for row in rows],
                                                        position_m=[row['initial_position_m'] for row in rows]))
    assert validate_navigation_report(report)['passed']
    failed = copy.deepcopy(report)
    for row in [r for r in failed['episodes'] if r['scene_id'].startswith('D')][:11]:
        row.update(arrived=False,outcome='collision')
    result = validate_navigation_report(failed)
    assert result['tasks']['static']['passed'] and not result['tasks']['dynamic']['passed']
    failed['episodes'].pop()
    with pytest.raises(ValueError,match='Missing'):
        validate_navigation_report(failed)
