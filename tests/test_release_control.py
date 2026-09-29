import copy

import pytest

from drone_playground.evaluation.release_control import validate_control_report


def report(completed=95, error=.1):
    return dict(split='heldout', parameters_frozen=True, parameter_sha256='test', num_trials=100,
                episodes=[dict(seed=30000+i, completed=i<completed, failed=i>=completed, rmse_m=error)
                          for i in range(100)])


def test_tracking_uses_95_percent_and_keeps_failures():
    assert validate_control_report(report(), 'tracking')['passed']
    result = validate_control_report(report(94), 'tracking')
    assert not result['passed'] and result['failed'] == 6
    assert validate_control_report(report(94), 'racing')['passed']
    assert not validate_control_report(report(100,.26), 'tracking')['passed']


@pytest.mark.parametrize('change', ['duplicates', 'missing', 'dev', 'unfrozen'])
def test_rejects_invalid_release_evidence(change):
    value = copy.deepcopy(report())
    if change == 'duplicates':
        value['episodes'][1]['seed'] = value['episodes'][0]['seed']
    elif change == 'missing':
        value['episodes'].pop()
    elif change == 'dev':
        value['split'] = 'dev'
    else:
        value['parameters_frozen'] = False
    with pytest.raises(ValueError):
        validate_control_report(value, 'tracking')


def test_native_solver_needs_runtime_identity_and_does_not_require_training_seeds():
    value = report()
    value['parameter_identity_kind'] = 'resolved optimization configuration'
    with pytest.raises(ValueError, match='runtime identity'):
        validate_control_report(value, 'tracking')
    value['runtime_identity'] = {'libacados.so': 'a'*64}
    result = validate_control_report(value, 'tracking')
    assert result['passed'] and result['caveat'] == 'Frozen solver; no learning seeds required'
