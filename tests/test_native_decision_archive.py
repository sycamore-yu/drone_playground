"""Replay downstream commands from recorded native outputs, without a planner."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from drone_playground.evaluation.decision_archive import (
    NativeDecisionRecorder,
    load_native_decisions,
)
from drone_playground.execution.native_tracking import NativeTracking
from drone_playground.native.contracts import MotionCommand, Trajectory, Waypoint


def environment():
    return SimpleNamespace(
        freq=50, low=np.array([-1., -1., -3.2, 0.]), high=np.array([1., 1., 3.2, 1.]),
        default=SimpleNamespace(params=SimpleNamespace(mass=np.array([.027]))),
        controller=SimpleNamespace(input_kind='attitude_thrust'),
        controller_observation=lambda state: state,
    )


@pytest.mark.parametrize('kind', ['trajectory', 'waypoint', 'motion'])
def test_replay_restores_outputs_and_controller_fallback_state(tmp_path, kind):
    env = environment()
    state = dict(pos=np.array([0., 0., 1.]), vel=np.zeros(3), quat=np.array([0., 0., 0., 1.]))
    coefficients = np.zeros((1, 4, 3))
    coefficients[0, :3, 0] = [0., 0., 1.]
    coefficients[0, 0, 1] = .5
    curve = Trajectory(0., [2.], coefficients)
    output = {'trajectory': curve, 'waypoint': Waypoint([[.3, .1, 1.]], .1),
              'motion': MotionCommand('attitude_thrust', [.03, .04, .1, .27])}[kind]
    settings = {'name': {'trajectory': 'trajectory_tracking', 'waypoint': 'waypoint_tracking',
                         'motion': 'direct'}[kind]}
    tracker = NativeTracking(settings, env, state, tmp_path)
    archive = tmp_path / 'decisions'
    expected = []
    with NativeDecisionRecorder(archive, 'attitude_thrust') as recorder:
        for tick in range(3):
            body = dict(state, pos=np.array([tick * .03, .02 * tick, 1.]))
            reply = (dict(output=output, trajectory=curve if kind == 'trajectory' else None,
                          reference=curve.sample(tick / 50) if kind == 'trajectory' else None,
                          valid_until=2., commands=tick + 1)
                     if tick < 2 else dict(output=None, reference=None, trajectory=None,
                                          rejected_reference=True))
            command = tracker.command(reply, body, tick)
            recorder.record(tick, tick / 50, body, reply, command)
            expected.append(command)
    tracker.close()
    fresh = NativeTracking(settings, env, state, tmp_path)
    restored = list(load_native_decisions(archive))
    assert len(restored) == 3 and restored[2]['reply']['rejected_reference'] is True
    for row, command in zip(restored, expected):
        np.testing.assert_array_equal(fresh.command(row['reply'], row['state'], row['tick']), command)
        np.testing.assert_array_equal(row['command'], command)
    fresh.close()
    index = json.loads((archive / 'index.json').read_text())
    assert index['unique_outputs'] == 1  # Curve/output aliases and repeated replies share storage.
    assert index['command_kind'] == 'attitude_thrust'


def test_interruption_retains_decision_without_claiming_applied_command(tmp_path):
    with pytest.raises(RuntimeError, match='controller failed'):
        with NativeDecisionRecorder(tmp_path / 'decisions', 'attitude_thrust') as recorder:
            recorder.record(0, 0., {}, {'output': Waypoint([[1, 2, 3]], .1)}, None)
            raise RuntimeError('controller failed before physics')
    index = json.loads((tmp_path / 'decisions/index.json').read_text())
    assert index['interrupted'] and index['frames'] == 1
    rows = list(load_native_decisions(tmp_path / 'decisions'))
    assert rows[0]['command'] is None and isinstance(rows[0]['reply']['output'], Waypoint)
    frames = tmp_path / 'decisions/frames.jsonl.gz'
    frames.write_bytes(frames.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='digest'):
        list(load_native_decisions(tmp_path / 'decisions'))
