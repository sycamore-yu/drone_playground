"""Causality, clock and independent-episode checks for command prediction."""

import crazyflow  # noqa: F401
import numpy as np
import pytest

from drone_playground.actions.controllers.mpc.delay_prediction import IssuedCommandPredictor


def state():
    return dict(pos=np.zeros(3), vel=np.zeros(3), quat=np.array([0., 0., 0., 1.]),
                ang_vel=np.zeros(3))


def acceleration_model(x, u):
    result = np.zeros(12)
    result[:3] = x[6:9]
    result[6] = u[0]
    return result


def test_prediction_uses_only_commands_that_have_arrived_and_resets_history():
    predictor = IssuedCommandPredictor(acceleration_model, .038, np.zeros(4))
    obs = state()
    np.testing.assert_allclose(predictor.predict(obs, 0.)['pos'], 0.)
    first = np.array([2., 0., 0., 0.])
    predictor.record(0., first)
    first[0] = 100  # Issued values are owned by the predictor, not caller memory.
    # At t=.020 the command issued at zero will arrive at .038. Only .020s
    # of the .038s forecast sees acceleration=2.
    predicted = predictor.predict(obs, .020)
    np.testing.assert_allclose(predicted['vel'], [.04, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(predicted['pos'], [.0004, 0., 0.], atol=1e-12)
    predictor.record(.020, [4., 0., 0., 0.])
    predicted = predictor.predict(obs, .040)
    np.testing.assert_allclose(predicted['vel'], [.116, 0., 0.], atol=1e-12)
    np.testing.assert_allclose(predicted['pos'], [.001844, 0., 0.], atol=1e-12)
    np.testing.assert_array_equal(obs['pos'], np.zeros(3))
    with pytest.raises(ValueError, match=r'precede recording'):
        predictor.predict(obs, .020)
    predictor.reset()
    np.testing.assert_array_equal(predictor.predict(obs, 0.)['pos'], np.zeros(3))
    predictor.record(0., [1., 0., 0., 0.])


def test_hover_orientation_and_body_rate_round_trip():
    from scipy.spatial.transform import Rotation

    obs = state()
    obs['quat'] = Rotation.from_euler('xyz', [.1, -.2, .3]).as_quat()
    obs['ang_vel'] = np.array([.05, -.03, .1])
    predictor = IssuedCommandPredictor(lambda x, u: np.zeros(12), .038, np.zeros(4))
    predicted = predictor.predict(obs, 0.)
    np.testing.assert_allclose(predicted['quat'], obs['quat'], atol=1e-10)
    np.testing.assert_allclose(predicted['ang_vel'], obs['ang_vel'], atol=1e-8)


@pytest.mark.parametrize('delay', [-1., 0., np.nan, np.inf])
def test_rejects_invalid_delay_estimates(delay):
    with pytest.raises(ValueError, match=r'finite and positive'):
        IssuedCommandPredictor(acceleration_model, delay, np.zeros(4))
