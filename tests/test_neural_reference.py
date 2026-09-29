"""Frozen controllers consume external physical references with recorded semantics."""

import json
from types import SimpleNamespace

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.acme import running_statistics, specs

from drone_playground.composition import compose_method, native_training_config
from drone_playground.environments.observations import NavigationObservation
from drone_playground.execution.controllers.crazyflow import AttitudeControl
from drone_playground.integrations.pipeline import FrozenNeuralCommand, PipelinePlanner
from drone_playground.native.contracts import Trajectory, Waypoint
from drone_playground.networks.policies import network_factory
from drone_playground.runs.checkpoints import save_policy


@pytest.fixture
def artifact(tmp_path):
    config = compose_method('learning/bptt', 'tracking')
    config['network']['hidden_sizes'] = [16, 16]
    native = native_training_config(config)
    network = network_factory(native)(43, 4)
    params = (running_statistics.init_state(specs.Array((43,), jnp.float32)),
              network.policy_network.init(jax.random.key(9)))
    checkpoint = save_policy(tmp_path/'weights', params, native, 0)
    model = config['env']['execution']['dynamics']
    control = AttitudeControl().bind([-1., -1., -3.14, 0.], [1., 1., 3.14, 1.])
    # The simulator may expose navigation features; this controller's reference
    # observation must be reconstructed from typed state and the upstream plan.
    env = SimpleNamespace(freq=50, drone=model['drone'], dynamics=model['forward'],
                          controller=control, physical_action=control.physical_action,
                          observer=NavigationObservation(), observation_size=20)
    packet = dict(time=2., position=[1., 2., 3.], velocity=[.1, .2, .3],
                  quaternion=[0., 0., 0., 1.], angular_velocity=[.01, .02, .03],
                  policy_observation=[999.]*20)
    curve = Trajectory(2., [2.], np.array([[[1., .5], [2., -.2], [3., .1], [0., 0.]]]))
    return checkpoint, env, packet, curve


def test_upstream_curve_replaces_reference_fields_and_retains_state_semantics(artifact):
    checkpoint, env, packet, curve = artifact
    module = FrozenNeuralCommand(checkpoint, env, input_kind='trajectory')
    module.start({}, [100., 100., 100.], None, {})
    reply = module.step(packet, curve)
    samples = curve.sample_many(2. + np.arange(10)*.1)['position']
    expected_obs = np.r_[packet['position'], packet['quaternion'], packet['velocity'],
                        packet['angular_velocity'], (samples-packet['position']).ravel()]
    expected = env.physical_action(module.policy.act(jnp.asarray(expected_obs)))
    np.testing.assert_allclose(reply['output'].values, expected, atol=1e-6)
    assert module.provenance['reference_source'] == 'upstream_trajectory'
    np.testing.assert_allclose(module.provenance['reference_offsets_seconds'], np.arange(10)*.1)
    other = curve.coefficients.copy()
    other[0, 1, 0] += 2.
    changed = module.step(packet, Trajectory(2., [2.], other))
    assert np.linalg.norm(changed['output'].values-reply['output'].values) > 1e-5
    # Source-only mode still rejects a simulator observation with different meanings.
    with pytest.raises(ValueError, match='observation'):
        FrozenNeuralCommand(checkpoint, env)


def test_missing_future_is_visible_and_untimed_waypoints_are_not_invented_trajectories(artifact):
    checkpoint, env, packet, curve = artifact
    module = FrozenNeuralCommand(checkpoint, env, input_kind='trajectory')
    module.start({}, [0., 0., 1.], None, {})
    for short in [Trajectory(2., [.8], curve.coefficients),
                  Trajectory(2.02, [2.], curve.coefficients)]:
        result = module.step(packet, short)
        assert result['output'] is None and result['reason'] == 'insufficient_reference_horizon'
        assert result['required_reference_until'] == pytest.approx(2.9)
    with pytest.raises(ValueError, match='Trajectory'):
        module.step(packet, Waypoint([[2., 0., 1.]], .1))
    with pytest.raises(ValueError, match='explicitly timed'):
        FrozenNeuralCommand(checkpoint, env, input_kind='waypoint')
    metadata = json.loads(checkpoint.with_suffix('.json').read_text())
    metadata['config']['env']['observation']['interval'] = .105
    checkpoint.with_suffix('.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='integer number'):
        FrozenNeuralCommand(checkpoint, env, input_kind='trajectory')


def test_waypoint_planning_to_frozen_controller_is_a_configured_chain(artifact, tmp_path):
    checkpoint, env, packet, _ = artifact
    settings = dict(output='attitude_thrust', stages=[
        dict(implementation='goal'), dict(implementation='minimum_jerk'),
        dict(implementation='frozen_neural', checkpoint=str(checkpoint), input='trajectory')])
    chain = PipelinePlanner(settings, tmp_path/'chain', env)
    try:
        chain.start({}, [5., 1., 3.])
        reply = chain.step(dict(packet, time=0.))
        assert reply['output'].kind == 'attitude_thrust'
        assert chain.module_calls == [1, 1, 1]
        assert chain.contracts[-1]['input'] == 'trajectory'
        assert chain.contracts[-1]['output'] == 'attitude_thrust'
        assert chain.contracts[-1]['derivatives'] == 'none'
    finally:
        chain.close()


def test_public_pipeline_accepts_state_only_host_for_external_reference_controllers():
    from drone_playground.composition import validate_config

    config = compose_method('pipeline', 'hovering', ['method.input_sensor=none'])
    config['env']['observation']['name'] = 'state'
    validate_config(config)
    config['env']['observation']['name'] = 'navigation_depth'
    with pytest.raises(ValueError, match='state or state_reference'):
        validate_config(config)
