import copy
from dataclasses import asdict

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.networks.physical_outputs import PhysicalOutput


def test_trajectory_decoder_matches_all_physical_endpoints_and_jax_derivative():
    head = PhysicalOutput('trajectory', anchor='goal', horizon_seconds=1.7,
                          position_scale_m=(2,3,4), velocity_scale_mps=(4,2,1),
                          acceleration_scale_mps2=(3,3,3))
    action = jnp.linspace(-.3,.5,9)
    start, velocity, goal = map(jnp.array, ([1.,2.,3.],[.4,-.2,.1],[5.,1.,4.]))
    decoded = jax.jit(head.decode)(action,start,velocity,goal)
    curve = head.message(decoded, 2.3)
    samples = curve.sample_many([2.3,4.])
    np.testing.assert_allclose(samples['position'], [start,goal+action[:3]*jnp.array([2,3,4])], atol=2e-5)
    np.testing.assert_allclose(samples['velocity'], [velocity,action[3:6]*jnp.array([4,2,1])], atol=2e-5)
    np.testing.assert_allclose(samples['acceleration'], [np.zeros(3),action[6:]*3], atol=3e-5)
    assert not curve.yaw_defined
    # An interior reference sample must preserve parameter derivatives, rather
    # than merely having the right final-point values or shape.
    powers = jnp.asarray([.6**i for i in range(6)])
    def sample(a):
        return head.decode(a,start,velocity,goal)[:3] @ powers
    derivative = jax.jacfwd(sample)(action)
    eps = 1e-3
    finite = np.stack([(np.asarray(sample(action.at[i].add(eps)))-np.asarray(sample(action.at[i].add(-eps))))/(2*eps) for i in range(9)],axis=-1)
    np.testing.assert_allclose(derivative,finite,atol=5e-4,rtol=3e-3)
    assert np.linalg.norm(derivative) > .1
    batched = jax.jit(jax.vmap(head.decode))(jnp.stack([action,action]),jnp.stack([start,start]),jnp.stack([velocity,velocity]),jnp.stack([goal,goal]))
    np.testing.assert_allclose(batched[0],decoded,atol=1e-5)


def test_waypoint_head_keeps_order_anchor_and_physical_scale():
    head = PhysicalOutput('waypoint', count=2, anchor='position',position_scale_m=(2,3,4),tolerance_m=.15)
    decoded = head.decode(jnp.array([0.,.1,.2,.3,.4,.5]), jnp.array([1.,2.,3.]),jnp.zeros(3),jnp.ones(3)*10)
    points = head.message(decoded,0.)
    np.testing.assert_allclose(points.positions,[[1.,2.3,3.8],[1.6,3.2,5.]],atol=1e-6)
    assert points.tolerance == .15
    with pytest.raises(ValueError,match='dimensions'):
        head.decode(jnp.zeros(4),jnp.zeros(3),jnp.zeros(3),jnp.zeros(3))


@pytest.mark.parametrize('kind',['waypoint','trajectory'])
def test_neural_physical_artifact_roundtrip_and_pipeline_consumption(tmp_path,kind):
    from brax.training.acme import running_statistics, specs

    from drone_playground.composition import (
        build_environment,
        compose_method,
        native_training_config,
    )
    from drone_playground.integrations.pipeline import PipelinePlanner
    from drone_playground.networks.policies import network_factory
    from drone_playground.runs.checkpoints import save_policy

    base = compose_method('learning/bptt','tracking')
    base['network']['hidden_sizes'] = [16,16]
    config = copy.deepcopy(base)
    config['method']['output'] = kind
    config['env']['task']['freq'] = 10
    decoder = dict(kind=kind, anchor='goal', position_scale_m=[2.,2.,2.])
    native = native_training_config(config)
    width = PhysicalOutput(**decoder).action_size
    network = network_factory(native)(43,width)
    params = (running_statistics.init_state(specs.Array((43,),jnp.float32)),
              network.policy_network.init(jax.random.key(7)))
    checkpoint = save_policy(tmp_path/'weights',params,native,0,physical_decoder=decoder)
    env = build_environment(base,'cpu')
    try:
        stages = [dict(implementation='frozen_neural',checkpoint=str(checkpoint),frequency_hz=10)]
        if kind == 'waypoint':
            stages.append(dict(implementation='minimum_jerk'))
        planner = PipelinePlanner(dict(output='trajectory',stages=stages),tmp_path/'chain',env)
        try:
            planner.start({},[2.,0.,1.])
            state=env.reset(jax.random.key(2))
            packet=dict(time=0.,position=[0.,0.,1.],velocity=[.1,0.,0.],quaternion=[0.,0.,0.,1.],policy_observation=np.asarray(state.obs).tolist())
            result=planner.step(packet)
            curve=result['output']
            expected=planner.modules[0].policy.act(state.obs)[:3]*2+jnp.array([2.,0.,1.])
            np.testing.assert_allclose(curve.sample(curve.end_time)['position'],expected,atol=2e-5)
            np.testing.assert_allclose(curve.sample(0.)['velocity'],packet['velocity'],atol=1e-6)
            fields=planner.contracts[0]['provenance']['physical_decoder']
            recorded=PhysicalOutput(**fields)
            assert asdict(recorded)==asdict(PhysicalOutput(**decoder))
            assert set(fields)==set(asdict(recorded))
            # A 10Hz policy is held between calls in a 50Hz execution loop.
            planner.step(dict(packet,time=.02))
            planner.step(dict(packet,time=.04))
            assert planner.module_calls[0] == 1
            planner.step(dict(packet,time=.1))
            assert planner.module_calls[0] == 2
        finally:
            planner.close()
        with pytest.raises(ValueError,match='clock'):
            PipelinePlanner(dict(output=kind,stages=[dict(
                implementation='frozen_neural',checkpoint=str(checkpoint))]),tmp_path/'bad-clock',env)
        # The sidecar must not silently reinterpret a geometric policy as a
        # vector without units, or assign the wrong number of waypoints.
        import json

        metadata_path=checkpoint.with_suffix('.json')
        metadata=json.loads(metadata_path.read_text())
        invalid=copy.deepcopy(metadata)
        invalid.pop('physical_decoder')
        metadata_path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError,match='explicit physical decoder'):
            PipelinePlanner(dict(output=kind,stages=stages[:1]),tmp_path/'missing-decoder',env)
        if kind == 'waypoint':
            metadata['physical_decoder']['count']=2
            metadata_path.write_text(json.dumps(metadata))
            with pytest.raises(ValueError,match='dimensions'):
                PipelinePlanner(dict(output=kind,stages=stages[:1]),tmp_path/'bad-width',env)
    finally:
        env.close()


def test_physical_head_contract_rejects_invalid_units_and_horizons():
    for spec in [dict(kind='trajectory',horizon_seconds=0),dict(kind='waypoint',count=0),
                 dict(kind='waypoint',position_scale_m=[1,float('nan'),1]),dict(kind='trajectory',count=2)]:
        with pytest.raises(ValueError):
            PhysicalOutput(**spec)
