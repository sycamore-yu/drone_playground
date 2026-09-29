"""JAX network -> physical reference -> PD -> delayed actuator -> dynamics.

The wrapped environment still owns actuator units, delay, physics and task
events. Network action dimensions never enter the actuator's four-value queue.
"""

import jax.numpy as jnp
from brax.envs.base import Wrapper

from drone_playground.networks.physical_outputs import PhysicalOutput

from .controllers.trajectory_jax import JaxTrajectoryTracking


class GeometricPolicyExecution(Wrapper):
    def __init__(self, env, decoder, tracker):
        super().__init__(env)
        self.decoder = PhysicalOutput(**decoder)
        if self.decoder.kind == 'waypoint' and self.decoder.count != 1:
            raise ValueError('JAX waypoint PD currently requires one target per policy decision')
        self.tracker = JaxTrajectoryTracking(**{k:v for k,v in tracker.items() if k!='name'}).bind(env.low,env.high)
        if self.decoder.kind == 'trajectory' and not 0 < self.tracker.lead_seconds <= self.decoder.horizon_seconds:
            raise ValueError('JAX trajectory PD requires a positive lead within its trajectory horizon')
        self.reset_info_fields = getattr(env,'reset_info_fields',())
        self.hover_action = jnp.zeros(self.action_size)

    @property
    def action_size(self):
        return self.decoder.action_size

    def policy_command(self, state, action):
        data = state.pipeline_state.sim_data.states
        body = {name:getattr(data,name)[0,0] for name in ('pos','vel','quat','ang_vel')}
        goal = self.env.observer.reference_goal(state.obs)
        decoded = self.decoder.decode(action,body['pos'],body['vel'],goal)
        if self.decoder.kind == 'waypoint':
            reference = dict(position=decoded[0],velocity=jnp.zeros(3),acceleration=jnp.zeros(3))
        else:
            t = self.tracker.lead_seconds
            powers = jnp.asarray([t**i for i in range(6)])
            first = jnp.asarray([0.,1.,2*t,3*t**2,4*t**3,5*t**4])
            second = jnp.asarray([0.,0.,2.,6*t,12*t**2,20*t**3])
            reference = dict(position=decoded[:3]@powers,velocity=decoded[:3]@first,
                             acceleration=decoded[:3]@second)
        x,y,z,w = body['quat']
        reference['yaw'] = jnp.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
        mass = self.env.default.params.mass.reshape(-1)[0] if hasattr(self.env.default,'params') else self.env.default.sim_data.params.mass.reshape(-1)[0]
        return self.tracker.command(body,reference,mass)

    def reset(self, rng, *args, **kwargs):
        state = self.env.reset(rng,*args,**kwargs)
        return state.replace(info={**state.info,'applied_action':self.env.hover_action,
                                   'requested_geometric_action':jnp.zeros(self.action_size)})

    def step(self, state, action):
        physical = self.policy_command(state,action)
        normalized = 2*(physical-self.env.low)/(self.env.high-self.env.low)-1
        result = self.env.step(state,normalized)
        # Delayed execution records its actually applied command. Without delay,
        # the current command is applied directly and must replace the reset value.
        applied = result.info['applied_action'] if hasattr(self.env,'delay_steps') or hasattr(self.env,'delay_range_ms') else normalized
        return result.replace(info={**result.info,'applied_action':applied,
                                    'requested_geometric_action':action})

    def physical_action(self, action):
        raise ValueError('Geometric actions require state; use policy_command(state, action)')
