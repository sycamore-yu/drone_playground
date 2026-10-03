"""A Brax environment composed from six physical/task components.

The environment owns command execution and the physics clock. Tasks supply reset,
observations, events and rewards. Neither side selects a learning algorithm or a
planner implementation.
"""

from __future__ import annotations

from brax.envs.base import Env, State

from drone_playground.control.setpoints import Actuation, Setpoint
from drone_playground.control.transition import ActionTransition
from drone_playground.references import Reference


class DroneEnvironment(Env):
    """Execute one complete control interval and then evaluate the task."""

    def __init__(
        self,
        *,
        dynamics,
        controller,
        reference,
        scene,
        sensor,
        task,
        device="cpu",
        role="eval",
        count=1,
        seed=20000,
        conditions=None,
        name=None,
    ):
        self.dynamics = dynamics
        self.controller = controller
        self.reference = reference
        self.scene = scene
        self.sensor = sensor
        self.task = task
        self.name = task.name if name is None else name
        self.device, self.role, self.count = device, role, count
        self.conditions = {} if conditions is None else conditions
        self.reference_seed = seed
        self.freq, self.duration = task.freq, task.duration
        if self.freq <= 0 or self.duration <= 0 or count < 1:
            raise ValueError("Environment frequency, duration and count must be positive")
        self.episode_length = round(self.freq * self.duration)
        self.time_limit_kind = task.time_limit_kind
        self.command_distribution = self.conditions.get(
            "command_distribution", getattr(task, "command_distribution", {})
        )
        self.observation_noise = self.conditions.get("observation_noise") or {}
        self.environment_effects = {
            name: self.conditions.get(name)
            for name in ("reset_randomization", "observation_noise", "action_noise", "disturbance")
        }
        self.drone = dynamics.drone
        task.bind(self)
        if self.physics_freq <= 0 or self.physics_freq % self.freq:
            raise ValueError("Control frequency must divide the actual physics frequency")
        self.substeps = self.physics_freq // self.freq
        self.dt_physics = 1.0 / self.physics_freq
        self.low, self.high = controller.low, controller.high
        self.transition = ActionTransition(
            lambda data, physical: controller.apply(task.physics(data), physical),
            lambda data, control, dt: task.with_physics(
                data, dynamics.step(task.physics(data), control, dt)
            ),
            self.substeps,
            self.dt_physics,
        )
        self.reset_info_fields = tuple(getattr(task, "reset_info_fields", ()))
        if hasattr(dynamics, "physical_parameters"):
            self.reset_info_fields += ("physical_parameters",)

    @property
    def dt(self):
        """Duration of one environment control interval in seconds."""
        return 1.0 / self.freq

    @property
    def action_size(self):
        """Width of the configured policy action encoding."""
        return len(self.low)

    @property
    def observation_size(self):
        """Size declared by the task's observation specification."""
        if hasattr(self.task, "observation_size"):
            return self.task.observation_size
        return self.task.observation.size

    @property
    def backend(self):
        """Concrete dynamics implementation used for this run."""
        return type(self.dynamics).__name__

    def reset(self, rng, *args, **kwargs) -> State:
        """Reset only the requested instance; wrappers own the batch mask."""
        return self.task.reset(self, rng, *args, **kwargs)

    def observation(self, data):
        """Encode task inputs without modifying physical state."""
        return self.task.observe(self, data)

    def physical_action(self, action):
        """Decode normalized policy output with the selected controller."""
        return self.controller.physical_action(action)

    def step(self, state: State, action) -> State:
        """Apply an action, advance physics, then compute task events and reward."""
        if isinstance(action, Reference):
            control = self.controller.apply(self.task.physics(state.pipeline_state), action)
            return self._typed_transition(state, control)
        if isinstance(action, Setpoint | Actuation):
            return self._typed_transition(state, action)
        return self._transition(state, action, self.physical_action(action))

    def _typed_transition(self, state, value):
        physical = self.controller.input_values(value)
        action = self.controller.normalize(physical)
        return self._transition(state, action, value, record_physical=physical)

    def step_physical(self, state, physical):
        """Execute a physical input in the configured control units."""
        if isinstance(physical, Reference | Setpoint | Actuation):
            return self.step(state, physical)
        action = self.controller.normalize(physical)
        return self._transition(state, action, physical)

    def step_schedule(self, state, commands):
        """Execute commands delivered on physics ticks within one control interval."""
        physical = commands[-1]
        action = self.controller.normalize(physical)
        return self._transition(state, action, physical, commands)

    def _transition(self, state, action, physical, commands=None, record_physical=None):
        advance = getattr(self.task, "advance", None)
        if advance is not None:
            data, evidence = advance(self, state, physical, commands)
            values = physical if record_physical is None else record_physical
            return self.task.finish(self, state, data, action, values, evidence)
        probe = self.task.probe(self, state)
        if commands is not None:
            result = self.transition.step_schedule(state.pipeline_state, commands, probe)
        elif probe is not None:
            result = self.transition.step_with_evidence(state.pipeline_state, physical, probe)
        else:
            result = self.transition.step(state.pipeline_state, physical)
        if probe is None:
            data, evidence = result, None
        else:
            data, clearance, collided = result
            evidence = (clearance, collided)
        values = physical if record_physical is None else record_physical
        return self.task.finish(self, state, data, action, values, evidence)

    def index(self, data):
        """Return task reference progress, independent of simulation backend."""
        return self.task.index(self, data)

    def scenario(self, scenario_id):
        """Return the task's scene identity for recording."""
        return self.bank.describe(int(scenario_id))

    def controller_observation(self, state):
        """Expose a narrow physical view to host controllers."""
        return self.task.controller_observation(self, state)

    def proprioception(self, data):
        """Return the task's declared critic-state view."""
        return self.task.proprioception(self, data)

    @property
    def realised_sensor_rate_hz(self):
        """Actual capture rate after control-clock scheduling."""
        return None if self.sensor is None else self.freq / self.sensor_period

    def close(self):
        """Release task-owned rendering and native simulation resources."""
        close = getattr(self.task, "close", None)
        if close is not None:
            close(self)
