"""Create a native simulation without giving dynamics ownership of a task."""

import jax.numpy as jnp
from crazyflow.envs import FigureEightEnv
from crazyflow.envs.drone_env import DroneEnv
from crazyflow.utils import leaf_replace


def initialize_simulation(
    model, duration, freq, device, start, *, figure_eight=False, control_mode=None
):
    """Retain the pinned reset distributions while separating their ownership.

    The task selects its initial-condition rule. Model-specific nominal parameters
    and native integration are installed by ``bind``. The returned upstream object
    is an initialization/rendering handle; it never executes the task's step.
    """
    source_dynamics = getattr(model, "scene_reference_dynamics", model.forward)
    source_drone = getattr(model, "scene_reference_drone", model.drone)
    use_figure_eight = figure_eight and source_dynamics == model.forward
    if use_figure_eight:
        simulation = FigureEightEnv(
            num_envs=1,
            freq=freq,
            dynamics=source_dynamics,
            drone=source_drone,
            device=device,
            trajectory_time=duration,
            max_episode_time=duration,
        )
    else:

        def nominal_rotors(data, default, mask):
            del default
            speed = 10000.0 if source_dynamics == "first_principles" else 0.05
            return data.replace(
                states=leaf_replace(
                    data.states, mask, rotor_vel=jnp.full_like(data.states.rotor_vel, speed)
                )
            )

        simulation = DroneEnv(
            num_envs=1,
            freq=freq,
            max_episode_time=duration,
            dynamics=source_dynamics,
            drone=source_drone,
            device=device,
            reset_randomization=nominal_rotors,
        )
    try:
        if not use_figure_eight:
            simulation.sim.data = simulation.sim.data.replace(
                states=simulation.sim.data.states.replace(
                    pos=simulation.sim.data.states.pos.at[0, 0].set(jnp.asarray(start))
                )
            )
            simulation.sim.build_default_data()
        model.bind(simulation.sim, control_mode=control_mode)
        if simulation.sim.freq % freq:
            raise ValueError("Control frequency must divide the native physics frequency")
        simulation.n_substeps = simulation.sim.freq // freq
        return simulation
    except BaseException:
        simulation.close()
        raise


def initialize_execution(env, physics_freq=None):
    """Build and own common simulation, controller and sensor runtime resources.

    Tasks may supply a native scene initializer. They return its resource and
    action space; only this function installs the shared environment fields.
    """
    create = getattr(env.task, "create_simulation", None)
    simulation, action_space = (None, None) if create is None else create(env)
    env.simulation = simulation
    env.sim = None if simulation is None else simulation.sim
    try:
        if env.sim is not None:
            actual = env.sim.freq
            if physics_freq is not None and physics_freq != actual:
                raise ValueError(
                    "Requested physics frequency differs from the actual dynamics backend"
                )
            env.physics_freq = actual
            env.default = env.sim.default_data
            env.reset_fn = env.sim.build_reset_fn()
            env.controller.bind(action_space.low, action_space.high, env.dynamics)
            hover = jnp.array([0.0, 0.0, 0.0, float(env.default.params.mass[0]) * 9.81])
            if hasattr(env.controller, "hover"):
                hover = env.controller.hover(env.default)
            env.hover_action = (
                2 * (hover - env.controller.low) / (env.controller.high - env.controller.low) - 1
            )
        else:
            if physics_freq is None:
                raise ValueError(
                    "An unbound dynamics model requires an explicit environment physics_freq"
                )
            env.physics_freq = physics_freq
            env.hover_action = jnp.zeros(len(env.controller.low))
        if env.physics_freq <= 0 or env.physics_freq % env.freq:
            raise ValueError("Control frequency must divide the actual physics frequency")
        env.substeps = env.physics_freq // env.freq
        env.dt_physics = 1.0 / env.physics_freq
        env.low, env.high = env.controller.low, env.controller.high
        env.sensor_calibration = None if env.sensor is None else env.sensor.calibration()
        env.sensor_period = 1
        if env.sensor is not None:
            period = getattr(env.sensor, "period_steps", None)
            env.sensor_period = (
                period(env.freq)
                if period is not None
                else round(env.freq / env.sensor.source_rate_hz)
            )
        env.handles_transport_delay = bool(getattr(env.task, "handles_transport_delay", False))
    except BaseException:
        if env.sim is not None:
            env.sim.close()
        raise


def initialize_rigid_body(env, *, start, figure_eight=False, reset=False):
    """Create the common native resource from the task's initial-condition rule."""
    simulation = initialize_simulation(
        env.dynamics,
        env.duration,
        env.freq,
        env.device,
        start,
        figure_eight=figure_eight,
        control_mode=env.controller.native_mode,
    )
    try:
        if reset:
            simulation.sim.reset()
        return simulation, simulation.single_action_space
    except BaseException:
        simulation.close()
        raise
