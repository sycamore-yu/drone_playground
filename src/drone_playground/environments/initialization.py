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
    if figure_eight and source_dynamics == model.forward:
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
        simulation.sim.data = simulation.sim.data.replace(
            states=simulation.sim.data.states.replace(
                pos=simulation.sim.data.states.pos.at[0, 0].set(jnp.asarray(start))
            )
        )
        simulation.sim.build_default_data()
    model.bind(simulation.sim, control_mode=control_mode)
    if simulation.sim.freq % freq:
        simulation.close()
        raise ValueError("Control frequency must divide the native physics frequency")
    simulation.n_substeps = simulation.sim.freq // freq
    return simulation
