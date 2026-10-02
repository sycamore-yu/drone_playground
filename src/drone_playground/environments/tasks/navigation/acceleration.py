"""Composition boundary for the paper's point-cloud/acceleration flight task."""

import jax
import jax.numpy as jnp
from hydra.utils import instantiate

from drone_playground.dynamics.point_mass import PointMassState
from drone_playground.environments.scenes.geometry import clearance_and_collision


class AccelerationNavigationEnv:
    """Native point-mass state, sharing geometry and protocol utilities with navigation."""

    action_size = 3
    physics_engine = "paper PointMassLag JAX; MuJoCo is used only for replay"

    def __init__(self, config):
        from drone_playground.environments.environment import build_dynamics, component_identity

        self.config = config
        self.observation_noise = (
            (config["training"].get("observation_noise") or {})
            if config.get("mode", "train") == "train"
            else {}
        )
        self.model = build_dynamics(config)
        self.controller = instantiate(
            config["env"]["action"]["controller"], _convert_="all"
        )
        self.sensor = instantiate(config["env"]["sensor"], _convert_="all")
        self.observer = instantiate(
            {
                k: v
                for k, v in config["env"]["observation"].items()
                if k != "sensor"
            }
        )
        self.scene = instantiate(config["env"]["scene"], _convert_="all")
        self.objective = instantiate(config["objective"], _convert_="all")
        self.freq = int(config["env"]["task"]["freq"])
        self.dt = 1.0 / self.freq
        self.duration = float(config["env"]["task"]["duration"])
        self.episode_length = round(self.duration * self.freq)
        self.body_radius = float(config["env"]["task"]["body_radius"])
        self.goal_radius = float(config["env"]["task"]["goal_radius"])
        from drone_playground.environments.tasks.navigation.rigid_body import NavigationTask

        self.task_definition = NavigationTask(
            self.goal_radius, self.body_radius
        )
        self.arrival_sampling = config["env"]["task"].get(
            "arrival_sampling", "policy"
        )
        self.physics_freq = int(config["env"]["task"]["physics_freq"])
        self.task = config["env"]["task"]["name"]
        self.dynamics = self.model.forward
        self.drone = self.model.drone
        self.component_identity = component_identity(config)
        self.sensor_calibration = self.sensor.calibration()
        self.observation_size = (
            self.sensor.points_per_frame * self.sensor.policy_value_channels
            + self.observer.proprioception_size
        )

    def initial_state(self, bank, key=None):
        key = jax.random.PRNGKey(0) if key is None else key
        state = PointMassState.create(bank.start).replace(
            measurement_key=jax.random.split(key, bank.num_instances)
        )
        if getattr(self, "role", "train") == "train":
            from drone_playground.environments.randomization import reset_point_mass_state

            state = reset_point_mass_state(
                state,
                jax.random.fold_in(key, 1),
                self.config["training"].get("reset_randomization") or {},
            )
        return self.model.randomize(state, jax.random.fold_in(key, 2))

    def observation(self, bank, state, time, speeds):
        ids = jnp.arange(bank.num_instances)
        points, valid = jax.vmap(
            lambda i, p, r: self.sensor.sample(bank, i, p, r, time)
        )(ids, state.pos, state.rotation)
        from drone_playground.environments.randomization import noisy_point_mass_state

        measured = noisy_point_mass_state(
            state, time, self.dt, self.observation_noise
        )
        proprio, _ = self.observer.proprioception(
            measured, bank.goal, speeds, self.body_radius
        )
        # Corrupt the actor input without changing the ground-truth loss target.
        _, target = self.observer.proprioception(
            state, bank.goal, speeds, self.body_radius
        )
        points, valid = self.measurement(points, valid, state, time)
        return points, valid, proprio, target

    def measurement(self, points, valid, state, time):
        """Seeded noise in physical sensor units, shared by train and frozen eval."""
        noise = self.observation_noise
        std, dropout = noise.get("sensor_std_m", 0.0), noise.get(
            "sensor_dropout_probability", 0.0
        )
        if not std and not dropout:
            return points, valid
        from drone_playground.environments.randomization import point_measurement_noise

        keys = jax.vmap(jax.random.fold_in)(
            state.measurement_key,
            jnp.broadcast_to(
                jnp.floor(jnp.asarray(time) / self.dt).astype(jnp.int32),
                state.pos.shape[:-1],
            ),
        )
        if self.sensor.policy_value_channels == 1:
            noisy, mask = jax.vmap(
                lambda x, v, k: point_measurement_noise(
                    x[..., None], v, k, std, dropout
                )
            )(points, valid, keys)
            return noisy[..., 0], mask
        noisy, mask = jax.vmap(
            lambda x, v, k: point_measurement_noise(x, v, k, std, dropout)
        )(points, valid, keys)
        return noisy, mask

    def command(self, body_action, state):
        body_action = self.uncertain_action(body_action, state)
        world = jnp.einsum("...ij,...j->...i", state.rotation, body_action)
        return self.controller.physical_action(world)

    def uncertain_action(self, action, state):
        noise = (
            getattr(self, "environment_effects", {}).get("action_noise") or {}
        )
        if not noise:
            return action
        if set(noise) - {"std_physical", "bias_physical"}:
            raise ValueError(
                "Acceleration action uncertainty uses physical m/s^2 units"
            )
        keys = jax.vmap(jax.random.fold_in)(
            state.measurement_key,
            jnp.floor(state.elapsed_time / self.dt).astype(jnp.int32),
        )
        draws = jax.vmap(lambda key: jax.random.normal(key, (3,)))(keys)
        return (
            action
            + draws * jnp.asarray(noise.get("std_physical", 0.0))
            + jnp.asarray(noise.get("bias_physical", 0.0))
        )

    def proximity(self, bank, state, time):
        """Continuous signed clearance and approaching speed for the training objective."""
        offset = jnp.einsum(
            "...ij,j->...i", state.rotation, jnp.array([0.0, 0.0, 0.005])
        )
        centre = state.pos + offset

        def measure(i, p, v):
            def distance(query):
                return clearance_and_collision(
                    bank, i, time, query, self.body_radius
                )[0]

            clearance, normal = jax.value_and_grad(distance)(p)
            approaching = jax.lax.stop_gradient(
                jnp.maximum(-jnp.sum(normal * v), 0.0)
            )
            return clearance, approaching

        return jax.vmap(measure)(
            jnp.arange(bank.num_instances), centre, state.vel
        )

    def scenario(self, index):
        return {"scenario_id": int(index), **self.bank.labels(int(index))}

    def close(self):
        pass
