"""Composition boundary for the paper's point-cloud/acceleration flight task."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from hydra.utils import instantiate

from drone_playground.dynamics.point_mass import PointMassState
from drone_playground.tasks.scenes.navigation import clearance_and_collision, euclidean_norm


@dataclass(frozen=True)
class PaperObservation:
    name: str = "paper_pointcloud_state"
    proprioception_size: int = 10

    def proprioception(self, state, goal, speed, radius):
        """Body velocity/target velocity, world-up row of attitude, vehicle radius."""
        delta = goal - jax.lax.stop_gradient(state.pos)
        distance = euclidean_norm(delta)
        target = delta * jnp.minimum(1.0, speed / jnp.maximum(distance, 1e-6))[..., None]
        body_velocity = jnp.einsum("...ij,...i->...j", state.rotation, state.vel)
        body_target = jnp.einsum("...ij,...i->...j", state.rotation, target)
        r = jnp.full((*state.pos.shape[:-1], 1), radius)
        return jnp.concatenate(
            [body_velocity, body_target, state.rotation[..., 2, :], r], -1
        ), target


class PointCloudTask:
    """Native point-mass state, sharing geometry and protocol utilities with navigation."""

    action_size = 3
    physics_engine = "paper PointMassLag JAX; MuJoCo is used only for replay"

    def __init__(self, config):
        self.config = config
        self.model = instantiate(config["dynamics"], _convert_="all")
        self.controller = instantiate(config["controller"], _convert_="all")
        self.sensor = instantiate(config["observation"]["sensor"], _convert_="all")
        self.observer = instantiate(
            {k: v for k, v in config["observation"].items() if k != "sensor"}
        )
        self.scene = instantiate(config["scene"], _convert_="all")
        self.objective = instantiate(config["objective"], _convert_="all")
        self.freq = int(config["task"]["freq"])
        self.dt = 1.0 / self.freq
        self.duration = float(config["task"]["duration"])
        self.episode_length = round(self.duration * self.freq)
        self.body_radius = float(config["task"]["body_radius"])
        self.goal_radius = float(config["task"]["goal_radius"])
        self.physics_freq = int(config["task"]["physics_freq"])
        self.task = config["task"]["name"]
        self.dynamics = self.model.forward
        self.drone = self.model.drone
        self.component_identity = {
            k: config[k]
            for k in (
                "policy",
                "controller",
                "dynamics",
                "scene",
                "observation",
                "task",
                "objective",
            )
        }
        self.sensor_calibration = self.sensor.calibration()
        self.observation_size = self.sensor.points_per_frame * 4 + self.observer.proprioception_size

    def initial_state(self, bank):
        return PointMassState.create(bank.start)

    def observation(self, bank, state, time, speeds):
        ids = jnp.arange(bank.num_instances)
        points, valid = jax.vmap(lambda i, p, r: self.sensor.sample(bank, i, p, r, time))(
            ids, state.pos, state.rotation
        )
        proprio, target = self.observer.proprioception(state, bank.goal, speeds, self.body_radius)
        return points, valid, proprio, target

    def command(self, body_action, state):
        world = jnp.einsum("...ij,...j->...i", state.rotation, body_action)
        return self.controller.physical_action(world)

    def proximity(self, bank, state, time):
        """Continuous signed clearance and approaching speed for the training objective."""
        offset = jnp.einsum("...ij,j->...i", state.rotation, jnp.array([0.0, 0.0, 0.005]))
        centre = state.pos + offset

        def measure(i, p, v):
            def distance(query):
                return clearance_and_collision(bank, i, time, query, self.body_radius)[0]

            clearance, normal = jax.value_and_grad(distance)(p)
            approaching = jax.lax.stop_gradient(jnp.maximum(-jnp.sum(normal * v), 0.0))
            return clearance, approaching

        return jax.vmap(measure)(jnp.arange(bank.num_instances), centre, state.vel)

    def scenario(self, index):
        return {"scenario_id": int(index), **self.bank.labels(int(index))}

    def close(self):
        pass


def validate_paper_config(config):
    """Validate capabilities independently of the historical P5 task constants."""
    if config["policy"]["output"] != "world_acceleration":
        raise ValueError(
            "Paper policy requires the three-dimensional acceleration command contract"
        )
    if config["controller"]["name"] != "acceleration_passthrough":
        raise ValueError("Select the point-mass acceleration execution preset")
    if config["dynamics"]["forward"] != "point_mass_lag":
        raise ValueError("This method adapter qualifies the point-mass forward model")
    if config["dynamics"]["backward"] not in ("direct", "exponential"):
        raise ValueError("Unqualified point-mass derivative rule")
    if config["algorithm"]["name"] != "pointcloud_bptt":
        raise ValueError("Select the recurrent paper BPTT trainer")
    frequency = config["task"]["freq"]
    if frequency <= 0 or config["task"]["physics_freq"] % frequency:
        raise ValueError("Evaluation physics frequency must be divisible by the policy frequency")
    if config["task"]["body_radius"] <= 0 or config["task"]["duration"] <= 0:
        raise ValueError("Body radius and duration must be positive")
    settings = config["training"]
    if settings["num_envs"] < 1 or settings["policy_updates"] < 1 or settings["seed"] < 0:
        raise ValueError("Training counts must be positive and seed nonnegative")
    horizon = config["algorithm"]["horizon_length"]
    if horizon < 1:
        raise ValueError("BPTT horizon must be positive")
    budget = settings["policy_updates"] * settings["num_envs"] * horizon
    if settings.get("num_timesteps") not in (None, budget):
        raise ValueError("Declared budget differs from updates × environments × horizon")
    if config.get("mode", "train") == "train" and config["scene"]["name"] == "navigation8":
        raise ValueError("Navigation8 is held out for frozen-policy transfer evaluation")
