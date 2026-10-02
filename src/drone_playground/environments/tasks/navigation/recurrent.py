"""Explicit navigation-domain adaptation of the recurrent point-cloud method.

The actor still receives only the paper point cloud and ten proprioceptive
fields. Geometry is used for physical evaluation, differentiable loss, and
collision-free initial-state sampling, never for a hidden route or controller.
"""

import jax
import jax.numpy as jnp
from hydra.utils import instantiate

from drone_playground.dynamics.point_mass import PointMassState, acceleration_attitude
from drone_playground.environments.tasks.navigation.acceleration import AccelerationNavigationEnv


def _select(mask, new, old):
    return jax.tree.map(
        lambda a, b: jnp.where(
            mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim)), a, b
        ),
        new,
        old,
    )


class RecurrentNavigationEnv(AccelerationNavigationEnv):
    handles_transport_delay = True

    def __init__(self, config):
        super().__init__(config)
        self.settings = config["env"]["task"]
        self.physics_dt = 1 / self.physics_freq
        self.substeps = self.physics_freq // self.freq
        self.bank, self.manifest = self.scene.build()
        self.training_bank, self.training_manifest = self.bank, self.manifest
        distribution = config["training"].get("scene_distribution") or {
            "type": "fixed"
        }
        if distribution.get("type") not in ("fixed", "generated", "procedural"):
            raise ValueError(
                "Scene distribution must be fixed, generated or procedural"
            )
        training_scene = distribution.get("scene")
        if training_scene and config.get("mode", "train") == "train":
            self.training_bank, self.training_manifest = instantiate(
                training_scene, _convert_="all"
            ).build()
            if self.training_bank.digest() == self.bank.digest():
                raise ValueError(
                    "Independent training geometry must differ from evaluation"
                )
        self.physics_engine = f"PointMassLag JAX {self.physics_freq}Hz with transport delay; MuJoCo replay only"

    def select_bank(self, indices, *, source=None):
        source = self.bank if source is None else source
        return source.select(indices)

    def clearance(self, bank, state, time):
        centre = state.pos + jnp.einsum(
            "...ij,j->...i", state.rotation, jnp.array([0, 0, 0.005])
        )
        return jax.vmap(
            lambda i, p, t: self.task_definition.clearance(bank, i, p, t)[0]
        )(
            jnp.arange(bank.num_instances),
            centre,
            jnp.broadcast_to(time, (bank.num_instances,)),
        )

    def observation(self, bank, state, time, speeds):
        points, valid = jax.vmap(
            lambda i, p, r, t: self.sensor.sample(bank, i, p, r, t)
        )(
            jnp.arange(bank.num_instances),
            state.pos,
            state.rotation,
            jnp.broadcast_to(time, (bank.num_instances,)),
        )
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

    def delays(self, key, count):
        low, high = self.config["runtime"]["action_delay_ms"]
        milliseconds = jax.random.uniform(
            key, (count,), minval=low, maxval=high
        )
        return jnp.ceil(milliseconds / (1000 * self.physics_dt) - 1e-6).astype(
            jnp.int32
        )

    def training_initial(self, key, count):
        ik, pk, tk, sk, vk, dk = jax.random.split(key, 6)
        ids = jax.random.randint(
            ik, (count,), 0, self.training_bank.num_instances
        )
        bank = self.select_bank(ids, source=self.training_bank)
        distribution = (
            self.config["training"].get("command_distribution")
            or self.settings["command_distribution"]
        )
        speed_range = distribution.get(
            "speed_range_mps",
            self.settings["command_distribution"]["speed_range_mps"],
        )
        speeds = jax.random.uniform(
            sk, (count,), minval=speed_range[0], maxval=speed_range[1]
        )
        if distribution.get("kind") == "position":
            from drone_playground.environments.randomization import sample_command

            goals = jax.vmap(
                lambda goal, rng: sample_command(goal, rng, distribution)
            )(bank.goal, jax.random.split(tk, count))
            bank = bank.replace(goal=goals)
        from drone_playground.environments.tasks.navigation.initialization import (
            sample_initial_state,
        )

        reset = self.config["training"].get("reset_randomization") or {}
        strata = (jnp.arange(count) + 0.5) / count
        keys = jax.random.split(pk, count)
        if (reset.get("position") or {}).get("stratified", False):
            pos, velocity, clocks = jax.vmap(
                lambda i, rng, speed, category: sample_initial_state(
                    bank, i, rng, self.body_radius, speed, reset, category
                )
            )(jnp.arange(count), keys, speeds, strata)
        else:
            pos, velocity, clocks = jax.vmap(
                lambda i, rng, speed: sample_initial_state(
                    bank, i, rng, self.body_radius, speed, reset
                )
            )(jnp.arange(count), keys, speeds)
        state = self.model.randomize(
            PointMassState.create(pos).replace(
                vel=velocity, measurement_key=jax.random.split(vk, count)
            ),
            dk,
        )
        rotation = acceleration_attitude(state.acc, state.vel, state.rotation)
        from drone_playground.environments.randomization import reset_point_mass_state

        state = reset_point_mass_state(
            state.replace(rotation=rotation),
            tk,
            reset,
            position_and_velocity=False,
        )
        return bank, state, clocks, speeds, self.delays(dk, count)

    def advance_checked(
        self, bank, state, command, previous, ticks, timestamp, outcome
    ):
        """Use the training integration clock, with exact first-event freezing."""
        initial_time = timestamp

        def step(carry, tick):
            physical, clock, result, minimum = carry
            active = result == 0
            due = jnp.where((tick < ticks)[:, None], previous, command)
            candidate = self.model.step(physical, due, self.physics_dt)
            now = initial_time + (tick + 1) * self.physics_dt
            finite = jnp.all(jnp.isfinite(candidate.vector()), axis=-1)
            clearance = self.clearance(bank, candidate, now)
            collision = clearance < 0
            _, _, _, ended = self.task_definition.events(
                bank, candidate.pos, bank.goal, collision, ~finite
            )
            return (
                _select(active & finite, candidate, physical),
                jnp.where(active, now, clock),
                jnp.where(active, ended, result),
                jnp.where(
                    active & finite, jnp.minimum(minimum, clearance), minimum
                ),
            ), None

        end, _ = jax.lax.scan(
            step,
            (state, timestamp, outcome, jnp.full_like(timestamp, jnp.inf)),
            jnp.arange(self.substeps),
        )
        physical, clock, result, minimum = end
        return (
            physical,
            clock,
            result,
            jnp.where(jnp.isfinite(minimum), minimum, 0.0),
        )
