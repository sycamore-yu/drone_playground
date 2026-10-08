"""Acceleration-driven navigation with shared first-event and sensing kernels."""

import jax
import jax.numpy as jnp

from drone_playground.dynamics.point_mass import PointMassState, acceleration_attitude
from drone_playground.environments.scenes.geometry import clearance_and_collision
from drone_playground.environments.scenes.procedural_navigation import make_bank
from drone_playground.environments.tasks.point_mass import PointMassTask


def _select(mask, new, old):
    return jax.tree.map(
        lambda a, b: jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim)), a, b),
        new,
        old,
    )


class AccelerationNavigationTask(PointMassTask):
    """Use one navigation task with configurable sensing, reset and integration."""

    def bind(self, env):
        super().bind(env)
        if hasattr(self.scene, "build"):
            self.bank, self.manifest = make_bank(self.scene, env.reference_seed, env.count)
        else:
            self.bank = self.scene.sample(jax.random.PRNGKey(env.reference_seed), env.count)
            self.manifest = dict(source=self.scene.name, bank_digest=self.bank.digest())
        self.training_bank, self.training_manifest = self.bank, self.manifest
        env.bank, env.scene_manifest = self.bank, self.manifest

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
                self.conditions.get("reset_randomization") or {},
            )
        return self.dynamics.randomize(state, jax.random.fold_in(key, 2))

    def command(self, body_action, state):
        body_action = self.uncertain_action(body_action, state)
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

    def select_bank(self, indices, *, source=None):
        source = self.bank if source is None else source
        return source.select(indices)

    def clearance(self, bank, state, time):
        centre = state.pos + jnp.einsum("...ij,j->...i", state.rotation, jnp.array([0, 0, 0.005]))
        return jax.vmap(lambda i, p, t: self.events.clearance(bank, i, p, t)[0])(
            jnp.arange(bank.num_instances),
            centre,
            jnp.broadcast_to(time, (bank.num_instances,)),
        )

    def measure(self, bank, state, time, speeds):
        points, valid = jax.vmap(lambda i, p, r, t: self.sensor.sample(bank, i, p, r, t))(
            jnp.arange(bank.num_instances),
            state.pos,
            state.rotation,
            jnp.broadcast_to(time, (bank.num_instances,)),
        )
        from drone_playground.environments.randomization import noisy_point_mass_state

        measured = noisy_point_mass_state(state, time, self.dt, self.observation_noise)
        proprio, _ = self.observation.proprioception(measured, bank.goal, speeds, self.body_radius)
        # Corrupt the actor input without changing the ground-truth loss target.
        _, target = self.observation.proprioception(state, bank.goal, speeds, self.body_radius)
        points, valid = self.measurement(points, valid, state, time)
        return points, valid, proprio, target

    def delays(self, key, count):
        if self.conditions["action_delay_ms"] is None:
            return jnp.zeros(count, jnp.int32)
        low, high = self.conditions["action_delay_ms"]
        milliseconds = jax.random.uniform(key, (count,), minval=low, maxval=high)
        return jnp.ceil(milliseconds / (1000 * self.physics_dt) - 1e-6).astype(jnp.int32)

    def training_initial(self, key, count):
        ik, pk, tk, sk, vk, dk = jax.random.split(key, 6)
        ids = jax.random.randint(ik, (count,), 0, self.training_bank.num_instances)
        bank = self.select_bank(ids, source=self.training_bank)
        distribution = (
            self.conditions.get("command_distribution") or self.settings["command_distribution"]
        )
        speed_range = distribution.get(
            "speed_range_mps",
            self.settings["command_distribution"]["speed_range_mps"],
        )
        speeds = jax.random.uniform(sk, (count,), minval=speed_range[0], maxval=speed_range[1])
        if distribution.get("kind") == "position":
            from drone_playground.environments.randomization import sample_command

            goals = jax.vmap(lambda goal, rng: sample_command(goal, rng, distribution))(
                bank.goal, jax.random.split(tk, count)
            )
            bank = bank.replace(goal=goals)
        from drone_playground.environments.tasks.navigation.initialization import (
            sample_initial_state,
        )

        reset = self.conditions.get("reset_randomization") or {}
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
        state = self.dynamics.randomize(
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

    def transition_events(self, bank=None, *, arrival_sampling="physics"):
        """Bind navigation event semantics to the selected scene batch."""
        bank = self.bank if bank is None else bank

        def event(previous, candidate, time, memory):
            del previous
            finite = jnp.all(jnp.isfinite(candidate.vector()), axis=-1)
            clearance = self.clearance(bank, candidate, time)
            _, _, _, outcome = self.events.events(
                bank,
                candidate.pos,
                bank.goal,
                clearance < 0,
                ~finite,
            )
            if arrival_sampling != "physics":
                outcome = jnp.where(outcome == 1, 0, outcome)
            return finite, outcome, clearance, memory

        return event
