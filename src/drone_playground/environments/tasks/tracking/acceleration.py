"""Acceleration-based tracking with the original reference and loss semantics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from drone_playground.dynamics.point_mass import PointMassState
from drone_playground.environments.scenes.geometry import euclidean_norm, signed_distance
from drone_playground.environments.scenes.mujoco_geometry import bank_from_environment
from drone_playground.environments.tasks.point_mass import PointMassTask


class AccelerationTrackingTask(PointMassTask):
    """Retain point-cloud transfer kernels without selecting networks or learners."""

    def bind(self, env):
        from drone_playground.environments.environment import build_reference_environment

        super().bind(env)
        native = build_reference_environment(
            self.name, env.device, scene=env.scene, reference=env.reference
        )
        try:
            self.bank, self.geometry_identity = bank_from_environment(native)
            physical = native.reset(jax.random.PRNGKey(0)).pipeline_state.sim_data.states
            self.start = jnp.asarray(physical.pos[0, 0])
            self.start_velocity = jnp.asarray(physical.vel[0, 0])
            self.reference_dt = native.dt
            self.references = jnp.asarray(native.trajectories[0])
            self.reference_velocity = jnp.asarray(
                np.gradient(np.asarray(self.references), self.reference_dt, axis=0)
            )
            if self.name == "racing":
                track = native.config.env.track
                self.gate_positions = jnp.asarray([g["pos"] for g in track.gates])
                self.gate_quaternions = jnp.asarray(
                    Rotation.from_euler(
                        "xyz", np.asarray([g["rpy"] for g in track.gates])
                    ).as_quat()
                )
                self.gate_order = jnp.asarray(np.abs(track.gate_order) - 1)
                self.gate_reverse = jnp.asarray(np.array(track.gate_order) < 0)
            self.bounds_low = (
                jnp.array([-4.0, -4.0, 0.0])
                if self.name != "racing"
                else jnp.array(native.config.env.track.safety_limits.pos_limit_low)
            )
            self.bounds_high = (
                jnp.array([4.0, 4.0, 4.0])
                if self.name != "racing"
                else jnp.array(native.config.env.track.safety_limits.pos_limit_high)
            )
        finally:
            native.close()
        self.bank = self.bank.replace(start=self.start[None])
        self.sensor_calibration = self.sensor.calibration()
        self.geometry_identity.update(
            task_adapter=self.settings["provenance"],
            reference_hz=1 / self.reference_dt,
            policy_hz=1 / self.dt,
            physics_hz=1 / self.physics_dt,
            start=np.asarray(self.start).tolist(),
            integration=self.settings["integration"],
            floor_rule=self.settings["floor_rule"],
        )

        env.bank, env.geometry_identity = self.bank, self.geometry_identity

    def reference(self, timestamp):
        raw = jnp.clip(timestamp / self.reference_dt, 0, len(self.references) - 1)
        index = jnp.floor(raw).astype(jnp.int32)
        fraction = (raw - index)[..., None]
        next_index = jnp.minimum(index + 1, len(self.references) - 1)
        pos = (1 - fraction) * self.references[index] + fraction * self.references[next_index]
        vel = (1 - fraction) * self.reference_velocity[index] + fraction * self.reference_velocity[
            next_index
        ]
        return pos, vel

    def delays(self, key, count):
        low, high = self.conditions.get("action_delay_ms") or (0.0, 0.0)
        milliseconds = jax.random.uniform(key, (count,), minval=low, maxval=high)
        ticks = jnp.ceil(milliseconds / (1000 * self.physics_dt) - 1e-6).astype(jnp.int32)
        return ticks, milliseconds

    def initial(self, keys):
        initial = self.conditions.get("initial_conditions") or {}
        jitter = jax.vmap(lambda k: jax.random.normal(k, (3,)))(keys) * initial.get(
            "position_std_m", 0.0
        )
        position = (
            (self.start + jitter).at[:, 2].set(jnp.maximum(self.start[2] + jitter[:, 2], 0.005))
        )
        return self.dynamics.randomize(
            PointMassState.create(position).replace(
                vel=jnp.broadcast_to(self.start_velocity, position.shape),
                measurement_key=keys,
            ),
            keys[0],
        )

    def training_initial(self, key, count, horizon):
        tk, pk, vk, dk = jax.random.split(key, 4)
        reset = self.conditions.get("reset_randomization") or {}
        phase = reset.get("scene_phase_s", [0.0, 0.0])
        times = jax.random.uniform(
            tk,
            (count,),
            minval=phase[0],
            maxval=min(phase[1], max(self.duration - horizon * self.dt, 0.0)),
        )
        times = times.at[: count // 2].set(0.0)
        pos, vel = self.reference(times)
        pos = pos.at[: count // 2].set(self.start)
        vel = vel.at[: count // 2].set(self.start_velocity)
        ticks, _ = self.delays(dk, count)
        from drone_playground.environments.randomization import reset_point_mass_state

        state = reset_point_mass_state(
            PointMassState.create(pos).replace(
                vel=vel, measurement_key=jax.random.split(pk, count)
            ),
            pk,
            reset,
        )
        state = self.dynamics.randomize(
            state.replace(pos=state.pos.at[:, 2].set(jnp.maximum(state.pos[:, 2], 0.005))),
            vk,
        )
        return state, times, ticks

    def measure(self, state, timestamps):
        points, valid = jax.vmap(lambda p, r, t: self.sensor.sample(self.bank, 0, p, r, t))(
            state.pos, state.rotation, timestamps
        )
        from drone_playground.environments.randomization import noisy_point_mass_state

        measured = noisy_point_mass_state(state, timestamps, self.dt, self.observation_noise)
        reference, velocity = self.reference(timestamps)
        target = velocity + self.settings["reference_gain"] * (
            reference - jax.lax.stop_gradient(measured.pos)
        )
        target = (
            target
            * jnp.minimum(
                1.0,
                self.settings["velocity_limit"] / jnp.maximum(euclidean_norm(target), 1e-6),
            )[..., None]
        )
        body_velocity = jnp.einsum("...ij,...i->...j", measured.rotation, measured.vel)
        body_target = jnp.einsum("...ij,...i->...j", measured.rotation, target)
        radius = jnp.full((*state.pos.shape[:-1], 1), self.body_radius)
        proprio = jnp.concatenate(
            [body_velocity, body_target, measured.rotation[..., 2, :], radius],
            axis=-1,
        )
        points, valid = self.measurement(points, valid, state, timestamps)
        return points, valid, proprio

    def command(self, action, state):
        action = self.uncertain_action(action, state)
        return self.controller.physical_action(
            jnp.einsum("...ij,...j->...i", state.rotation, action)
        )

    def clearance(self, positions):
        centre = self.bank.origin[0]
        rotation = self.bank.rotations[0]

        def one(position):
            local = jnp.einsum("nji,nj->ni", rotation, position - centre)
            distance = signed_distance(
                self.bank.kind[0],
                self.bank.size[0],
                jnp.zeros_like(centre),
                local,
            )
            obstacle = jnp.min(jnp.where(self.bank.active[0], distance, 1000.0)) - self.body_radius
            return jnp.minimum(obstacle, position[2])

        return jax.vmap(one)(positions)

    def scenario(self, case):
        if case != 0:
            raise IndexError("Control transfer has one fixed canonical geometry")
        return {
            "task": self.name,
            "adapter": self.settings["provenance"],
            "geometry": self.geometry_identity,
        }
