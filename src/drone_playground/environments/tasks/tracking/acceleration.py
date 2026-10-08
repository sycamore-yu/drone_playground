"""Acceleration-based tracking with the original reference and loss semantics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from drone_playground.dynamics.point_mass import PointMassState
from drone_playground.environments.scenes.geometry import signed_distance
from drone_playground.environments.scenes.mujoco_geometry import bank_from_model
from drone_playground.environments.tasks.point_mass import PointMassTask
from drone_playground.numerics import euclidean_norm


class AccelerationTrackingTask(PointMassTask):
    """Retain point-cloud transfer kernels without selecting networks or learners."""

    def bind(self, env):
        import mujoco

        from drone_playground.configuration import load_config
        from drone_playground.environments.scenes.racing import load_lsy_config
        from drone_playground.resources import resource_path

        super().bind(env)
        canonical = load_config("environment", ["env=" + self.name])
        reference_config = canonical["env"]
        reference_freq = reference_config["freq"]
        reference_duration = reference_config["task"]["duration"]
        order, bounds = (), None
        if self.name == "racing":
            model = mujoco.MjModel.from_xml_path(
                str(resource_path("assets/scenes/racing/lsy_level0.xml"))
            )
            track = load_lsy_config().env.track
            self.start = jnp.asarray(track.drones[0].pos, jnp.float32)
            self.start_velocity = jnp.asarray(track.drones[0].vel, jnp.float32)
            self.gate_positions = jnp.asarray([g["pos"] for g in track.gates])
            self.gate_quaternions = jnp.asarray(
                Rotation.from_euler("xyz", np.asarray([g["rpy"] for g in track.gates])).as_quat()
            )
            self.gate_order = jnp.asarray(np.abs(track.gate_order) - 1)
            self.gate_reverse = jnp.asarray(np.array(track.gate_order) < 0)
            self.bounds_low = jnp.asarray(track.safety_limits.pos_limit_low)
            self.bounds_high = jnp.asarray(track.safety_limits.pos_limit_high)
            bounds = (self.bounds_low, self.bounds_high)
            order = track.gate_order
        else:
            model = mujoco.MjModel.from_xml_string("<mujoco/>")
            self.start = jnp.asarray(env.scene.takeoff, jnp.float32)
            self.start_velocity = jnp.zeros(3)
            if env.reference.name == "figure8":
                # Preserve the pinned FigureEightEnv reset at key=0. The public
                # TrackingTask splits once before its upstream reset pipeline.
                key = jax.random.split(jax.random.key(0))[0]
                _, pos_key, vel_key = jax.random.split(key, 3)
                self.start = jax.random.uniform(
                    pos_key,
                    (1, 1, 3),
                    minval=jnp.array([-0.1, -0.1, 1.1]),
                    maxval=jnp.array([0.1, 0.1, 1.3]),
                )[0, 0]
                self.start_velocity = jax.random.uniform(
                    vel_key, (1, 1, 3), minval=-0.5, maxval=0.5
                )[0, 0]
            self.bounds_low, self.bounds_high = (
                jnp.array([-4.0, -4.0, 0.0]),
                jnp.array([4.0, 4.0, 4.0]),
            )
        self.reference_dt = 1.0 / reference_freq
        references = env.reference.build(
            canonical["runtime"]["scene_seed_eval"],
            1,
            reference_duration,
            reference_freq,
            self.start,
        )
        self.references = jnp.asarray(references[0], jnp.float32)
        self.reference_velocity = jnp.asarray(
            np.gradient(np.asarray(self.references), self.reference_dt, axis=0)
        )
        self.bank, self.geometry_identity = bank_from_model(
            model,
            task=self.name,
            start=self.start,
            goal=self.references[-1],
            bounds=bounds,
            gate_order=order,
        )
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

    def transition_events(self, bank=None):
        """Supply native gate/contact rules without owning a physics loop."""
        del bank
        from drone_playground.environments.tasks.tracking.events import tracking_events

        return lambda old, new, time, passed: tracking_events(self, old, new, time, passed)
