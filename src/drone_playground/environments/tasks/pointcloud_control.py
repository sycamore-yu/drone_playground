"""Qualified transfer of the paper's PointNet/GRU acceleration policy to control.

The reference-task adapter supplies desired velocity from a position reference.
The paper policy and lag dynamics are reused; the task objective, state sampling,
and 500Hz held-command integration are explicitly platform transfer conditions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate
from scipy.spatial.transform import Rotation

from drone_playground.environments.scenes.control_geometry import bank_from_environment
from drone_playground.environments.scenes.navigation import euclidean_norm, signed_distance
from drone_playground.models.point_mass import PointMassState


def delayed_step(model, state, command, previous, delay_ticks, physics_dt, substeps):
    """Apply the previous command until delivery, then the current command."""

    def one(current, tick):
        due = jnp.where((tick < delay_ticks)[..., None], previous, command)
        nxt = model.step(current, due, physics_dt)
        return nxt, nxt.pos

    return jax.lax.scan(one, state, jnp.arange(substeps))


class ReferencePointCloudTask:
    action_size = 3

    def __init__(self, config, device=None):
        from drone_playground.composition import (
            build_dynamics,
            build_environment,
            component_identity,
            compose_method,
        )

        self.config = config
        self.objective = config["objective"]
        self.physics_engine = "PointMassLag JAX at 500Hz; MuJoCo replay only"
        self.settings = config["env"]["task"]
        self.task = self.settings["control_task"]
        self.freq = int(self.settings["freq"])
        self.physics_freq = int(self.settings["physics_freq"])
        self.dt = 1 / self.settings["freq"]
        self.physics_dt = 1 / self.settings["physics_freq"]
        self.substeps = round(self.dt / self.physics_dt)
        self.duration = float(self.settings["duration"])
        self.episode_length = round(self.duration / self.dt)
        self.body_radius = float(self.settings["body_radius"])
        self.model = build_dynamics(config)
        self.sensor = instantiate(config["env"]["sensor"], _convert_="all")
        self.controller = instantiate(config["env"]["execution"]["controller"], _convert_="all")
        self.component_identity = component_identity(config)
        native_config = compose_method("learning/apg", self.task)
        device = device or config["runtime"]["device"]
        native = build_environment(native_config, device, "dev", 1)
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
            if self.task == "racing":
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
                if self.task != "racing"
                else jnp.array(native.config.env.track.safety_limits.pos_limit_low)
            )
            self.bounds_high = (
                jnp.array([4.0, 4.0, 4.0])
                if self.task != "racing"
                else jnp.array(native.config.env.track.safety_limits.pos_limit_high)
            )
        finally:
            native.close()
        self.bank = self.bank.replace(start=self.start[None])
        self.sensor_calibration = self.sensor.calibration()
        self.geometry_identity.update(
            task_adapter=self.settings["adapter_identity"],
            reference_hz=1 / self.reference_dt,
            policy_hz=1 / self.dt,
            physics_hz=1 / self.physics_dt,
            start=np.asarray(self.start).tolist(),
            integration=self.settings["integration"],
            floor_rule=self.settings["floor_rule"],
        )

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
        low, high = self.config["runtime"].get("action_delay_ms") or (0.0, 0.0)
        milliseconds = jax.random.uniform(key, (count,), minval=low, maxval=high)
        ticks = jnp.ceil(milliseconds / (1000 * self.physics_dt) - 1e-6).astype(jnp.int32)
        return ticks, milliseconds

    def initial(self, keys):
        jitter = (
            jax.vmap(lambda k: jax.random.normal(k, (3,)))(keys)
            * self.settings["initial_perturbation_m"]
        )
        position = (
            (self.start + jitter).at[:, 2].set(jnp.maximum(self.start[2] + jitter[:, 2], 0.005))
        )
        return PointMassState.create(position).replace(
            vel=jnp.broadcast_to(self.start_velocity, position.shape)
        )

    def training_initial(self, key, count, horizon):
        tk, pk, vk, dk = jax.random.split(key, 4)
        times = jax.random.uniform(
            tk, (count,), maxval=max(self.duration - horizon * self.dt, 0.001)
        )
        times = times.at[: count // 2].set(0.0)
        pos, vel = self.reference(times)
        pos = pos.at[: count // 2].set(self.start)
        vel = vel.at[: count // 2].set(self.start_velocity)
        pos = pos + jax.random.normal(pk, pos.shape) * self.settings["position_randomization_m"]
        pos = pos.at[:, 2].set(jnp.maximum(pos[:, 2], 0.005))
        vel = vel + jax.random.normal(vk, vel.shape) * self.settings["velocity_randomization_mps"]
        ticks, _ = self.delays(dk, count)
        return PointMassState.create(pos).replace(vel=vel), times, ticks

    def observation(self, state, timestamps):
        points, valid = jax.vmap(lambda p, r, t: self.sensor.sample(self.bank, 0, p, r, t))(
            state.pos, state.rotation, timestamps
        )
        reference, velocity = self.reference(timestamps)
        target = velocity + self.settings["reference_gain"] * (
            reference - jax.lax.stop_gradient(state.pos)
        )
        target = (
            target
            * jnp.minimum(
                1.0, self.settings["velocity_limit"] / jnp.maximum(euclidean_norm(target), 1e-6)
            )[..., None]
        )
        body_velocity = jnp.einsum("...ij,...i->...j", state.rotation, state.vel)
        body_target = jnp.einsum("...ij,...i->...j", state.rotation, target)
        radius = jnp.full((*state.pos.shape[:-1], 1), self.body_radius)
        proprio = jnp.concatenate(
            [body_velocity, body_target, state.rotation[..., 2, :], radius], axis=-1
        )
        return points, valid, proprio

    def command(self, action, state):
        return self.controller.physical_action(
            jnp.einsum("...ij,...j->...i", state.rotation, action)
        )

    def clearance(self, positions):
        centre = self.bank.origin[0]
        rotation = self.bank.rotations[0]

        def one(position):
            local = jnp.einsum("nji,nj->ni", rotation, position - centre)
            distance = signed_distance(
                self.bank.kind[0], self.bank.size[0], jnp.zeros_like(centre), local
            )
            obstacle = jnp.min(jnp.where(self.bank.active[0], distance, 1000.0)) - self.body_radius
            return jnp.minimum(obstacle, position[2])

        return jax.vmap(one)(positions)

    def close(self):
        pass

    def scenario(self, case):
        if case != 0:
            raise IndexError("Control transfer has one fixed canonical geometry")
        return {
            "task": self.task,
            "adapter": self.settings["adapter_identity"],
            "geometry": self.geometry_identity,
        }
