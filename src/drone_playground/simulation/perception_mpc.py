"""JAX/Crazyflow predictive-sampling MPC with explicit landmark image costs.

Inspired by Falanga et al., IROS 2018 (PAMPC), and by DeepMind's Predictive
Sampling MJPC planner. This is a new sampling-based controller, not a port of
the RPG ACADO solver, which uses a different trajectory/transcription method.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import load_params
from crazyflow.sim import Sim

from drone_playground.simulation.methods import Setpoint


def landmark_image_coordinates(position, quaternion_xyzw, landmark):
    """Project a known world landmark onto the quadrotor's forward-x camera.

    Returns normalized image xy and positive optical depth. The camera is
    centered at the body origin; a sensor-specific extrinsic can be introduced
    without changing the full-state optimization contract.
    """
    world = jnp.asarray(landmark) - jnp.asarray(position)
    q = jnp.asarray(quaternion_xyzw)
    vec, scalar = q[..., :3], q[..., 3:4]
    cross = jnp.cross(vec, world)
    body = world - 2 * scalar * cross + 2 * jnp.cross(vec, cross)
    depth = body[..., 0]
    pixel = body[..., 1:] / jnp.maximum(depth[..., None], 0.2)
    return pixel, depth


def perception_cost(pixel, depth, previous_pixel, *, horizontal_tangent, vertical_tangent):
    """Penalize viewing outside the FOV and high projected landmark velocity."""
    limits = jnp.asarray([horizontal_tangent, vertical_tangent])
    outside = jnp.maximum(jnp.abs(pixel) - limits, 0)
    visibility = jnp.sum(outside**2, axis=-1) + 5.0 * (depth < 0.2)
    image_motion = jnp.sum((pixel - previous_pixel) ** 2, axis=-1)
    return visibility, image_motion


class PerceptionAwareSamplingMPC:
    """A known-landmark perception-aware MPC using JAX predictive sampling.

    The controller receives the same Crazyflow Task trajectory as other native
    trackers and predicts actual Crazyflow rotor-drag state dynamics. For Racing,
    it also receives *known* ordered gate-center landmarks, not sensor-estimated
    points; consequently it is not a vision-only comparison against learned RL.
    """

    execution = "host"
    output_level = "attitude_thrust"

    def __init__(
        self,
        drone="cf21B_500",
        frequency=50.0,
        samples=256,
        horizon=12,
        prediction_seconds=0.48,
        device="cpu",
        yaw=0.0,
        visibility_weight=90.0,
        image_velocity_weight=7.0,
        horizontal_fov_deg=87.0,
        vertical_fov_deg=58.0,
    ):
        """Build a JAX predictor, retaining complete command and solver state."""
        if (
            type(samples) is not int
            or type(horizon) is not int
            or samples < 2
            or horizon < 2
            or not np.isfinite(
                [frequency, prediction_seconds, visibility_weight, image_velocity_weight]
            ).all()
            or min(frequency, prediction_seconds) <= 0
            or min(visibility_weight, image_velocity_weight) < 0
            or not 1 < horizontal_fov_deg < 179
            or not 1 < vertical_fov_deg < 179
        ):
            raise ValueError("Invalid perception-aware predictive sampling parameters")
        self.horizon = horizon
        self.samples = samples
        self.frequency = float(frequency)
        self.yaw = float(yaw)
        self.prediction_dt = float(prediction_seconds / horizon)
        prediction_hz = round(1 / self.prediction_dt)
        if not np.isclose(self.prediction_dt * prediction_hz, 1.0):
            raise ValueError("Prediction dt must admit an integer simulation frequency")
        self.reference_offsets = (np.arange(horizon) + 1) * self.prediction_dt
        self.sim = Sim(
            n_worlds=samples,
            drone=drone,
            dynamics="so_rpy_rotor_drag",
            control="attitude",
            freq=prediction_hz,
            attitude_freq=prediction_hz,
            device=device,
        )
        base, step = self.sim.data, self.sim.build_step_fn()
        hardware = load_params("first_principles", drone)
        self.hover = float(np.asarray(hardware["mass"]).item() * 9.81)
        self.thrust_time = 1 / float(np.asarray(base.params.thrust_dyn_coef).reshape(-1)[0])
        self.neutral = jnp.asarray([0.0, 0.0, self.yaw, self.hover])
        max_thrust = 4 * float(np.asarray(hardware["thrust_max"]).item())
        bounds = (
            jnp.asarray([-0.6, -0.6, -np.pi, 0.0]),
            jnp.asarray([0.6, 0.6, np.pi, max_thrust]),
        )
        spread = jnp.asarray([0.10, 0.10, 0.18, 0.07])
        hfov = np.tan(np.deg2rad(horizontal_fov_deg / 2))
        vfov = np.tan(np.deg2rad(vertical_fov_deg / 2))
        self.gates = np.empty((0, 3))
        self.gate_index = 0

        def optimize(body, memory, targets, target_velocities, landmark):
            rng, sample_rng = jax.random.split(memory["rng"])
            candidate = (
                jnp.clip(
                    memory["command"][None]
                    + jax.random.normal(sample_rng, (samples, horizon, 4)) * spread,
                    *bounds,
                )
                .at[0]
                .set(memory["command"])
            )
            state = base.states.replace(
                pos=jnp.broadcast_to(body.pos[0], base.states.pos.shape),
                quat=jnp.broadcast_to(body.quat[0], base.states.quat.shape),
                vel=jnp.broadcast_to(body.vel[0], base.states.vel.shape),
                ang_vel=jnp.broadcast_to(body.ang_vel[0], base.states.ang_vel.shape),
                rotor_vel=jnp.full_like(base.states.rotor_vel, memory["thrust"]),
            )
            previous, _ = landmark_image_coordinates(state.pos[:, 0], state.quat[:, 0], landmark)

            def advance(carry, stage):
                simulation, pixel_old = carry
                command, target, target_velocity = stage
                simulation = simulation.replace(
                    controls=simulation.controls.replace(
                        attitude=simulation.controls.attitude.replace(staged_cmd=command[:, None])
                    )
                )
                simulation = step(simulation, 1)
                pos, vel = simulation.states.pos[:, 0], simulation.states.vel[:, 0]
                pixel, depth = landmark_image_coordinates(
                    pos, simulation.states.quat[:, 0], landmark
                )
                outside, image_motion = perception_cost(
                    pixel,
                    depth,
                    pixel_old,
                    horizontal_tangent=hfov,
                    vertical_tangent=vfov,
                )
                cost = (
                    50.0 * jnp.sum((pos - target) ** 2, axis=-1)
                    + jnp.sum((vel - target_velocity) ** 2, axis=-1)
                    + 5.0 * jnp.sum(command[:, :2] ** 2, axis=-1)
                    + 5.0 * (command[:, 3] - self.hover) ** 2
                    + 0.5 * command[:, 2] ** 2
                    + visibility_weight * outside
                    + image_velocity_weight * image_motion
                )
                return (simulation, pixel), cost

            _, costs = jax.lax.scan(
                advance,
                (base.replace(states=state), previous),
                (candidate.transpose(1, 0, 2), targets, target_velocities),
            )
            best = jnp.argsort(costs.sum(axis=0))[: max(1, samples // 100)]
            updated = candidate[best].mean(axis=0)
            samples_t = jnp.arange(horizon) * self.prediction_dt
            shifted = jax.vmap(
                lambda commands, rest: jnp.interp(
                    samples_t + 1 / frequency, samples_t, commands, right=rest
                ),
                in_axes=(1, 0),
                out_axes=1,
            )(updated, self.neutral)
            thrust = (
                memory["thrust"] + (updated[0, 3] - memory["thrust"]) / self.thrust_time / frequency
            )
            return updated[0], {"rng": rng, "command": shifted, "thrust": thrust}

        self._optimize = jax.jit(optimize)

    def set_course(self, task):
        """Bind known gate-center landmarks in the published Racing order."""
        if task.name != "racing":
            raise ValueError("Known-landmark perception-aware MPC currently requires Racing")
        self.gates = np.asarray(task.gate_positions)[np.asarray(task.gate_order)]
        self.gate_index = 0

    def initialize_memory(self, batch, seed=0):
        """Reset sampling state and the landmark selection for each episode."""
        if batch != 1:
            raise ValueError("Native predictive-sampling MPC executes one world")
        if not len(self.gates):
            raise ValueError("Racing gate landmarks must be bound before initialization")
        self.gate_index = 0
        return {
            "rng": jax.random.PRNGKey(seed),
            "command": jnp.tile(self.neutral, (self.horizon, 1)),
            "thrust": jnp.asarray(self.hover),
        }

    def __call__(self, physics, reference, memory):
        """Optimize tracking and landmark visibility over the same predicted flight."""
        positions, velocities, _ = reference
        if positions.shape != (1, self.horizon, 3):
            raise ValueError("Perception MPC requires a complete supplied reference horizon")
        current = np.asarray(physics.states.pos[0, 0])
        while (
            self.gate_index < len(self.gates) - 1
            and np.linalg.norm(current - self.gates[self.gate_index]) < 0.27
        ):
            self.gate_index += 1
        command, memory = self._optimize(
            physics.states,
            memory,
            positions[0],
            velocities[0],
            jnp.asarray(self.gates[self.gate_index]),
        )
        if not np.isfinite(np.asarray(command)).all():
            raise FloatingPointError("PAMPC predictive sampling produced a nonfinite command")
        return Setpoint(command[None]), memory

    def close(self):
        """Release native JAX predictive simulator resources."""
        self.sim.close()
