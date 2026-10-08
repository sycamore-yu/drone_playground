"""Crazyflow's elite-mean sampling MPC, with an explicit task reference input.

Algorithm reference: learnsyslab/crazyflow examples/control/sampling.py at
36f584d114d9d331f0cee0fe4b9066f821c0fbfd (MIT). Candidate distribution, costs,
elite arithmetic mean, shifted warm start and thrust observer follow that source.
The task supplies its reference and pole positions instead of the demo's grid.
"""

from __future__ import annotations

import time

import crazyflow  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.dynamics import load_params
from crazyflow.sim import Sim


class SamplingMPC:
    def __init__(
        self,
        *,
        drone,
        reference,
        frequency=50,
        samples=2000,
        horizon=25,
        prediction_seconds=1.0,
        device="cpu",
        obstacles=None,
        seed=0,
    ):
        if min(samples, horizon, frequency) < 1:
            raise ValueError("Sample, horizon and frequency counts must be positive")
        self.samples, self.horizon, self.frequency = samples, horizon, frequency
        self.prediction_device = jax.devices(device)[0]
        self.predict_dt = prediction_seconds / horizon
        predicted_frequency = round(1 / self.predict_dt)
        if abs(predicted_frequency * self.predict_dt - 1) > 1e-6:
            raise ValueError("Prediction step must be an integer simulation frequency")
        self.sim = Sim(
            n_worlds=samples,
            drone=drone,
            dynamics="so_rpy_rotor_drag",
            control="attitude",
            freq=predicted_frequency,
            attitude_freq=predicted_frequency,
            device=device,
        )
        self.sim.reset()
        base = self.sim.data
        step_fn = self.sim.build_step_fn()
        hardware = load_params("first_principles", drone)
        self.hover = float(np.asarray(hardware["mass"]).item() * 9.81)
        self.thrust_time = float(np.asarray(base.params.thrust_time_coef).reshape(-1)[0])
        self.thrust_estimate = self.hover
        self.hover_cmd = jnp.array([0.0, 0.0, 0.0, self.hover], jnp.float32)
        self.mean = jnp.broadcast_to(self.hover_cmd, (horizon, 4))
        self.key = jax.random.key(seed)
        angle = np.deg2rad(60)
        low = jnp.array([-angle, -angle, 0.0, 0.0], jnp.float32)
        high = jnp.array(
            [
                angle,
                angle,
                0.0,
                4 * float(np.asarray(hardware["thrust_max"]).item()),
            ],
            jnp.float32,
        )
        sigma = jnp.array([0.1, 0.1, 0.0, 0.08], jnp.float32)
        reference = np.asarray(reference, dtype=np.float32)
        if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 2:
            raise ValueError("Reference must be [T,3] with at least two samples")
        self.refs = reference
        self.refs_vel = np.gradient(reference, 1 / frequency, axis=0)
        self.ref_times = np.arange(len(reference)) / frequency
        poles = jnp.asarray(
            np.zeros((0, 3)) if obstacles is None else obstacles,
            dtype=jnp.float32,
        )
        self.last_diagnostics = {}
        self.last_prediction = None

        def update(obs, key, mean, goals, velocities, yaw):
            key, sample_key = jax.random.split(key)
            candidates = jnp.clip(
                mean[None] + jax.random.normal(sample_key, (samples, horizon, 4)) * sigma,
                low,
                high,
            )
            candidates = candidates.at[0].set(mean)
            candidates = candidates.at[:, :, 2].set(yaw)
            states = base.states.replace(
                pos=base.states.pos.at[...].set(obs["pos"]),
                quat=base.states.quat.at[...].set(obs["quat"]),
                vel=base.states.vel.at[...].set(obs["vel"]),
                ang_vel=base.states.ang_vel.at[...].set(obs["ang_vel"]),
                rotor_vel=base.states.rotor_vel.at[...].set(obs["collective_thrust"]),
            )
            data = base.replace(states=states)

            def predict(data, row):
                command, goal, velocity, target_yaw = row
                data = data.replace(
                    controls=data.controls.replace(
                        attitude=data.controls.attitude.replace(staged_cmd=command[:, None])
                    )
                )
                nxt = step_fn(data, 1)
                p, v = nxt.states.pos[:, 0], nxt.states.vel[:, 0]
                cmd = nxt.controls.attitude.staged_cmd[:, 0]
                cost = 50 * jnp.sum((p - goal) ** 2, axis=-1) + jnp.sum(
                    (v - velocity) ** 2, axis=-1
                )
                cost += (
                    5 * jnp.sum(cmd[:, :2] ** 2, axis=-1)
                    + 5 * (cmd[:, 3] - self.hover) ** 2
                    + 100 * (cmd[:, 2] - target_yaw) ** 2
                )
                pole_distance = jnp.linalg.norm(p[:, None, :2] - poles[None, :, :2], axis=-1)
                cost += 1000 * jnp.sum(pole_distance < (0.055 + 0.12 + 0.02), axis=-1)
                return nxt, (cost, p)

            _, (cost, positions) = jax.lax.scan(
                predict,
                data,
                (candidates.transpose(1, 0, 2), goals, velocities, yaw),
            )
            cost = jnp.sum(cost, axis=0)
            elite = jnp.argsort(cost)[: max(1, int(samples * 0.01))]
            updated = jnp.mean(candidates[elite], axis=0)
            times = jnp.arange(horizon) * self.predict_dt
            shifted = times + 1 / frequency
            next_mean = jax.vmap(
                lambda control, hover: jnp.interp(shifted, times, control, right=hover),
                in_axes=(1, 0),
                out_axes=1,
            )(updated, self.hover_cmd)
            return (
                updated[0],
                key,
                next_mean,
                positions[:, elite[0]],
                jnp.min(cost),
            )

        self._update = jax.jit(update)

    def compute_from_data(self, data, tick):
        state = data.states
        obs = {name: getattr(state, name)[0, 0] for name in ("pos", "quat", "vel", "ang_vel")}
        return self.compute_control(obs, tick)

    def compute_control(self, observation, tick, *, trajectory=None, yaw=None):
        """Consume the public body-state view, retaining private thrust estimation."""
        obs = {name: jnp.asarray(observation[name]) for name in ("pos", "quat", "vel", "ang_vel")}
        obs["collective_thrust"] = jnp.float32(self.thrust_estimate)
        offsets = (np.arange(self.horizon) + 1) * self.predict_dt
        if trajectory is None:
            stamps = tick / self.frequency + offsets
            goals = np.column_stack([np.interp(stamps, self.ref_times, x) for x in self.refs.T])
            velocities = np.column_stack(
                [np.interp(stamps, self.ref_times, x) for x in self.refs_vel.T]
            )
            yaws = np.zeros(self.horizon)
        else:
            from drone_playground.references import reference_horizon

            reference = reference_horizon(trajectory, tick / self.frequency, offsets, yaw=yaw)
            goals, velocities, yaws = (reference[name] for name in ("position", "velocity", "yaw"))
        tic = time.perf_counter()
        inputs = jax.device_put(
            (
                obs,
                self.key,
                self.mean,
                jnp.asarray(goals, dtype=jnp.float32),
                jnp.asarray(velocities, dtype=jnp.float32),
                jnp.asarray(yaws, dtype=jnp.float32),
            ),
            self.prediction_device,
        )
        with jax.default_device(self.prediction_device):
            action, self.key, self.mean, prediction, cost = self._update(*inputs)
        action = np.asarray(action)
        self.last_prediction = np.asarray(prediction)
        elapsed = time.perf_counter() - tic
        self.thrust_estimate += (
            (float(action[3]) - self.thrust_estimate) / self.thrust_time / self.frequency
        )
        self.last_diagnostics = dict(
            decision_seconds=elapsed,
            samples=self.samples,
            horizon=self.horizon,
            cost=float(cost),
            prediction_model="so_rpy_rotor_drag",
            prediction_device=str(self.prediction_device),
            elite_fraction=0.01,
            method="elite-arithmetic-mean",
            status=0 if np.isfinite(action).all() else -1,
        )
        if not np.isfinite(action).all():
            raise FloatingPointError("Sampling MPC produced a non-finite action")
        return action

    def reset(self, seed=0):
        self.key = jax.random.key(seed)
        self.mean = jnp.broadcast_to(self.hover_cmd, (self.horizon, 4))
        self.thrust_estimate = self.hover

    def step(self, observation, tick):
        """Return one physical action while retaining this solver's warm start."""
        return self.compute_control(observation, tick)

    def after_step(self, command, observation, reward, done):
        """The sampling controller has no separate reference-clock callback."""
        del command, observation, reward, done
        return False

    def close(self):
        self.sim.close()
