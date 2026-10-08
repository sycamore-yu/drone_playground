"""Reset distributions, measurement noise, action uncertainty and runtime forces.

These are independent of dynamics parameter randomization and the learner's
exploration distribution. All randomness is episode state, never a host RNG.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Wrapper
from jax.scipy.spatial.transform import Rotation


def validate_effects(settings):
    """Validate and normalize configured environment randomization effects."""
    allowed = {
        "observation_noise": {
            "position_std_m",
            "orientation_std_rad",
            "velocity_std_mps",
            "angular_velocity_std_radps",
            "sensor_std_m",
            "sensor_dropout_probability",
        },
        "action_noise": {
            "std_normalized",
            "bias_normalized",
            "std_physical",
            "bias_physical",
        },
        "disturbance": {
            "force_world_n",
            "torque_body_nm",
            "gust_std_n",
            "force_uniform_half_width_n",
            "gust_period_s",
            "acceleration_world_mps2",
            "gust_std_mps2",
        },
        "reset_randomization": {
            "position_half_width_m",
            "position_std_m",
            "orientation_half_width_rad",
            "velocity_std_mps",
            "angular_velocity_std_radps",
            "scene_phase_s",
            "position",
        },
    }
    for group, fields in allowed.items():
        spec = settings.get(group) or {}
        if not isinstance(spec, dict) or set(spec) - fields:
            raise ValueError(
                f"Unsupported {group} fields: "
                f"{set(spec) - fields if isinstance(spec, dict) else spec}"
            )
        for name, value in spec.items():
            if name == "position":
                if not isinstance(value, dict) or value.get("distribution", "mixture") != "mixture":
                    raise ValueError(
                        "Navigation initial positions use a declared collision-safe mixture "
                        "distribution"
                    )
                if set(value) - {
                    "distribution",
                    "mixture_weights",
                    "goal_region_width_m",
                    "candidates",
                    "minimum_clearance_m",
                    "stratified",
                }:
                    raise ValueError("Unknown initial-position distribution field")
                weights = np.asarray(value.get("mixture_weights", [1, 0, 0]), dtype=float)
                if (
                    weights.shape != (3,)
                    or not np.isfinite(weights).all()
                    or (weights < 0).any()
                    or not np.isclose(weights.sum(), 1)
                ):
                    raise ValueError(
                        "Initial-position mixture needs three nonnegative weights summing to one"
                    )
                if (
                    not isinstance(value.get("candidates", 32), int)
                    or value.get("candidates", 32) < 1
                ):
                    raise ValueError("Initial-position candidates must be a positive integer")
                for field in ("goal_region_width_m", "minimum_clearance_m"):
                    numbers = np.asarray(value.get(field, 0), dtype=float)
                    if not np.isfinite(numbers).all() or (numbers < 0).any():
                        raise ValueError(f"Initial-position {field} must be finite and nonnegative")
                continue
            values = np.asarray(value, dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{group}.{name} must be finite")
            if name not in (
                "scene_phase_s",
                "gust_period_s",
            ) and values.shape not in ((), (3,), (4,)):
                raise ValueError(
                    f"{group}.{name} needs a scalar or vector matching its physical quantity"
                )
            if group != "action_noise" and values.shape == (4,):
                raise ValueError(f"{group}.{name} needs a scalar or three spatial axes")
            if (
                name
                in (
                    "sensor_std_m",
                    "sensor_dropout_probability",
                    "gust_period_s",
                )
                and values.shape != ()
            ):
                raise ValueError(f"{group}.{name} must be scalar")
            if (
                any(term in name for term in ("std", "half_width", "probability"))
                and (values < 0).any()
            ):
                raise ValueError(f"{group}.{name} must be nonnegative")
            if name == "sensor_dropout_probability" and (values > 1).any():
                raise ValueError("Sensor dropout probability must lie in [0, 1]")
            if name == "gust_period_s" and float(value) <= 0:
                raise ValueError("Gust period must be positive")
            if name == "scene_phase_s" and (
                values.shape != (2,) or values[0] < 0 or values[1] < values[0]
            ):
                raise ValueError("Scene phase needs ordered nonnegative seconds")


def point_measurement_noise(points, valid, key, standard_deviation_m=0.0, dropout_probability=0.0):
    """Apply reproducible noise and dropout to point measurements."""
    if not math.isfinite(standard_deviation_m) or standard_deviation_m < 0:
        raise ValueError("Measurement standard deviation must be finite and nonnegative")
    if not 0 <= dropout_probability <= 1:
        raise ValueError("Measurement dropout probability must lie in [0, 1]")
    noise_key, dropout_key = jax.random.split(key)
    if standard_deviation_m:
        points = points + jnp.where(
            valid[..., None],
            jax.random.normal(noise_key, points.shape) * standard_deviation_m,
            0.0,
        )
    if dropout_probability:
        valid = valid & (jax.random.uniform(dropout_key, valid.shape) >= dropout_probability)
    return jnp.where(valid[..., None], points, 0.0), valid


def noisy_physical_state(states, key, settings):
    """Create a measurement view; never change the physical state or reward."""
    keys = jax.random.split(key, 4)
    changes = {}
    for index, (field, name) in enumerate(
        (
            ("pos", "position_std_m"),
            ("vel", "velocity_std_mps"),
            ("ang_vel", "angular_velocity_std_radps"),
        )
    ):
        std = settings.get(name, 0.0)
        if np.any(np.asarray(std)):
            changes[field] = getattr(states, field) + jax.random.normal(
                keys[index], getattr(states, field).shape
            ) * jnp.asarray(std)
    std = settings.get("orientation_std_rad", 0.0)
    if np.any(np.asarray(std)):
        error = Rotation.from_rotvec(jax.random.normal(keys[3], (3,)) * jnp.asarray(std))
        quat = (Rotation.from_quat(states.quat[0, 0]) * error).as_quat()
        changes["quat"] = quat[None, None]
    return states.replace(**changes)


def noisy_point_mass_state(state, time, dt, settings):
    """State-estimation measurements for acceleration-driven environments."""
    if np.any(settings.get("angular_velocity_std_radps", 0)):
        raise ValueError("Point-mass observation has no measured angular velocity")
    indices = jnp.broadcast_to(
        jnp.floor(jnp.asarray(time) / dt).astype(jnp.int32),
        state.pos.shape[:-1],
    )
    keys = jax.vmap(jax.random.fold_in)(state.measurement_key, indices)
    changes = {}
    for index, (field, name) in enumerate((("pos", "position_std_m"), ("vel", "velocity_std_mps"))):
        std = settings.get(name, 0.0)
        if np.any(np.asarray(std)):
            draw = jax.vmap(
                lambda key, index=index: jax.random.normal(
                    jax.random.fold_in(key, index + 10), (3,)
                )
            )(keys)
            changes[field] = getattr(state, field) + draw * jnp.asarray(std)
    std = settings.get("orientation_std_rad", 0.0)
    if np.any(np.asarray(std)):
        angles = jax.vmap(lambda key: jax.random.normal(jax.random.fold_in(key, 12), (3,)))(
            keys
        ) * jnp.asarray(std)
        changes["rotation"] = jax.vmap(
            lambda matrix, angle: (
                Rotation.from_matrix(matrix) * Rotation.from_rotvec(angle)
            ).as_matrix()
        )(state.rotation, angles)
    return state.replace(**changes)


def reset_point_mass_state(state, key, settings, position_and_velocity=True):
    """Sample a randomized initial state for point-mass motion."""
    keys = jax.random.split(key, 3)
    changes = {}
    if position_and_velocity:
        if "position_std_m" in settings:
            jitter = jax.random.normal(keys[0], state.pos.shape) * jnp.asarray(
                settings["position_std_m"]
            )
        else:
            jitter = jax.random.uniform(
                keys[0], state.pos.shape, minval=-1, maxval=1
            ) * jnp.asarray(settings.get("position_half_width_m", 0))
        changes["pos"] = state.pos + jitter
        changes["vel"] = state.vel + jax.random.normal(keys[1], state.vel.shape) * jnp.asarray(
            settings.get("velocity_std_mps", 0)
        )
    angles = jax.random.uniform(keys[2], state.pos.shape, minval=-1, maxval=1) * jnp.asarray(
        settings.get("orientation_half_width_rad", 0)
    )
    changes["rotation"] = jax.vmap(
        lambda matrix, angle: (
            Rotation.from_matrix(matrix) * Rotation.from_euler("xyz", angle)
        ).as_matrix()
    )(state.rotation, angles)
    return state.replace(**changes)


def validate_command_distribution(spec):
    """Validate the commanded motion distribution for a task."""
    if not isinstance(spec, dict) or set(spec) - {
        "kind",
        "distribution",
        "value",
        "low",
        "high",
        "speed_range_mps",
    }:
        raise ValueError("Unknown command distribution field")
    if spec.get("kind") not in (
        "position",
        "velocity",
        "reference",
        "goal_velocity",
    ):
        raise ValueError("Command kind must be position, velocity, reference or goal_velocity")
    distribution = spec.get("distribution")
    if distribution not in ("fixed", "uniform", "catalog", "reference_bank"):
        raise ValueError("Unknown command distribution")
    if spec["kind"] == "goal_velocity":
        bounds = np.asarray(spec.get("speed_range_mps"), dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 <= bounds[0] <= bounds[1]:
            raise ValueError("Goal velocity needs an ordered nonnegative speed_range_mps")
    elif distribution == "uniform":
        low, high = (
            np.asarray(spec.get("low"), dtype=float),
            np.asarray(spec.get("high"), dtype=float),
        )
        if (
            low.shape != (3,)
            or high.shape != (3,)
            or not np.isfinite([low, high]).all()
            or np.any(high < low)
        ):
            raise ValueError("Uniform commands need three finite ordered low/high bounds")
    elif "value" in spec:
        value = np.asarray(spec["value"], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError("Fixed command value needs three finite axes")


def sample_command(default, key, spec):
    """Position goals, velocity commands and references share distribution terms."""
    if not spec or spec.get("distribution", "fixed") in (
        "fixed",
        "reference_bank",
        "catalog",
    ):
        return jnp.asarray(spec.get("value", default) if spec else default)
    if spec["distribution"] != "uniform":
        raise ValueError("Command distribution must be fixed, uniform, catalog or reference_bank")
    low, high = np.asarray(spec["low"]), np.asarray(spec["high"])
    if low.shape != high.shape or not np.isfinite([low, high]).all() or np.any(high < low):
        raise ValueError("Command bounds must be finite, ordered and have equal shapes")
    return jax.random.uniform(key, low.shape, minval=jnp.asarray(low), maxval=jnp.asarray(high))


class EnvironmentEffects(Wrapper):
    """The same task with selected training or frozen evaluation conditions."""

    def __init__(self, env, settings):
        super().__init__(env)
        validate_effects(settings)
        self.settings = settings
        if "position" in (settings.get("reset_randomization") or {}):
            raise ValueError(
                "Collision-safe position mixtures belong to navigation; reference tasks "
                "use position_std_m or position_half_width_m"
            )
        if "scene_phase_s" in (settings.get("reset_randomization") or {}) and not hasattr(
            env.default, "reference_phase_ticks"
        ):
            raise ValueError("This task has no moving-scene or reference phase")
        for name, value in (settings.get("action_noise") or {}).items():
            if np.asarray(value).shape not in ((), (env.action_size,)):
                raise ValueError(
                    f"Action uncertainty {name} must match action size {env.action_size}"
                )
        if set(settings.get("disturbance") or {}) & {
            "acceleration_world_mps2",
            "gust_std_mps2",
        }:
            raise ValueError(
                "Rigid-body disturbance uses force in N and torque in Nm; acceleration "
                "disturbances require a point-mass model"
            )
        if env.dynamics.forward == "lotf_simplified" and np.any(
            (settings.get("disturbance") or {}).get("torque_body_nm", 0)
        ):
            raise ValueError("Simplified body-rate dynamics has no external torque response")
        self.reset_info_fields = (
            *getattr(env, "reset_info_fields", ()),
            "effects_key",
            "disturbance_key",
        )
        if not hasattr(env.default, "sim_data") and not hasattr(env.default, "states"):
            raise ValueError("Selected task does not expose the physical-state effects interface")

    def _measure(self, state):
        noise = self.settings.get("observation_noise") or {}
        if not noise:
            return state
        data = state.pipeline_state
        key, next_key = jax.random.split(state.info["effects_key"])
        measured = noisy_physical_state(data.sim_data.states, key, noise)
        view = data.replace(sim_data=data.sim_data.replace(states=measured))
        return state.replace(
            obs=self.env.observation(view),
            info={**state.info, "effects_key": next_key},
        )

    def reset(self, rng, *args, **kwargs):
        environment_key, effects_key, reset_key = jax.random.split(rng, 3)
        state = self.env.reset(environment_key, *args, **kwargs)
        reset = self.settings.get("reset_randomization") or {}
        if reset:
            data = state.pipeline_state
            states = data.sim_data.states
            keys = jax.random.split(reset_key, 4)
            updates = {}
            for index, (field, name, normal) in enumerate(
                (
                    ("pos", "position_half_width_m", False),
                    ("vel", "velocity_std_mps", True),
                    ("ang_vel", "angular_velocity_std_radps", True),
                )
            ):
                width = reset.get(name, 0.0)
                if field == "pos" and "position_std_m" in reset:
                    width, normal = reset["position_std_m"], True
                if np.any(np.asarray(width)):
                    draw = (
                        jax.random.normal(keys[index], getattr(states, field).shape)
                        if normal
                        else jax.random.uniform(
                            keys[index],
                            getattr(states, field).shape,
                            minval=-1,
                            maxval=1,
                        )
                    )
                    updates[field] = getattr(states, field) + draw * jnp.asarray(width)
            width = reset.get("orientation_half_width_rad", 0.0)
            if np.any(np.asarray(width)):
                angle = jax.random.uniform(keys[3], (3,), minval=-1, maxval=1) * jnp.asarray(width)
                updates["quat"] = (
                    Rotation.from_quat(states.quat[0, 0]) * Rotation.from_euler("xyz", angle)
                ).as_quat()[None, None]
            data = data.replace(sim_data=data.sim_data.replace(states=states.replace(**updates)))
            if "scene_phase_s" in reset:
                if not hasattr(data, "reference_phase_ticks"):
                    raise ValueError("This task has no moving-scene or reference phase")
                low, high = reset["scene_phase_s"]
                phase = jax.random.uniform(keys[3], (), minval=low, maxval=high)
                data = data.replace(
                    reference_phase_ticks=jnp.floor(phase / self.dt).astype(jnp.int32)
                )
            state = state.replace(pipeline_state=data, obs=self.env.observation(data))
        state = state.replace(
            info={
                **state.info,
                "effects_key": effects_key,
                "disturbance_key": reset_key,
            }
        )
        return self._measure(self._prepare(state))

    def _prepare(self, state):
        disturbance = self.settings.get("disturbance") or {}
        if not disturbance:
            return state
        data = state.pipeline_state
        sim = data.sim_data
        force = jnp.asarray(disturbance.get("force_world_n", [0, 0, 0]), jnp.float32)
        torque = jnp.asarray(disturbance.get("torque_body_nm", [0, 0, 0]), jnp.float32)
        sim = sim.replace(
            plugins={
                **sim.plugins,
                "external_force_world_n": force,
                "external_torque_body_nm": torque,
                "gust_std_n": jnp.asarray(disturbance.get("gust_std_n", 0.0)),
                "force_uniform_half_width_n": jnp.asarray(
                    disturbance.get("force_uniform_half_width_n", 0.0)
                ),
                "gust_period_s": jnp.asarray(disturbance.get("gust_period_s", 1.0)),
                "disturbance_key": state.info["disturbance_key"],
            }
        )
        return state.replace(pipeline_state=data.replace(sim_data=sim))

    def _perturb_action(self, state, action):
        key, next_key = jax.random.split(state.info["effects_key"])
        noise = self.settings.get("action_noise") or {}
        scale = (self.high - self.low) / 2
        action = jnp.clip(
            action
            + jnp.asarray(noise.get("bias_normalized", 0.0))
            + jnp.asarray(noise.get("bias_physical", 0.0)) / scale
            + jax.random.normal(key, (self.action_size,))
            * (
                jnp.asarray(noise.get("std_normalized", 0.0))
                + jnp.asarray(noise.get("std_physical", 0.0)) / scale
            ),
            -1,
            1,
        )
        state = state.replace(info={**state.info, "effects_key": next_key})
        return state, action

    def step(self, state, action):
        state, action = self._perturb_action(state, action)
        result = self.env.step(self._prepare(state), action)
        return self._measure(result)

    def step_schedule(self, state, commands):
        normalized = 2 * (commands - self.low) / (self.high - self.low) - 1
        state, normalized = self._perturb_action(state, normalized)
        commands = self.low + (normalized + 1) * (self.high - self.low) / 2
        return self._measure(self.env.step_schedule(self._prepare(state), commands))

    def step_physical(self, state, command):
        return self.step(state, 2 * (command - self.low) / (self.high - self.low) - 1)
