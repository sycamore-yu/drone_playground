"""Physical parameter sampling and external loads using native Crazyflow pipelines."""

from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow.sim.pipeline import append_fn, insert_fn_before
from crazyflow.utils import CORE_NDIM_KEY
from flax import struct
from jax.scipy.spatial.transform import Rotation


@struct.dataclass
class DisturbanceState:
    """Per-world random keys; forces follow simulation time rather than host calls."""

    key: jax.Array = struct.field(metadata={CORE_NDIM_KEY: 1})


def install_randomization(sim, parameters=None, disturbance=None):
    """Install supported parameter ranges and world-force/body-torque disturbances."""
    parameters, disturbance = dict(parameters or {}), dict(disturbance or {})
    if set(parameters) - {"mass", "inertia", "motor_strength", "drag"}:
        raise ValueError("Unknown physical randomization parameter")
    names = {f.name for f in fields(sim.data.params)}
    targets = {"mass": ("mass",), "inertia": ("J", "J_inv"), "drag": ("drag_matrix",)}
    targets["motor_strength"] = (
        ("rpm2thrust", "rpm2torque") if "rpm2thrust" in names else ("cmd_f_coef",)
    )
    ranges = {}
    for name, value in parameters.items():
        if value is None:
            continue
        bounds = np.asarray(value, float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or not 0 < bounds[0] <= bounds[1]:
            raise ValueError(f"{name} requires positive ordered scale bounds")
        if not set(targets[name]) <= names:
            raise ValueError(f"{name} is not effective for {sim.dynamics}")
        ranges[name] = tuple(bounds)
    allowed = {"force_world_n", "torque_body_nm", "gust_std_n", "gust_period_s"}
    if set(disturbance) - allowed:
        raise ValueError("Unknown disturbance parameter")
    force = np.asarray(disturbance.get("force_world_n", [0, 0, 0]), np.float32)
    torque = np.asarray(disturbance.get("torque_body_nm", [0, 0, 0]), np.float32)
    std = np.broadcast_to(np.asarray(disturbance.get("gust_std_n", 0), np.float32), (3,))
    period = float(disturbance.get("gust_period_s", 0.2))
    if force.shape != (3,) or torque.shape != (3,) or not np.isfinite([force, torque, std]).all():
        raise ValueError("External loads must be finite three-vectors")
    if np.any(std < 0) or not np.isfinite(period) or period <= 0:
        raise ValueError("Gust scale must be nonnegative and period positive")
    if not ranges and not disturbance:
        return
    nominal = sim.data.params
    changes = {}
    for name in ranges:
        for field in targets[name]:
            value = getattr(nominal, field)
            changes[field] = jnp.broadcast_to(value, (sim.n_worlds, sim.n_drones, *value.shape))
    if changes:
        sim.data = sim.data.replace(params=nominal.replace(**changes))
    if disturbance:
        sim.data = sim.data.replace(
            plugins={
                **sim.data.plugins,
                "disturbance": DisturbanceState(
                    jax.random.split(jax.random.PRNGKey(0), sim.n_worlds)
                ),
            }
        )
    sim.default_data = sim.data

    def randomize(data, default, mask=None):
        keys = jax.random.split(default.core.rng_key, len(ranges) + 1)
        params = data.params
        active = jnp.ones(sim.n_worlds, bool) if mask is None else mask
        for index, (name, bounds) in enumerate(ranges.items()):
            factor = jax.random.uniform(
                keys[index], (sim.n_worlds, 1), minval=bounds[0], maxval=bounds[1]
            )
            updates = {}
            for field in targets[name]:
                base = getattr(nominal, field)
                scale = factor.reshape((*factor.shape, *((1,) * base.ndim)))
                value = base / scale if field == "J_inv" else base * scale
                old = getattr(params, field)
                updates[field] = jnp.where(
                    active.reshape((len(active),) + (1,) * (old.ndim - 1)), value, old
                )
            params = params.replace(**updates)
        data = data.replace(params=params)
        if disturbance:
            state = data.plugins["disturbance"]
            new_keys = jax.random.key_data(jax.random.split(keys[-1], sim.n_worlds))
            state = state.replace(key=jnp.where(active[:, None], new_keys, state.key))
            data = data.replace(plugins={**data.plugins, "disturbance": state})
        return data

    append_fn(sim.reset_pipeline, randomize, "physical_randomization")
    if disturbance:

        def external_loads(data):
            time = data.core.steps[:, 0] / data.core.freq
            interval = jnp.floor(time / period).astype(jnp.int32)
            keys = jax.vmap(jax.random.fold_in)(data.plugins["disturbance"].key, interval)
            gust = jax.vmap(lambda key: jax.random.normal(key, (3,)))(keys) * std
            world_torque = Rotation.from_quat(data.states.quat[:, 0]).apply(
                jnp.broadcast_to(jnp.asarray(torque), (sim.n_worlds, 3))
            )
            return data.replace(
                states=data.states.replace(
                    force=(jnp.asarray(force) + gust)[:, None], torque=world_torque[:, None]
                )
            )

        insert_fn_before(sim.step_pipeline, "integration", external_loads, "external_loads")
