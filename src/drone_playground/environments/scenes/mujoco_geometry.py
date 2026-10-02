"""Read the actual control-course geometry into the shared analytic sensor bank.

The canonical MuJoCo course supplies primitive sizes and world rotations. Robot
and visualization-only geoms are excluded. This is a sensing representation;
the original course retains ownership of rigid-body contacts and gate events.
"""

from __future__ import annotations

import jax.numpy as jnp
import mujoco
import numpy as np

from drone_playground.environments.scenes.geometry import (
    KIND_BOX,
    KIND_CAPSULE,
    KIND_CYLINDER,
    KIND_SPHERE,
    SceneBank,
)


def bank_from_environment(env):
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    source = getattr(env.sim, "mjx_data", None)
    if source is not None:
        for field in ("mocap_pos", "mocap_quat", "qpos"):
            destination = getattr(data, field)
            value = np.asarray(getattr(source, field))
            if value.size == destination.size:
                destination[...] = value.reshape(destination.shape)
            elif value.size and destination.size:
                destination[...] = value.reshape((-1, *destination.shape))[0]
    mujoco.mj_forward(model, data)
    ids, names, kinds, sizes = [], [], [], []
    for index in range(model.ngeom):
        body = int(model.geom_bodyid[index])
        ancestors = []
        while body:
            ancestors.append(model.body(body).name or "")
            body = int(model.body_parentid[body])
        if any(
            "drone" in name.lower() or "prop" in name.lower()
            for name in ancestors
        ):
            continue
        kind = model.geom_type[index]
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if not (model.geom_contype[index] or model.geom_conaffinity[index]):
            continue
        size = np.asarray(model.geom_size[index], dtype=np.float32)
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            kinds.append(KIND_BOX)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            kinds.append(KIND_CYLINDER)
            size = np.array([size[0], 2 * size[1], 0], np.float32)
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            kinds.append(KIND_CAPSULE)
            size = np.array([size[0], 2 * size[1], 0], np.float32)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            kinds.append(KIND_SPHERE)
        else:
            raise ValueError(
                f"Unsupported physical course geometry {model.geom(index).name}: {kind}"
            )
        ids.append(index)
        names.append(
            model.geom(index).name or "/".join(ancestors) + f"/geom-{index}"
        )
        sizes.append(size)
    capacity = max(1, len(ids))
    start = np.asarray(env.sim.default_data.states.pos[0, 0], np.float32)
    goal = np.asarray(env.trajectories)[0, -1]
    kind = np.zeros(capacity, np.int32)
    size = np.ones((capacity, 3), np.float32)
    origin = np.zeros((capacity, 3), np.float32)
    rotation = np.broadcast_to(np.eye(3), (capacity, 3, 3)).copy()
    active = np.zeros(capacity, bool)
    if ids:
        kind[: len(ids)], size[: len(ids)], origin[: len(ids)] = (
            kinds,
            sizes,
            data.geom_xpos[ids],
        )
        rotation[: len(ids)] = data.geom_xmat[ids].reshape(-1, 3, 3)
        active[: len(ids)] = True
    if env.task == "racing":
        limits = env.config.env.track.safety_limits
        lower = np.array(limits.pos_limit_low, np.float32)
        lower[2] = max(0.0, lower[2])
        upper = np.array(limits.pos_limit_high, np.float32)
        order = list(env.config.env.track.gate_order)
    else:
        lower, upper, order = (
            np.array([-10.0, -10.0, 0.0]),
            np.array([10.0, 10.0, 10.0]),
            [],
        )
    bank = SceneBank(
        jnp.array(kind[None]),
        jnp.array(size[None]),
        jnp.array(origin[None]),
        jnp.zeros((1, capacity), jnp.int32),
        jnp.zeros((1, capacity, 5)),
        jnp.array(active[None]),
        jnp.array(start[None]),
        jnp.array(goal[None]),
        jnp.zeros(1, jnp.int32),
        jnp.zeros(1, jnp.int32),
        jnp.array(lower),
        jnp.array(upper),
        (env.task,),
        jnp.array(rotation[None], jnp.float32),
    )
    return bank, dict(
        source="actual MuJoCo task geometry",
        geometry_names=names,
        geometry_ids=ids,
        gate_order=order,
        bank_digest=bank.digest(),
        geometry_rotations="world-from-local",
        task=env.task,
    )
