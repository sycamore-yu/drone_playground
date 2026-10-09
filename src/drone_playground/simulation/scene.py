"""Compiled MJCF geometry with pure JAX distance and first-return queries.

Lengths are metres, time is seconds. MuJoCo resolves defaults, includes and body
transforms once at construction; JAX evaluates the asset's prescribed motion.
Collision queries use contact-enabled geoms, rays use visible geoms (including
visual-only geometry), matching MuJoCo's distinction between contact and vision.
"""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from importlib import resources
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from jax import Array

_GEOMS = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
}
_NAVIGATION = ("S01", "S02", "S03", "D01", "D02", "D03", "S06", "D06")


def _norm(x: Array) -> Array:
    """Euclidean norm with a finite zero subgradient."""
    square = jnp.sum(x * x, axis=-1)
    return jnp.where(square > 0, jnp.sqrt(jnp.maximum(square, 1e-30)), 0.0)


def _distance(kind: str, p: Array, size: Array) -> Array:
    if kind == "plane":
        return p[..., 2]
    if kind == "sphere":
        return _norm(p) - size[..., 0]
    if kind == "capsule":
        p = p.at[..., 2].add(-jnp.clip(p[..., 2], -size[..., 1], size[..., 1]))
        return _norm(p) - size[..., 0]
    if kind == "cylinder":
        q = jnp.stack(
            (_norm(p[..., :2]) - size[..., 0], jnp.abs(p[..., 2]) - size[..., 1]), axis=-1
        )
    else:
        q = jnp.abs(p) - size
    return _norm(jnp.maximum(q, 0)) + jnp.minimum(jnp.max(q, axis=-1), 0)


def _roots(p: Array, d: Array, radius: Array) -> tuple[Array, Array]:
    a = jnp.sum(d * d, axis=-1)
    b = jnp.sum(p * d, axis=-1)
    c = jnp.sum(p * p, axis=-1) - radius * radius
    discriminant = b * b - a * c
    root = jnp.sqrt(jnp.maximum(discriminant, 1e-30))
    denominator = jnp.where(a > 0, a, 1.0)
    valid = (a > 0) & (discriminant >= 0)
    return (
        jnp.where(valid, (-b - root) / denominator, jnp.inf),
        jnp.where(valid, (-b + root) / denominator, jnp.inf),
    )


def _positive(x: Array) -> Array:
    return jnp.where(x >= 0, x, jnp.inf)


def _intersection(kind: str, p: Array, d: Array, size: Array) -> Array:
    """First nonnegative surface intersection, also for origins inside solids."""
    if kind == "plane":
        denominator = jnp.where(d[..., 2] != 0, d[..., 2], 1.0)
        root = -p[..., 2] / denominator
        xy = p[..., :2] + root[..., None] * d[..., :2]
        visible = (d[..., 2] < 0) & jnp.all(
            (size[..., :2] <= 0) | (jnp.abs(xy) <= size[..., :2]), axis=-1
        )
        return jnp.where(visible, _positive(root), jnp.inf)
    if kind == "box":
        parallel = d == 0
        denominator = jnp.where(parallel, 1.0, d)
        a, b = (-size - p) / denominator, (size - p) / denominator
        near = jnp.max(jnp.where(parallel, -jnp.inf, jnp.minimum(a, b)), axis=-1)
        far = jnp.min(jnp.where(parallel, jnp.inf, jnp.maximum(a, b)), axis=-1)
        valid = (near <= far) & ~jnp.any(parallel & (jnp.abs(p) > size), axis=-1)
        return jnp.where(valid, _positive(jnp.where(near >= 0, near, far)), jnp.inf)
    if kind == "sphere":
        a, b = _roots(p, d, size[..., 0])
        return jnp.minimum(_positive(a), _positive(b))

    radius, half = size[..., 0], size[..., 1]
    a, b = _roots(p[..., :2], d[..., :2], radius)
    result = jnp.full_like(a, jnp.inf)
    for root in (a, b):
        z = p[..., 2] + jnp.where(jnp.isfinite(root), root, 0) * d[..., 2]
        result = jnp.minimum(result, jnp.where(jnp.abs(z) <= half, _positive(root), jnp.inf))
    for sign in (-1, 1):
        if kind == "capsule":
            center = p.at[..., 2].add(-sign * half)
            for root in _roots(center, d, radius):
                z = p[..., 2] + jnp.where(jnp.isfinite(root), root, 0) * d[..., 2]
                result = jnp.minimum(result, jnp.where(sign * z >= half, _positive(root), jnp.inf))
        else:
            denominator = jnp.where(d[..., 2] != 0, d[..., 2], 1.0)
            root = (sign * half - p[..., 2]) / denominator
            xy = p[..., :2] + root[..., None] * d[..., :2]
            valid = (d[..., 2] != 0) & (jnp.sum(xy * xy, axis=-1) <= radius * radius)
            result = jnp.minimum(result, jnp.where(valid, _positive(root), jnp.inf))
    return result


class Scene:
    """One fixed scene shared by independently timed worlds.

    Args:
        name: ``empty``, a Navigation8 ID, ``racing``, or a custom MJCF path.

    ``model`` is the authoritative compiled MuJoCo model; do not mutate it after
    construction. Array order and ``geom_names`` follow its geom IDs. The geometry
    hash covers compiled shapes, transforms, contact/visibility and motion, not
    textures. Methods are JIT/autodiff compatible with this scene closed over.
    """

    def __init__(self, name: str | Path = "empty"):
        """Load the named geometry and its prescribed obstacle motion."""
        self.name = str(name)
        self._resources = ExitStack()
        self.xml_path: Path | None = None
        if self.name == "empty":
            self.model = mujoco.MjModel.from_xml_string('<mujoco model="empty"/>')
        else:
            if self.name in _NAVIGATION or self.name == "racing":
                root = resources.files("drone_playground").joinpath("assets/scenes")
                if root.is_dir():
                    root = self._resources.enter_context(resources.as_file(root))
                else:
                    root = Path(__file__).resolve().parents[3] / "assets/scenes"
                relative = (
                    f"navigation/{self.name}.xml"
                    if self.name in _NAVIGATION
                    else "racing/lsy_level0.xml"
                )
                self.xml_path = Path(root) / relative
            else:
                self.xml_path = Path(name)
                if not self.xml_path.is_file():
                    raise ValueError(
                        f"Unknown scene {name!r}; expected empty, racing or {_NAVIGATION}"
                    )
            self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        model = self.model
        if model.njnt:
            raise ValueError("Scene joints are unsupported; use prescribed mocap motion")
        self.geom_names = tuple(model.geom(i).name for i in range(model.ngeom))
        for i, kind in enumerate(model.geom_type):
            if int(kind) not in _GEOMS:
                raise ValueError(f"Unsupported geom {self.geom_names[i]!r}: {mujoco.mjtGeom(kind)}")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        motion = np.zeros((model.nbody, 6))
        if model.nuser_body:
            if model.nuser_body != 6:
                raise ValueError("body.user must contain motion type and five parameters")
            motion[:] = model.body_user
        if not np.isfinite(motion).all() or not np.isin(motion[:, 0], (0, 1, 2)).all():
            raise ValueError("Unsupported or nonfinite body motion")
        moving = motion[:, 0] != 0
        if np.any(moving & (model.body_mocapid < 0)):
            raise ValueError("Moving scene bodies must be mocap bodies")
        if np.any((motion[:, 0] == 1) & (motion[:, 5] <= 0)) or np.any(
            (motion[:, 0] == 2) & (motion[:, 4] <= 0)
        ):
            raise ValueError("Motion period/slower must be positive")
        # Descendants inherit their mocap ancestor's world-space translation.
        geom_motion = []
        for body in model.geom_bodyid:
            ancestor = int(body)
            while ancestor and motion[ancestor, 0] == 0:
                ancestor = int(model.body_parentid[ancestor])
            geom_motion.append(motion[ancestor])
        self._motion = jnp.asarray(np.asarray(geom_motion).reshape(-1, 6))
        self._base = jnp.asarray(data.geom_xpos.copy())
        self.rotations = jnp.asarray(data.geom_xmat.reshape(-1, 3, 3).copy())
        self.sizes = jnp.asarray(model.geom_size.copy())
        self._contact = jnp.asarray((model.geom_contype != 0) | (model.geom_conaffinity != 0))
        rgba = model.geom_rgba.copy()
        for i, material in enumerate(model.geom_matid):
            if material >= 0:
                rgba[i] = model.mat_rgba[material]
        self._visible = jnp.asarray(rgba[:, 3] > 0)
        self._groups = tuple(
            (kind, np.flatnonzero(model.geom_type == code))
            for code, kind in _GEOMS.items()
            if np.any(model.geom_type == code)
        )
        digest = hashlib.sha256(b"drone-playground-geometry-v1")
        digest.update(repr(self.geom_names).encode())
        for array in (
            model.geom_type,
            model.geom_size,
            data.geom_xpos,
            data.geom_xmat,
            model.geom_contype,
            model.geom_conaffinity,
            rgba[:, 3],
            np.asarray(geom_motion),
        ):
            digest.update(np.asarray(array, dtype="<f8").tobytes())
        self.geometry_hash = digest.hexdigest()
        self.geometry_identity = {
            "name": self.name,
            "sha256": self.geometry_hash,
            "geom_names": self.geom_names,
        }
        gate_ids = [i for i in range(model.nbody) if model.body(i).name.startswith("gate:")]
        self.gate_positions = jnp.asarray(data.xpos[gate_ids].copy())
        self.gate_rotations = jnp.asarray(data.xmat[gate_ids].reshape(-1, 3, 3).copy())
        self.gate_order = (
            tuple(int(x) for x in model.numeric("gate_order").data) if self.name == "racing" else ()
        )

    def positions(self, t: Array | float = 0.0) -> Array:
        """Return geom centers with shape ``t.shape + (ngeom, 3)`` at exact time."""
        t = jnp.asarray(t)[..., None]
        kind, sx, sy, sz, fourth, fifth = (self._motion[:, i] for i in range(6))
        u = 2 * t / jnp.where(kind == 1, fifth, 1) + fourth
        trefoil = jnp.stack(
            (
                sx / 6 * (jnp.sin(u) + 2 * jnp.sin(2 * u)),
                sy / 5 * (jnp.cos(u) - 2 * jnp.cos(2 * u)),
                -sz / 2 * jnp.sin(3 * u),
            ),
            axis=-1,
        )
        phase = t / jnp.where(kind == 2, fourth, 1) + fifth
        triangle = 2 * jnp.abs(2 * (phase - jnp.floor(phase + 0.5))) - 1
        bounce = triangle[..., None] * jnp.stack((sx, sy, sz), axis=-1)
        return self._base + jnp.where(
            (kind == 1)[..., None], trefoil, jnp.where((kind == 2)[..., None], bounce, 0)
        )

    def clearance(self, position: Array, t: Array | float = 0.0) -> Array:
        """Signed point clearance to contact geoms; subtract body radius externally.

        ``position`` is ``(..., 3)`` and ``t`` broadcasts to its leading shape.
        For multiple points per world pass ``t[..., None]``. Empty space is +inf.
        """
        position = jnp.asarray(position)
        t = jnp.broadcast_to(jnp.asarray(t), position.shape[:-1])
        centers = self.positions(t)
        result = jnp.full(position.shape[:-1], jnp.inf)
        for kind, ids in self._groups:
            local = jnp.einsum(
                "...gi,gij->...gj",
                position[..., None, :] - centers[..., ids, :],
                self.rotations[ids],
            )
            distances = _distance(kind, local, self.sizes[ids])
            result = jnp.minimum(
                result, jnp.min(jnp.where(self._contact[ids], distances, jnp.inf), axis=-1)
            )
        return result

    def raycast(
        self,
        origin: Array,
        directions: Array,
        t: Array | float = 0.0,
        max_range: Array | float = 40.0,
    ) -> Array:
        """Return metric ranges, capped at max_range on misses.

        Origins are ``(..., 3)``, directions ``(..., rays, 3)`` or one ``(3,)``
        direction, time is scalar or one value per world (origin's leading shape).
        Shared directions broadcast across worlds. Directions need not be unit
        length; zero directions miss. Work is chunked to bound ray/geom storage.
        A miss at exactly max_range is intentionally indistinguishable from a
        surface at that cutoff; sensor masks use the half-open range interval.
        """
        origin, directions = jnp.asarray(origin), jnp.asarray(directions)
        single_ray = directions.ndim == 1
        if single_ray:
            directions = directions[None, :]
        batch = jnp.broadcast_shapes(origin.shape[:-1], directions.shape[:-2])
        count = directions.shape[-2]
        origin = jnp.broadcast_to(origin, (*batch, 3))
        times = jnp.broadcast_to(jnp.asarray(t), batch)
        directions = jnp.broadcast_to(directions, (*batch, count, 3))
        origins = jnp.broadcast_to(origin[..., None, :], directions.shape).reshape(-1, 3)
        times = jnp.broadcast_to(times[..., None], (*batch, count)).reshape(-1)
        ranges = jnp.broadcast_to(jnp.asarray(max_range), batch)
        ranges = jnp.broadcast_to(ranges[..., None], (*batch, count)).reshape(-1)

        def cast(inputs):
            p, direction, time, limit = inputs
            length = _norm(direction)
            direction = direction / jnp.where(length > 0, length, 1)[..., None]
            centers = self.positions(time)
            result = limit
            for kind, ids in self._groups:
                local_p = jnp.einsum(
                    "...gi,gij->...gj", p[..., None, :] - centers[..., ids, :], self.rotations[ids]
                )
                local_d = jnp.einsum("...i,gij->...gj", direction, self.rotations[ids])
                hits = _intersection(kind, local_p, local_d, self.sizes[ids])
                result = jnp.minimum(
                    result, jnp.min(jnp.where(self._visible[ids], hits, jnp.inf), axis=-1)
                )
            return jnp.where(length > 0, result, limit)

        inputs = (origins, directions.reshape(-1, 3), times, ranges)
        if origins.shape[0] <= 8192:
            result = cast(inputs)
        else:
            result = jax.lax.map(jax.checkpoint(cast), inputs, batch_size=8192)
        result = result.reshape((*batch, count))
        return result[..., 0] if single_ray else result
