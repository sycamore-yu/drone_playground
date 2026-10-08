"""SANDO-style analytic navigation scenes: geometry, motion and instances.

Geometry and obstacle-motion semantics are reproduced from the pinned SANDO
reference (``research_dev/sando`` at ``3a4450dcc5a8ed5ca825c9da7966e2642be091de``,
BSD-3-Clause, read-only): the trefoil equations, the 0.65 dynamic fraction, the
0.8 m dynamic cube, the 1.0-1.5 m radius forest cylinder and the ``<= 0.5 m/s``
speed bound all come from ``scripts/paper_dynamic_scene.py``; the collision and
clearance rules follow ``scripts/integer_forest_geometry.py``; the start/goal
anchors and the extra-clearance constant follow ``scripts/integer_scene_protocol.py``.

The P5 corridor (20 x 10 x 5 m, 15 m navigation distance, 2 m start/goal height),
the per-difficulty obstacle counts and the ceiling-constrained vertical motion
amplitude are this project's own frozen recipe. Every departure from the source
is recorded in the instance manifest, so source identity and project recipe stay
separable.

Obstacles are analytic primitives. One description therefore serves training
geometry, reward clearance, collision and replay; display, sensor and collision
definitions cannot drift apart because there is only one of them.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from drone_playground.numerics import euclidean_norm, quat_to_matrix_xyzw

# --------------------------------------------------------------------------------------
# Obstacle taxonomy
# --------------------------------------------------------------------------------------

KIND_EMPTY = 0
KIND_CYLINDER = 1
KIND_BOX = 2
KIND_SPHERE = 3
KIND_CAPSULE = 4

MOTION_STATIC = 0
MOTION_TREFOIL = 1
MOTION_BOUNCE = 2

MOTION_NAMES = {
    MOTION_STATIC: "static",
    MOTION_TREFOIL: "trefoil",
    MOTION_BOUNCE: "linear_bounce",
}
KIND_NAMES = {
    KIND_EMPTY: "empty",
    KIND_CYLINDER: "cylinder",
    KIND_BOX: "box",
    KIND_SPHERE: "sphere",
    KIND_CAPSULE: "capsule",
}

# Instances are padded to this capacity so JAX shapes stay static.
MAX_OBSTACLES = 64
MOTION_PARAMS = 5

# --------------------------------------------------------------------------------------
# SANDO source constants (paper_dynamic_scene.py, integer_scene_protocol.py)
# --------------------------------------------------------------------------------------

SANDO_GLOBAL_TIME_SCALE = 7.9442501919
SANDO_DYNAMIC_FRACTION = 0.65
SANDO_CUBE_M = 0.8
SANDO_CYLINDER_RADIUS_M = (1.0, 1.5)
SANDO_CYLINDER_HEIGHT_M = 6.0
SANDO_TREFOIL_SCALES_M = (2.0, 4.0)
SANDO_TREFOIL_OFFSET_S = (0.0, 3.0)
SANDO_TREFOIL_SLOWER_RAW = (4.0, 6.0)
SANDO_TREFOIL_SPEED_BOUND_M_PER_S = 0.5
SANDO_EXTRA_CLEARANCE_M = 0.5
SANDO_ROBOT_RADIUS_M = 0.1
SANDO_NOMINAL_START_M = (0.0, 0.0, 2.0)
SANDO_MAX_PLACEMENT_ATTEMPTS = 10000

# Element size classes. ``column`` is SANDO's forest cylinder; the smaller
# classes are this project's mixed-obstacle elements, sized so a 20 x 10 m
# corridor can carry a meaningful density gradient.
COLUMN_RADIUS_M = SANDO_CYLINDER_RADIUS_M
PILLAR_RADIUS_M = (0.2, 0.35)
CUBE_HALF_M = (0.4, 0.9)
CROSSBAR_HALF_M = (0.12, 0.2)
CROSSER_HALF_M = 0.4

# Body collision sphere of the pinned Crazyflow model
# (crazyflow/drones/cf2x_L250.xml:67, name="col_sphere", size="0.07").
BODY_RADIUS_M = 0.07
BODY_SPHERE_OFFSET_M = (0.0, 0.0, 0.005)

DIFFICULTIES = ("easy", "medium", "hard")
STATIC_FAMILIES = ("cylinder_forest", "mixed")
DYNAMIC_FAMILIES = ("dynamic_forest", "crossers")
FAMILIES = STATIC_FAMILIES + DYNAMIC_FAMILIES


def trefoil_speed_bound(sx: float, sy: float, sz: float, slower: float) -> float:
    """SANDO's analytical upper bound on trefoil speed (paper_dynamic_scene.py:41)."""
    return 2.0 * math.sqrt((5 * sx / 6) ** 2 + sy**2 + (3 * sz / 2) ** 2) / slower


# --------------------------------------------------------------------------------------
# Instance description
# --------------------------------------------------------------------------------------


@dataclass
class Obstacle:
    """One analytic obstacle in the project world frame."""

    shape: str
    kind: int
    origin: tuple[float, float, float]
    """Static centre, or the trefoil/oscillation centre."""

    size: tuple[float, float, float]
    """Cylinder ``(radius, height, 0)``; box half extents ``(hx, hy, hz)``."""

    motion: int = MOTION_STATIC
    params: tuple[float, float, float, float, float] = (0.0,) * MOTION_PARAMS

    def half_extents(self) -> tuple[float, float, float]:
        if self.kind == KIND_CYLINDER:
            return (self.size[0], self.size[0], self.size[1] / 2.0)
        return (self.size[0], self.size[1], self.size[2])

    def position(self, time: float) -> tuple[float, float, float]:
        """Analytic obstacle centre at one simulation time (host-side reference)."""
        x0, y0, z0 = self.origin
        if self.motion == MOTION_STATIC:
            return (x0, y0, z0)
        if self.motion == MOTION_TREFOIL:
            sx, sy, sz, offset, slower = self.params
            tt = 2.0 * time / slower + offset
            return (
                sx / 6.0 * (math.sin(tt) + 2.0 * math.sin(2.0 * tt)) + x0,
                sy / 5.0 * (math.cos(tt) - 2.0 * math.cos(2.0 * tt)) + y0,
                -sz / 2.0 * math.sin(3.0 * tt) + z0,
            )
        if self.motion == MOTION_BOUNCE:
            ax, ay, az, period, phase = self.params
            u = time / period + phase
            tri = 2.0 * abs(2.0 * (u - math.floor(u + 0.5))) - 1.0
            return (x0 + ax * tri, y0 + ay * tri, z0 + az * tri)
        raise ValueError(f"Unknown motion code: {self.motion}")

    def excursion(self) -> tuple[float, float, float]:
        """Half-extent of the reachable centre displacement over all phases."""
        if self.motion == MOTION_STATIC:
            return (0.0, 0.0, 0.0)
        if self.motion == MOTION_TREFOIL:
            sx, sy, sz, _, _ = self.params
            # |sin(t)+2sin(2t)| <= 3, |cos(t)-2cos(2t)| <= 3, |sin(3t)| <= 1
            return (sx / 2.0, sy * 3.0 / 5.0, sz / 2.0)
        ax, ay, az, _, _ = self.params
        return (abs(ax), abs(ay), abs(az))

    def swept_aabb(self):
        """Axis-aligned centre bounds of the body over one full motion period."""
        dx, dy, dz = self.excursion()
        hx, hy, hz = self.half_extents()
        x0, y0, z0 = self.origin
        return (
            (x0 - dx - hx, y0 - dy - hy, z0 - dz - hz),
            (x0 + dx + hx, y0 + dy + hy, z0 + dz + hz),
        )

    def footprint(self, time: float = 0.0):
        """Axis-aligned body bounds at one instant."""
        hx, hy, hz = self.half_extents()
        x, y, z = self.position(time)
        return ((x - hx, y - hy, z - hz), (x + hx, y + hy, z + hz))


# --------------------------------------------------------------------------------------
# Frozen instance bank
# --------------------------------------------------------------------------------------


@struct.dataclass
class SceneBank:
    """Static per-instance scene description shared by every environment.

    The first axis indexes scene instances; ``scenario_id`` in the task state
    selects one. Instances are generated on the host once per role, which keeps
    JAX shapes static and makes train/eval isolation an explicit,
    inspectable list rather than an implicit draw.
    """

    kind: jax.Array
    size: jax.Array
    origin: jax.Array
    motion: jax.Array
    params: jax.Array
    active: jax.Array
    start: jax.Array
    goal: jax.Array
    difficulty: jax.Array
    subtype: jax.Array
    world_low: jax.Array
    world_high: jax.Array
    subtype_names: tuple = struct.field(pytree_node=False, default=())
    rotations: jax.Array | None = None
    asset_paths: tuple[str, ...] = struct.field(pytree_node=False, default=())

    @property
    def num_instances(self) -> int:
        return int(self.kind.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.kind.shape[1])

    def select(self, indices):
        names = (
            "kind",
            "size",
            "origin",
            "motion",
            "params",
            "active",
            "start",
            "goal",
            "difficulty",
            "subtype",
            "rotations",
        )
        return self.replace(
            **{
                name: getattr(self, name)[indices]
                for name in names
                if getattr(self, name) is not None
            }
        )

    def digest(self) -> str:
        """Content hash used to freeze scenes before benchmark evaluation."""
        digest = hashlib.sha256()
        for name in (
            "kind",
            "size",
            "origin",
            "motion",
            "params",
            "active",
            "start",
            "goal",
        ):
            array = np.asarray(getattr(self, name))
            digest.update(str((name, array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        if self.rotations is not None:
            array = np.asarray(self.rotations)
            digest.update(str(("rotations", array.shape, array.dtype)).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def labels(self, index: int) -> dict[str, str]:
        """Human-readable instance identity used in reports and manifests."""
        return {
            "difficulty": DIFFICULTIES[int(self.difficulty[index])],
            "subtype": str(self.subtype_names[int(self.subtype[index])]),
        }

    def describe(self, index: int) -> dict:
        """Describe the actual instance used by sensing, execution and replay."""
        return dict(
            scenario_id=int(index),
            **self.labels(index),
            obstacles=self.active_count(index),
            start=np.asarray(self.start[index]).tolist(),
            goal=np.asarray(self.goal[index]).tolist(),
        )

    def active_count(self, index: int) -> int:
        return int(np.asarray(self.active[index]).sum())


# --------------------------------------------------------------------------------------
# Analytic geometry evaluated on device
# --------------------------------------------------------------------------------------


def obstacle_positions(bank: SceneBank, scenario_id: jax.Array, time: jax.Array) -> jax.Array:
    """Obstacle centres ``[capacity, 3]`` for one instance at one simulation time."""
    origin = bank.origin[scenario_id]
    motion = bank.motion[scenario_id]
    params = bank.params[scenario_id]
    sx, sy, sz, offset, slower = (params[:, index] for index in range(MOTION_PARAMS))

    safe_slower = jnp.where(motion == MOTION_TREFOIL, slower, 1.0)
    tt = 2.0 * time / safe_slower + offset
    trefoil = jnp.stack(
        [
            sx / 6.0 * (jnp.sin(tt) + 2.0 * jnp.sin(2.0 * tt)) + origin[:, 0],
            sy / 5.0 * (jnp.cos(tt) - 2.0 * jnp.cos(2.0 * tt)) + origin[:, 1],
            -sz / 2.0 * jnp.sin(3.0 * tt) + origin[:, 2],
        ],
        axis=-1,
    )

    ax, ay, az, period, phase = (params[:, index] for index in range(MOTION_PARAMS))
    safe_period = jnp.where(motion == MOTION_BOUNCE, period, 1.0)
    u = time / safe_period + phase
    tri = 2.0 * jnp.abs(2.0 * (u - jnp.floor(u + 0.5))) - 1.0
    bounce = origin + jnp.stack([ax, ay, az], axis=-1) * tri[:, None]

    moving = jnp.where((motion == MOTION_TREFOIL)[:, None], trefoil, bounce)
    return jnp.where((motion == MOTION_STATIC)[:, None], origin, moving)


def signed_distance(kind: jax.Array, size: jax.Array, centre: jax.Array, point: jax.Array):
    """Exact signed distance from a point to an analytic primitive.

    Cylinder and box forms follow SANDO's ``signed_distance_point_cylinder``
    and ``signed_distance_point_box``. Negative inside, zero on the surface.
    """
    delta = point - centre
    radial = euclidean_norm(delta[..., :2]) - size[..., 0]
    vertical = jnp.abs(delta[..., 2]) - size[..., 1] / 2.0
    cylinder = euclidean_norm(
        jnp.stack([jnp.maximum(radial, 0.0), jnp.maximum(vertical, 0.0)], axis=-1)
    ) + jnp.minimum(jnp.maximum(radial, vertical), 0.0)

    q = jnp.abs(delta) - size
    box = euclidean_norm(jnp.maximum(q, 0.0)) + jnp.minimum(jnp.max(q, axis=-1), 0.0)
    sphere = euclidean_norm(delta) - size[..., 0]
    axis_delta = delta.at[..., 2].set(
        delta[..., 2] - jnp.clip(delta[..., 2], -size[..., 1] / 2, size[..., 1] / 2)
    )
    capsule = euclidean_norm(axis_delta) - size[..., 0]
    return jnp.where(
        kind == KIND_CAPSULE,
        capsule,
        jnp.where(
            kind == KIND_SPHERE,
            sphere,
            jnp.where(kind == KIND_CYLINDER, cylinder, box),
        ),
    )


def clearance_and_collision(
    bank: SceneBank,
    scenario_id: jax.Array,
    time: jax.Array,
    body_centre: jax.Array,
    body_radius: float = BODY_RADIUS_M,
):
    """Signed clearance to the nearest obstacle/ground and strict contact flag.

    Clearance is negative inside an obstacle, so a body centre inside geometry
    never clamps to a false positive. Contact is ``< radius`` for every
    primitive: SANDO's strict cylinder rule extended to the body sphere. The
    point-mass benchmark rule's box-inclusive boundary is recorded in the P5
    source inventory as the difference.
    """
    centre = obstacle_positions(bank, scenario_id, time)
    if bank.rotations is None:
        distance = signed_distance(
            bank.kind[scenario_id], bank.size[scenario_id], centre, body_centre
        )
    else:
        local = jnp.einsum("nji,nj->ni", bank.rotations[scenario_id], body_centre - centre)
        distance = signed_distance(
            bank.kind[scenario_id],
            bank.size[scenario_id],
            jnp.zeros_like(centre),
            local,
        )
    distance = jnp.where(bank.active[scenario_id], distance, jnp.inf)
    # The same ground plane is rendered and ray-cast at world_low[2]. Walls
    # and the upper limit are flight-volume boundaries, not physical planes.
    ground_distance = body_centre[2] - bank.world_low[2]
    clearance = jnp.minimum(jnp.min(distance), ground_distance)
    return clearance - body_radius, clearance < body_radius


def rotate_body_offset(quat: jax.Array, offset: jax.Array) -> jax.Array:
    """Rotate a fixed body-frame offset into world coordinates.

    Crazyflow stores drone orientation as an **xyzw** quaternion
    (``crazyflow/dynamics/symbols.py``); MuJoCo's ``mocap_quat`` is wxyz and the
    replay exporter converts when it writes the trace. Using the wrong layout
    here would silently rotate the collision sphere by 180 degrees.
    """
    return quat_to_matrix_xyzw(quat) @ offset


def body_centre_from_state(pos: jax.Array, quat: jax.Array) -> jax.Array:
    """Collision sphere centre in world coordinates from the body pose."""
    offset = jnp.asarray(BODY_SPHERE_OFFSET_M, jnp.float32)
    return pos + rotate_body_offset(quat, offset)


# --------------------------------------------------------------------------------------
# Corridor and generation
# --------------------------------------------------------------------------------------


def _pack_instance(obstacles: list[Obstacle], capacity: int) -> dict[str, np.ndarray]:
    if len(obstacles) > capacity:
        raise ValueError(f"instance has {len(obstacles)} obstacles, capacity is {capacity}")
    instance = {
        "kind": np.zeros(capacity, np.int32),
        "size": np.zeros((capacity, 3), np.float32),
        "origin": np.zeros((capacity, 3), np.float32),
        "motion": np.zeros(capacity, np.int32),
        "params": np.zeros((capacity, MOTION_PARAMS), np.float32),
        "active": np.zeros(capacity, bool),
    }
    for index, obstacle in enumerate(obstacles):
        instance["kind"][index] = obstacle.kind
        instance["size"][index] = obstacle.size
        instance["origin"][index] = obstacle.origin
        instance["motion"][index] = obstacle.motion
        instance["params"][index] = obstacle.params
        instance["active"][index] = True
    return instance


def _stack_instances(instances: list[dict[str, np.ndarray]]) -> SceneBank:
    return SceneBank(
        kind=jnp.asarray(np.stack([item["kind"] for item in instances])),
        size=jnp.asarray(np.stack([item["size"] for item in instances])),
        origin=jnp.asarray(np.stack([item["origin"] for item in instances])),
        motion=jnp.asarray(np.stack([item["motion"] for item in instances])),
        params=jnp.asarray(np.stack([item["params"] for item in instances])),
        active=jnp.asarray(np.stack([item["active"] for item in instances])),
        start=jnp.zeros((len(instances), 3)),
        goal=jnp.zeros((len(instances), 3)),
        difficulty=jnp.zeros((len(instances),), jnp.int32),
        subtype=jnp.zeros((len(instances),), jnp.int32),
        world_low=jnp.zeros(3),
        world_high=jnp.zeros(3),
        subtype_names=(),
    )
