"""Analytic ray casting against the frozen scene primitives.

One geometry description serves collision, reward clearance and both sensor
models, so display, sensor and collision cannot drift apart. The intersection
math is exact for the two swept primitives the scene uses (upright finite
cylinder, axis-aligned box) plus the flat ground plane.

Rays are parameterised as ``p(t) = origin + t * direction``. Directions are
deliberately **not** normalised for the depth camera: with an optical-frame
direction whose z component is one, ``t`` is exactly the axial depth.

Provenance: the primitive-distance forms follow the signed-distance rules in
SANDO's ``scripts/integer_forest_geometry.py``; the intersection forms are this
project's own and are verified against ``mujoco.mj_ray``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from drone_playground.environments.scenes.navigation import KIND_CYLINDER, KIND_SPHERE

EPS = 1e-6
NO_HIT = jnp.inf
FLOOR_MARGIN_M = 100.0
"""The ground plane extends this far past the navigation bounds.

The corridor bounds define where the drone may fly and where a body is out of
bounds; they are not an environment edge. A physically sensible world has ground
beyond the corridor, and without the margin a downward range ray escapes into a
void and reports no return."""


def _cylinder_hit(origin, direction, radius, half_height):
    """Nearest positive intersection distance with an upright finite cylinder."""
    ox, oy, oz = origin[..., 0], origin[..., 1], origin[..., 2]
    dx, dy, dz = direction[..., 0], direction[..., 1], direction[..., 2]
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    c = ox * ox + oy * oy - radius * radius
    discriminant = b * b - 4.0 * a * c
    # A negative discriminant means the ray misses the infinite cylinder. It must
    # not be clamped to zero: the tangent point would then pass the height test
    # and produce a hit for a ray that never touches the surface.
    intersects = (a > EPS) & (discriminant >= 0.0)
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    safe_a = jnp.where(a > EPS, a, 1.0)
    near = (-b - root) / (2.0 * safe_a)
    far = (-b + root) / (2.0 * safe_a)

    def side(t):
        height = oz + t * dz
        return jnp.where((t > EPS) & (jnp.abs(height) <= half_height), t, NO_HIT)

    side_hit = jnp.where(intersects, side(near), NO_HIT)
    side_hit = jnp.where(side_hit == NO_HIT, jnp.where(intersects, side(far), NO_HIT), side_hit)

    def cap(z_cap):
        safe_dz = jnp.where(jnp.abs(dz) > EPS, dz, 1.0)
        t = (z_cap - oz) / safe_dz
        radial = jnp.hypot(ox + t * dx, oy + t * dy)
        return jnp.where((jnp.abs(dz) > EPS) & (t > EPS) & (radial <= radius), t, NO_HIT)

    return jnp.minimum(side_hit, jnp.minimum(cap(half_height), cap(-half_height)))


def _box_hit(origin, direction, half):
    """Nearest positive intersection distance with an axis-aligned box."""
    safe = jnp.where(jnp.abs(direction) > EPS, direction, jnp.where(direction < 0, -EPS, EPS))
    t_low = (-half - origin) / safe
    t_high = (half - origin) / safe
    t_near = jnp.max(jnp.minimum(t_low, t_high), axis=-1)
    t_far = jnp.min(jnp.maximum(t_low, t_high), axis=-1)
    # A zero direction component is a hit only when the origin is inside the slab.
    parallel_outside = jnp.any((jnp.abs(direction) <= EPS) & (jnp.abs(origin) > half), axis=-1)
    hit = (t_far >= jnp.maximum(t_near, EPS)) & ~parallel_outside
    distance = jnp.where(t_near > EPS, t_near, t_far)
    return jnp.where(hit, distance, NO_HIT)


def _plane_hit(origin, direction, height, x_range, y_range):
    """Nearest positive intersection distance with a bounded horizontal plane."""
    dz = direction[..., 2]
    safe_dz = jnp.where(jnp.abs(dz) > EPS, dz, 1.0)
    distance = (height - origin[..., 2]) / safe_dz
    point_x = origin[..., 0] + distance * direction[..., 0]
    point_y = origin[..., 1] + distance * direction[..., 1]
    inside = (
        (point_x >= x_range[0])
        & (point_x <= x_range[1])
        & (point_y >= y_range[0])
        & (point_y <= y_range[1])
    )
    return jnp.where((jnp.abs(dz) > EPS) & (distance > EPS) & inside, distance, NO_HIT)


def _sphere_hit(origin, direction, radius):
    a = jnp.sum(direction * direction, axis=-1)
    b = jnp.sum(origin * direction, axis=-1)
    c = jnp.sum(origin * origin, axis=-1) - radius * radius
    disc = b * b - a * c
    root = jnp.sqrt(jnp.maximum(disc, 0.0))
    near = (-b - root) / jnp.maximum(a, EPS)
    far = (-b + root) / jnp.maximum(a, EPS)
    distance = jnp.where(near > EPS, near, far)
    return jnp.where((disc >= 0.0) & (a > EPS) & (distance > EPS), distance, NO_HIT)


def primitive_hit(kind, size, centre, origin, direction, world):
    """Nearest positive hit distance for one scene slot against a ray bundle.

    Args:
        kind: int32 scalar primitive code.
        size: ``(3,)`` cylinder ``(radius, height, 0)`` or box half extents.
        centre: ``(3,)`` world centre.
        origin: ``(N, 3)`` ray origins.
        direction: ``(N, 3)`` ray directions (not necessarily unit).
        world: dict with ground height and the corridor x/y ranges.

    Returns:
        ``(N,)`` distances, ``inf`` where nothing is hit.
    """
    offset = origin - centre
    cylinder = _cylinder_hit(offset, direction, size[0], size[1] / 2.0)
    box = _box_hit(offset, direction, size)
    sphere = _sphere_hit(offset, direction, size[0])
    return jnp.where(kind == KIND_SPHERE, sphere, jnp.where(kind == KIND_CYLINDER, cylinder, box))


def cast_rays(
    kind,
    size,
    centres,
    active,
    origins,
    directions,
    world_low,
    world_high,
    include_ground: bool = True,
    floor_margin_m: float = FLOOR_MARGIN_M,
):
    """Nearest hit distance over every active primitive of one scene instance.

    Args:
        kind: ``(capacity,)`` primitive codes.
        size: ``(capacity, 3)`` primitive sizes.
        centres: ``(capacity, 3)`` primitive centres **already evaluated at the
            capture time**, so a moving obstacle is ray-cast where it actually is.
        active: ``(capacity,)`` validity mask.
        origins: ``(N, 3)`` world ray origins.
        directions: ``(N, 3)`` world ray directions.
        world_low, world_high: ``(3,)`` corridor bounds in world coordinates.
        include_ground: whether the flat corridor floor reflects rays.

    Returns:
        ``(N,)`` distance in units of ``|direction|``; ``inf`` for no return.
    """

    # Accept host arrays too: the caller may hand in numpy geometry, and a
    # numpy array indexed by a traced scan counter would raise.
    kind = jnp.asarray(kind)
    size = jnp.asarray(size)
    centres = jnp.asarray(centres)
    active = jnp.asarray(active)
    origins = jnp.asarray(origins)
    directions = jnp.asarray(directions)

    def one_slot(best, slot):
        hit = primitive_hit(kind[slot], size[slot], centres[slot], origins, directions, None)
        hit = jnp.where(active[slot], hit, NO_HIT)
        return jnp.minimum(best, hit), None

    best, _ = jax.lax.scan(
        one_slot, jnp.full(origins.shape[:-1], NO_HIT), jnp.arange(kind.shape[0])
    )
    if include_ground:
        ground = _plane_hit(
            origins,
            directions,
            world_low[2],
            (world_low[0] - floor_margin_m, world_high[0] + floor_margin_m),
            (world_low[1] - floor_margin_m, world_high[1] + floor_margin_m),
        )
        best = jnp.minimum(best, ground)
    return best
