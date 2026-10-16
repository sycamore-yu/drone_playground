"""Fixed-sampling scene complexity and collision-free grid reference paths."""

import heapq
import itertools

import jax
import jax.numpy as jnp
import numpy as np


class ReferencePath:
    """A conservative 26-connected grid, with certified spherical-body clearance."""

    def __init__(self, scene, low, high, radius, resolution=0.5):
        """Certify entire grid cells from the signed distance at their centers."""
        self.low, self.high = np.asarray(low), np.asarray(high)
        self.radius, self.resolution = radius, resolution
        self.axes = [
            np.arange(a + resolution / 2, b, resolution) for a, b in zip(low, high, strict=True)
        ]
        points = np.stack(np.meshgrid(*self.axes, indexing="ij"), -1)
        self.points = points
        self.clearance = jax.jit(scene.clearance)
        flat = points.reshape(-1, 3)
        distances = np.concatenate(
            [
                np.asarray(self.clearance(jnp.asarray(chunk)))
                for chunk in np.array_split(flat, max(1, len(flat) // 2048))
            ]
        )
        self.free = distances.reshape(points.shape[:-1]) > radius + np.sqrt(3) * resolution / 2
        self.offsets = [d for d in itertools.product((-1, 0, 1), repeat=3) if any(d)]

    def segment_free(self, start, end):
        """Bound between-sample clearance using the distance function's Lipschitz bound."""
        length = float(np.linalg.norm(end - start))
        count = max(2, int(np.ceil(length / 0.05)) + 1)
        points = np.linspace(start, end, count)
        if np.any(points < self.low) or np.any(points > self.high):
            return False
        distances = np.asarray(self.clearance(jnp.asarray(points)))
        return bool(np.all(distances > self.radius + length / (count - 1) / 2))

    def length(self, start, end):
        """Return a grid path length, or None when this resolution cannot certify a path."""
        start, end = np.asarray(start), np.asarray(end)
        if self.segment_free(start, end):
            return float(np.linalg.norm(end - start))

        def connector(point):
            distances = np.linalg.norm(self.points - point, axis=-1)
            nearby = np.argwhere(self.free & (distances < 2 * self.resolution))
            for index in sorted(map(tuple, nearby), key=lambda i: distances[i]):
                if self.segment_free(point, self.points[index]):
                    return index, float(distances[index])
            return None

        first, last = connector(start), connector(end)
        if first is None or last is None:
            return None
        source, prefix = first
        target, suffix = last
        costs, queue = {source: prefix}, [(prefix, prefix, source)]
        while queue:
            _, cost, node = heapq.heappop(queue)
            if cost != costs[node]:
                continue
            if node == target:
                return cost + suffix
            for offset in self.offsets:
                neighbor = tuple(a + b for a, b in zip(node, offset, strict=True))
                if any(i < 0 or i >= n for i, n in zip(neighbor, self.free.shape, strict=True)):
                    continue
                if not self.free[neighbor]:
                    continue
                proposed = cost + self.resolution * np.linalg.norm(offset)
                if proposed < costs.get(neighbor, float("inf")):
                    costs[neighbor] = proposed
                    estimate = np.linalg.norm(self.points[neighbor] - self.points[target])
                    heapq.heappush(queue, (proposed + estimate, proposed, neighbor))
        return None


def traversability(scene, low, high, radius, *, times):
    """Average horizontal free sight distance / body diameter on a fixed spatial grid."""
    low, high = np.asarray(low), np.asarray(high)
    axes = [np.linspace(a, b, n + 2)[1:-1] for a, b, n in zip(low, high, (20, 8, 3), strict=True)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    angles = np.arange(16) * (2 * np.pi / 16)
    directions = np.stack((np.cos(angles), np.sin(angles), np.zeros(16)), -1)
    valid_total, distances = 0, []
    clearance, raycast = jax.jit(scene.clearance), jax.jit(scene.raycast)
    for time in times:
        valid = np.asarray(clearance(jnp.asarray(points), time)) > radius
        ranges = np.asarray(raycast(jnp.asarray(points), jnp.asarray(directions), time, 40.0))
        # Treat the task boundary as the edge of the flyable volume.
        with np.errstate(divide="ignore", invalid="ignore"):
            boundary = np.where(
                directions[None] > 0,
                (high - points[:, None]) / directions[None],
                (low - points[:, None]) / directions[None],
            )
        boundary = np.where(np.abs(directions[None]) > 1e-12, boundary, np.inf).min(-1)
        distances.extend(np.minimum(ranges, boundary)[valid].ravel().tolist())
        valid_total += int(valid.sum())
    return {
        "value": float(np.mean(distances) / (2 * radius)) if distances else None,
        "rays": len(distances),
        "free_positions": valid_total,
        "protocol": {
            "grid": [20, 8, 3],
            "horizontal_directions": 16,
            "times_seconds": list(times),
            "max_range_metres": 40.0,
            "body_diameter_metres": 2 * radius,
            "geometry": "visible surfaces, contact-free origins, task bounds",
        },
    }
