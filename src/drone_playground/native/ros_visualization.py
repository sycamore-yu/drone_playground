"""ROS marker conversion without a ROS runtime dependency."""

import numpy as np

from .geometry import ConvexPolytope, SafeFlightCorridor


def corridor_from_markers(markers, name):
    """A complete MarkerArray snapshot: convex TRIANGLE_LIST meshes in world."""
    meshes = {}
    for marker in markers:
        if marker.action == 3:  # DELETEALL
            meshes.clear()
            continue
        key = (marker.ns, marker.id)
        if marker.action == 2:
            meshes.pop(key, None)
            continue
        if marker.type != 11 or marker.action != 0:
            continue
        if marker.header.frame_id != "world":
            raise ValueError("SFC markers must use the world frame")
        q = marker.pose.orientation
        quaternion = np.array([q.x, q.y, q.z, q.w], float)
        if not np.isfinite(quaternion).all() or not np.isclose(
            np.linalg.norm(quaternion), 1.0, atol=1e-4
        ):
            raise ValueError("SFC marker orientation must be a unit quaternion")
        x, y, z, w = quaternion
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        points = np.array([[p.x, p.y, p.z] for p in marker.points], float)
        scale = np.array([marker.scale.x, marker.scale.y, marker.scale.z])
        p = marker.pose.position
        vertices = np.unique(points * scale @ rotation.T + [p.x, p.y, p.z], axis=0)
        meshes[key] = ConvexPolytope(vertices=vertices)
    return SafeFlightCorridor(name, tuple(meshes.values()))
