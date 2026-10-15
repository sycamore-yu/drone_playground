"""Inspect geometric routes at observed S03 stopping positions, without simulation."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent


def main():
    """Measure planar shortest-path progress versus direct-goal progress at recorded positions."""
    resolution, margin, body_radius = 0.1, 0.15, 0.07
    x, y = np.arange(1001) * resolution, np.arange(401) * resolution - 20
    xx, yy = np.meshgrid(x, y)
    free = (xx > body_radius + margin) & (xx < 100 - body_radius - margin)
    free &= np.abs(yy) < 20 - body_radius - margin
    cylinders = []
    for body in ET.parse(ROOT / "assets/scenes/navigation/S03.xml").findall("worldbody/body"):
        geom = body.find("geom")
        center = np.array(list(map(float, body.attrib["pos"].split())))
        radius, half_height = map(float, geom.attrib["size"].split())
        assert geom.attrib["type"] == "cylinder" and center[2] == half_height == 3
        assert body.attrib["user"] == "0 0 0 0 0 0"
        cylinders.append((center[:2], radius))
        free &= (xx - center[0]) ** 2 + (yy - center[1]) ** 2 > (radius + body_radius + margin) ** 2
    indices = np.arange(free.size).reshape(free.shape)
    rows, cols, weights = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ya = slice(0, free.shape[0] - dy)
        yb = slice(dy, free.shape[0])
        xa = slice(max(0, -dx), free.shape[1] - max(0, dx))
        xb = slice(max(0, dx), free.shape[1] - max(0, -dx))
        valid = free[ya, xa] & free[yb, xb]
        if dy and dx:
            valid &= free[ya, xb] & free[yb, xa]
        a, b = indices[ya, xa][valid], indices[yb, xb][valid]
        rows.extend((a, b))
        cols.extend((b, a))
        weight = np.full(len(a), resolution * np.hypot(dx, dy))
        weights.extend((weight, weight))
    graph = coo_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(free.size, free.size),
    ).tocsr()
    target = indices[200, 980]
    assert free[200, 980]
    distance, predecessor = dijkstra(graph, indices=target, return_predecessors=True)
    summary = json.loads((OUTPUT / "summary.json").read_text())
    samples = [("nominal_start", np.array([2.0, 0.0, 3.0]))]
    samples += [
        (f"{name}_world{case['world_index']}", np.array(case["final_position_metres"]))
        for name, record in summary["runs"].items()
        for case in record["cases"]
    ]
    measured = []
    for name, point in samples:
        iy, ix = np.rint([(point[1] + 20) / resolution, point[0] / resolution]).astype(int)
        current = indices[iy, ix]
        assert free[iy, ix] and np.isfinite(distance[current])
        path = [int(current)]
        while current != target:
            previous = current
            current = predecessor[current]
            assert current >= 0 and distance[current] < distance[previous]
            path.append(int(current))
        py, px = np.unravel_index(path, free.shape)
        route = np.column_stack((x[px], y[py], np.full(len(path), point[2])))
        lengths = np.linalg.norm(np.diff(route, axis=0), axis=-1)
        assert np.isclose(lengths.sum(), distance[path[0]])
        assert np.all(lengths > 0)
        starts, delta = route[:-1, :2], np.diff(route[:, :2], axis=0)
        clearance = float("inf")
        for center, radius in cylinders:
            fraction = np.clip(np.sum((center - starts) * delta, axis=-1) / lengths**2, 0, 1)
            nearest = starts + fraction[:, None] * delta
            clearance = min(
                clearance,
                float((np.linalg.norm(nearest - center, axis=-1) - radius - body_radius).min()),
            )
        assert clearance > 0
        direct = np.linalg.norm(route - [98, 0, 3], axis=-1)
        progress = direct[:-1] - direct[1:]
        local = np.cumsum(lengths) <= 2
        negative = progress < -1e-6
        measured.append(
            {
                "sample": name,
                "observed_position_metres": point.tolist(),
                "grid_start_metres": route[0].tolist(),
                "planar_grid_path_metres": float(lengths.sum()),
                "direct_goal_distance_metres": float(direct[0]),
                "path_edges_with_negative_direct_progress": int(negative.sum()),
                "path_edge_count": len(lengths),
                "first2m_negative_direct_progress_edges": int((negative & local).sum()),
                "first2m_net_direct_progress_metres": float(progress[local].sum()),
                "first2m_total_backward_direct_distance_metres": float(
                    -progress[negative & local].sum()
                ),
                "minimum_segment_body_clearance_metres": clearance,
                "first2m_route_metres": route[: int(local.sum()) + 1].tolist(),
            }
        )
    result = {
        "scope": "Offline S03 geometric evidence; no reward change, policy learning or simulation",
        "grid_resolution_metres": resolution,
        "extra_obstacle_margin_metres": margin,
        "body_radius_metres": body_radius,
        "samples": measured,
        "limits": "Eight-neighbour planar grid approximation, not exact continuous geodesics. "
        "Routes keep observed height; reaching the 3D goal still requires vertical settling. "
        "No dynamics, controls, exploration or convergence benefit is established.",
        "reference": "https://aihabitat.org/docs/habitat-lab/habitat.tasks.nav.nav.DistanceToGoalReward.html",
        "interpretation": "Every recorded sample has a geometrically collision-free reference "
        "route with zero negative direct-progress edges. This does not support necessary "
        "distance-increasing detours as the specific cause of these stalls. It does not "
        "prove the Actor can discover/follow those routes or that geodesic shaping has no benefit.",
    }
    (OUTPUT / "geodesic.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in measured:
        print(
            {
                k: row[k]
                for k in (
                    "sample",
                    "planar_grid_path_metres",
                    "first2m_negative_direct_progress_edges",
                    "first2m_net_direct_progress_metres",
                )
            }
        )


if __name__ == "__main__":
    main()
