"""Bounded pixel-region grounding; no model calls, movement, or box expansion."""

import math
from collections import Counter

from sim_engine.navigation.budget import checkpoint

from .limits import distance_allowed, validate_distance_limits
from .visual_geometry import backproject, validate_floor_target


def validate_bbox(bbox):
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1000 for x in bbox)
    ):
        raise ValueError("invalid normalized region box")
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0 or x1 - x0 > 500 or y1 - y0 > 500 or (x1 - x0) * (y1 - y0) > 150000:
        raise ValueError("region box must be ordered and cover a small stopping patch")
    return list(bbox)


def project_world(point, intrinsics, sensor, width, height):
    import numpy as np

    q = np.asarray(sensor["rotation_xyzw"], dtype=float)
    q = q.copy()
    q[:3] *= -1
    delta = np.asarray(point) - np.asarray(sensor["position"])
    local = delta + 2 * np.cross(q[:3], np.cross(q[:3], delta) + q[3] * delta)
    if not np.isfinite(local).all() or local[2] >= -1e-6:
        raise ValueError("point is not in front of selected camera")
    k = np.asarray(intrinsics)
    px = k[0, 2] + k[0, 0] * local[0] / (-local[2])
    py = k[1, 2] - k[1, 1] * local[1] / (-local[2])
    return [float(px / (width - 1) * 1000), float(py / (height - 1) * 1000)]


def inside_bbox(uv, bbox):
    x0, y0, x1, y1 = validate_bbox(bbox)
    return x0 <= uv[0] <= x1 and y0 <= uv[1] <= y1


def _region_samples(bbox, shape, grid_size, safe_mask):
    """Cover occupied mask cells; examine all pixels when support is sparse.

    Pixel selection is vectorized. Expensive NavMesh/geodesic checks remain
    bounded by the original grid_size**2 sample budget.
    """
    import numpy as np

    x0, y0, x1, y1 = bbox
    budget = grid_size**2
    if safe_mask is None:
        return (
            [
                (float(u), float(v))
                for v in np.linspace(y0, y1, grid_size)
                for u in np.linspace(x0, x1, grid_size)
            ],
            {"sampling_policy": "uniform_grid_v1", "sampling_budget": budget},
        )
    height, width = shape
    yy, xx = np.nonzero(safe_mask)
    uv = np.column_stack((xx * 1000.0 / (width - 1), yy * 1000.0 / (height - 1)))
    uv = uv[(uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)]
    count = len(uv)
    metadata = {
        "sampling_policy": "visible_safe_mask_stratified_v1",
        "sampling_budget": budget,
        "safe_pixels_in_bbox": count,
        "sampling_cells_occupied": 0,
    }
    if not count:
        return [], metadata
    normalized = (uv - [x0, y0]) / [x1 - x0, y1 - y0]
    cells = np.minimum((normalized * grid_size).astype(int), grid_size - 1)
    cell_id = cells[:, 1] * grid_size + cells[:, 0]
    metadata["sampling_cells_occupied"] = len(np.unique(cell_id))
    center_distance = np.sum((normalized - 0.5) ** 2, axis=1)
    center_order = np.lexsort((uv[:, 0], uv[:, 1], center_distance))
    if count <= budget:
        selected = center_order
    else:
        # One representative per occupied cell prevents a dense central patch
        # from crowding out narrow or disconnected visible support elsewhere.
        cell_distance = np.sum((normalized * grid_size - cells - 0.5) ** 2, axis=1)
        order = np.lexsort((uv[:, 0], uv[:, 1], center_distance, cell_distance, cell_id))
        _, first = np.unique(cell_id[order], return_index=True)
        representatives = order[first]
        remaining = np.ones(count, dtype=bool)
        remaining[representatives] = False
        extras = center_order[remaining[center_order]][: budget - len(representatives)]
        selected = np.concatenate((representatives, extras))
        selected = selected[np.lexsort((uv[selected, 0], uv[selected, 1], center_distance[selected]))]
    return uv[selected].tolist(), metadata


def sample_region(
    bbox,
    depth,
    intrinsics,
    sensor,
    start,
    mesh,
    guard,
    shortest,
    *,
    grid_size=13,
    max_distance=None,
    min_distance=0.0,
    safe_mask=None,
):
    """Return center-ranked safe candidates strictly inside the model's original box."""
    import numpy as np

    bbox = validate_bbox(bbox)
    x0, y0, x1, y1 = bbox
    h, w = depth.shape
    validate_distance_limits(min_distance, max_distance)
    if safe_mask is not None and (safe_mask.shape != depth.shape or safe_mask.dtype != np.bool_):
        raise ValueError("invalid visible floor mask")
    if type(grid_size) is not int or not 3 <= grid_size <= 31:
        raise ValueError("region grid outside bounded limits")
    if min(h, w) < 2:
        raise ValueError("depth image too small for region sampling")
    samples, sampling = _region_samples(bbox, depth.shape, grid_size, safe_mask)
    center = np.asarray([(x0 + x1) / 2, (y0 + y1) / 2])
    candidates = []
    rejected = Counter()
    seen = set()
    for u, v in samples:
        checkpoint(query=1, phase="region_sampling")
        try:
            g = backproject(float(u), float(v), depth, intrinsics, sensor)
            if safe_mask is not None:
                px, py = g["pixel_xy"]
                if not safe_mask[py, px]:
                    raise ValueError("outside visible safe floor mask")
            point = np.asarray(g["world_point"])
            goal = np.asarray(mesh.snap_point(point))
            checks = validate_floor_target(point, goal, start)
            if not mesh.is_navigable(goal):
                raise ValueError("not navigable")
            clearance = float(guard.clearance(goal))
            if clearance + 1e-6 < guard.profile.clearance_m:
                raise ValueError("insufficient endpoint clearance")
            uv = project_world(goal, intrinsics, sensor, w, h)
            if not inside_bbox(uv, bbox):
                raise ValueError("projected NavMesh point leaves original box")
            route = shortest(start, goal)
            distance = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
            if not distance_allowed(distance, min_distance, max_distance):
                raise ValueError("outside configured distance or disconnected")
            key = tuple(np.round(goal / 0.025).astype(int))
            if key in seen:
                raise ValueError("duplicate surface sample")
            seen.add(key)
            rank = float(np.linalg.norm((np.asarray([u, v]) - center) / np.asarray([x1 - x0, y1 - y0])))
            candidates.append(
                dict(
                    g,
                    navmesh_point=goal.tolist(),
                    snap_distance_m=float(np.linalg.norm(point - goal)),
                    grounding_checks=checks,
                    clearance_m=clearance,
                    projected_uv=uv,
                    sample_uv=[float(u), float(v)],
                    center_rank=rank,
                    geodesic_m=distance,
                )
            )
        except (ValueError, RuntimeError) as error:
            rejected[str(error)] += 1
    candidates.sort(key=lambda x: (x["center_rank"], -min(x["clearance_m"], 0.15), x["sample_uv"]))
    return {
        "bbox": bbox,
        "grid_size": grid_size,
        "samples": len(samples),
        **sampling,
        "rejection_counts": dict(rejected),
        "candidates": candidates,
        "candidate_count": len(candidates),
        "box_expanded": False,
        "rank_policy": "nearest normalized image-box center, then clearance; minimum clearance mandatory",
        "distance_policy": {"min_geodesic_m": min_distance, "max_geodesic_m": max_distance},
    }
