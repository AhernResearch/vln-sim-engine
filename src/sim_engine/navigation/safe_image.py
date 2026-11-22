"""Visible depth-derived safe stopping pixels; no rendering, model call, or movement."""

import math
from collections import Counter

import numpy as np

from sim_engine.navigation.budget import checkpoint

from .limits import distance_allowed, validate_distance_limits
from .visual_region import validate_bbox


def depth_world(depth, intrinsics, sensor):
    if depth.ndim != 2 or min(depth.shape) < 3:
        raise ValueError("invalid depth image")
    k = np.asarray(intrinsics, dtype=float)
    q = np.asarray(sensor["rotation_xyzw"], dtype=float)
    position = np.asarray(sensor["position"], dtype=float)
    if (
        k.shape != (3, 3)
        or q.shape != (4,)
        or position.shape != (3,)
        or not np.isfinite(k).all()
        or not np.isfinite(q).all()
        or not np.isfinite(position).all()
        or min(k[0, 0], k[1, 1]) <= 0
        or not np.isclose(np.linalg.norm(q), 1, atol=1e-5)
    ):
        raise ValueError("invalid sensor calibration")
    valid = np.isfinite(depth) & (depth > 0)
    distance = np.where(valid, depth, 0.0)
    y, x = np.indices(depth.shape)
    local = np.stack(
        [(x - k[0, 2]) * distance / k[0, 0], -(y - k[1, 2]) * distance / k[1, 1], -distance], axis=-1
    )
    world = local + 2 * np.cross(q[:3], np.cross(q[:3], local) + q[3] * local) + position
    return world, valid


def safe_floor_mask(
    depth,
    intrinsics,
    sensor,
    start,
    mesh,
    guard,
    *,
    max_distance=None,
    min_distance=0.0,
    horizontal_limit=0.01,
    vertical_limit=0.2,
    slope_degrees=25.0,
    bbox=None,
    same_level=True,
):
    """Keep a per-pixel mapping to NavMesh, using actual visible depth surfaces.

    Connectivity is the start island; geodesic distance and controller feasibility
    remain selected-point checks. NavMesh height may differ from rendered floor.
    ``same_level=False`` drops the start-height clause so connected stairs and other
    levels remain candidates; the surface-to-NavMesh projection limit still applies.
    """
    start = np.asarray(start, dtype=float)
    validate_distance_limits(min_distance, max_distance)
    if (
        start.shape != (3,)
        or not np.isfinite(start).all()
        or not 0 < horizontal_limit <= 0.05
        or not 0 < vertical_limit <= 0.2
        or not 0 < slope_degrees <= 30
    ):
        raise ValueError("invalid safe mask limits")
    world, valid = depth_world(depth, intrinsics, sensor)
    normals = np.cross(world[1:-1, 2:] - world[1:-1, :-2], world[2:, 1:-1] - world[:-2, 1:-1])
    length = np.linalg.norm(normals, axis=-1)
    floor = np.zeros(depth.shape, dtype=bool)
    floor[1:-1, 1:-1] = (
        valid[1:-1, 1:-1]
        & valid[1:-1, 2:]
        & valid[1:-1, :-2]
        & valid[2:, 1:-1]
        & valid[:-2, 1:-1]
        & (length > 1e-10)
        & (np.abs(normals[:, :, 1]) / np.maximum(length, 1e-10) >= math.cos(math.radians(slope_degrees)))
    )
    horizontal = np.linalg.norm((world - start)[:, :, [0, 2]], axis=-1)
    floor &= (horizontal > 0) & (horizontal >= min_distance)
    if same_level:
        floor &= np.abs(world[:, :, 1] - start[1]) <= vertical_limit
    if max_distance is not None:
        floor &= horizontal <= max_distance
    scope = {}
    if bbox is not None:

        x0, y0, x1, y1 = validate_bbox(bbox)
        height, width = depth.shape
        # Conservative pixel halo covers both normalized-pixel arithmetic orders
        # used by support selection and stratified sampling, including roundoff.
        left = max(0, math.floor(x0 * (width - 1) / 1000) - 1)
        right = min(width, math.ceil(x1 * (width - 1) / 1000) + 2)
        top = max(0, math.floor(y0 * (height - 1) / 1000) - 1)
        bottom = min(height, math.ceil(y1 * (height - 1) / 1000) + 2)
        computed = np.zeros(depth.shape, dtype=bool)
        computed[top:bottom, left:right] = True
        scope = {"schema_version": 2, "mask_scope": "target_bbox_with_pixel_halo",
                 "requested_bbox": list(bbox), "computed_pixel_bounds_xyxy": [left, top, right, bottom],
                 "outside_computed_bounds": "unknown_not_obstacle",
                 "full_visible_floor_candidates": int(floor.sum())}
        floor &= computed
    mask = np.zeros(depth.shape, dtype=bool)
    goals = np.full((*depth.shape, 3), np.nan, dtype=np.float32)
    clearances = np.full(depth.shape, np.nan, dtype=np.float32)
    rejected = Counter()
    island = mesh.get_island(start.astype(np.float32))
    if island < 0:
        raise ValueError("start has no NavMesh island")
    for y, x in np.argwhere(floor):
        checkpoint(query=1, phase="floor_mask")
        point = world[y, x]
        try:
            goal = np.asarray(mesh.snap_point(point.astype(np.float32)), dtype=float)
            if goal.shape != (3,) or not np.isfinite(goal).all() or not mesh.is_navigable(goal):
                raise ValueError("not navigable")
            if np.linalg.norm((goal - point)[[0, 2]]) > horizontal_limit:
                raise ValueError("horizontal projection outside tolerance")
            if abs(goal[1] - point[1]) > vertical_limit or (
                same_level and abs(goal[1] - start[1]) > vertical_limit
            ):
                raise ValueError("surface height outside tolerance")
            if mesh.get_island(goal.astype(np.float32)) != island:
                raise ValueError("not connected to start")
            if not distance_allowed(
                float(np.linalg.norm((goal - start)[[0, 2]])), min_distance, max_distance
            ):
                raise ValueError("outside configured local distance")
            clearance = float(guard.clearance(goal))
            if not math.isfinite(clearance) or clearance + 1e-6 < guard.profile.clearance_m:
                raise ValueError("insufficient endpoint clearance")
            mask[y, x] = True
            goals[y, x] = goal
            clearances[y, x] = clearance
        except (ValueError, RuntimeError) as error:
            rejected[str(error)] += 1
    return (
        mask,
        goals,
        clearances,
        {
            **scope,
            "safe_pixels": int(mask.sum()),
            "total_pixels": int(mask.size),
            "visible_floor_candidates": int(floor.sum()),
            "rejection_counts": dict(rejected),
            "horizontal_projection_limit_m": horizontal_limit,
            "vertical_limit_m": vertical_limit,
            "slope_degrees": slope_degrees,
            "minimum_clearance_m": guard.profile.clearance_m,
            "min_euclidean_local_distance_m": min_distance,
            "max_euclidean_local_distance_m": max_distance,
            "connectivity": "same_start_navmesh_island",
            "same_level": same_level,
            "visible_surface_source": "observed_depth_with_floor_normal_filter",
            "selected_point_geodesic_and_controller_checks_required": True,
        },
    )
