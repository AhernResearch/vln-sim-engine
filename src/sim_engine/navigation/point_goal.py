"""Ground one selected pixel on reachable floor; no rendering, model call or movement."""

import numpy as np

from .safe_image import safe_floor_mask


class TargetRejected(ValueError):
    """The selection cannot become a leg goal; the message is safe to show a planner."""

    status = "invalid_target"

    def __init__(self, reason, private=None):
        super().__init__(reason)
        self.private = private or {}


class Unreachable(TargetRejected):
    status = "unreachable"


# safe_floor_mask rejection -> public reason.
REASONS = {
    "not connected to start": "that floor is not reachable from here",
    "insufficient endpoint clearance": "that spot is too close to obstacles",
}
NOT_FLOOR = "that spot is not on walkable floor"


def intrinsics(projection, width, height):
    """Pinhole K from a Habitat projection matrix, as in the collection camera record."""
    p = [[float(v) for v in row] for row in projection]
    return [[p[0][0] * width / 2, 0.0, (width - 1) / 2],
            [0.0, p[1][1] * height / 2, (height - 1) / 2],
            [0.0, 0.0, 1.0]]


def search_box(point, radius_px, width, height):
    """Normalized box around the point; at least one pixel on each side."""
    rx = max(radius_px, 1) * 1000 / (width - 1)
    ry = max(radius_px, 1) * 1000 / (height - 1)
    x, y = point
    return [max(0.0, x - rx), max(0.0, y - ry), min(1000.0, x + rx), min(1000.0, y + ry)]


def ground_point(depth, k, sensor, start, mesh, guard, point, legs):
    """Nearest safe floor pixel within ``legs.snap_radius`` of ``point``.

    Returns the selected pixel, its NavMesh goal and the snap distance. Raises
    ``TargetRejected`` with a planner-facing reason when no pixel qualifies.
    """
    depth = np.asarray(depth, dtype=float).squeeze()
    height, width = depth.shape
    px = round(point[0] * (width - 1) / 1000)
    py = round(point[1] * (height - 1) / 1000)
    radius = legs.snap_radius * (width - 1)
    mask, goals, clearances, stats = safe_floor_mask(
        depth, k, sensor, start, mesh, guard,
        horizontal_limit=legs.horizontal_limit_m, vertical_limit=legs.vertical_limit_m,
        slope_degrees=legs.slope_degrees, same_level=legs.same_level,
        bbox=search_box(point, radius, width, height),
    )
    ys, xs = np.nonzero(mask)
    distance = np.hypot(xs - px, ys - py)
    inside = distance <= max(radius, 0.5)
    if not inside.any():
        raise TargetRejected(rejection_reason(stats))
    best = np.argmin(np.where(inside, distance, np.inf))
    x, y = int(xs[best]), int(ys[best])
    return {
        "selected_pixel_xy": [px, py],
        "grounded_pixel_xy": [x, y],
        "snap_px": float(distance[best]),
        "goal": goals[y, x].astype(float).tolist(),
        "clearance_m": float(clearances[y, x]),
    }


def rejection_reason(stats):
    if not stats.get("visible_floor_candidates"):
        return "that spot is not visible floor"
    counts = stats.get("rejection_counts") or {}
    common = max(counts, key=counts.get, default=None)
    return REASONS.get(common, NOT_FLOOR)
