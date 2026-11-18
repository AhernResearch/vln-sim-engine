"""RGB-D projection and floor checks; no model interface."""

import math


def backproject(u, v, depth, intrinsics, sensor):
    """Normalized RGB pixel + axial -Z depth -> Habitat world XYZ, with xyzw quaternion."""
    import numpy as np

    if not all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1000 for x in (u, v)):
        raise ValueError("normalized point outside image")
    if depth.ndim != 2:
        raise ValueError("depth must be an image")
    height, width = depth.shape
    px, py = int(round(u * (width - 1) / 1000)), int(round(v * (height - 1) / 1000))
    distance = float(depth[py, px])
    if not math.isfinite(distance) or distance <= 0:
        raise ValueError("selected point has invalid depth")
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
        or k[0, 0] <= 0
        or k[1, 1] <= 0
        or not np.isclose(np.linalg.norm(q), 1, atol=1e-5)
    ):
        raise ValueError("invalid sensor calibration")
    local = np.asarray(
        [(px - k[0, 2]) * distance / k[0, 0], -(py - k[1, 2]) * distance / k[1, 1], -distance]
    )
    rotated = local + 2 * np.cross(q[:3], np.cross(q[:3], local) + q[3] * local)
    return {
        "pixel_xy": [px, py],
        "depth_m": distance,
        "camera_point": local.tolist(),
        "world_point": (position + rotated).tolist(),
    }


def validate_floor_target(point, snapped, start, *, horizontal_limit=0.05, vertical_limit=0.20):
    """Allow NavMesh voxel-height offset without moving a selected point across a doorway/wall."""
    import numpy as np

    point, snapped, start = (np.asarray(p, dtype=float) for p in (point, snapped, start))
    if any(p.shape != (3,) or not np.isfinite(p).all() for p in (point, snapped, start)):
        raise ValueError("invalid grounded target")
    horizontal = float(np.linalg.norm((point - snapped)[[0, 2]]))
    vertical = float(abs(point[1] - snapped[1]))
    if horizontal > horizontal_limit or vertical > vertical_limit or abs(snapped[1] - start[1]) > 0.20:
        raise ValueError("selected pixel is not near navigable current-floor surface")
    return {
        "horizontal_projection_m": horizontal,
        "vertical_projection_m": vertical,
        "horizontal_limit_m": horizontal_limit,
        "vertical_limit_m": vertical_limit,
    }
