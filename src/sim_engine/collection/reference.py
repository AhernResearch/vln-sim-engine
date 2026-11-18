"""Select augmentation centers only after clearance planning and smoothing."""

import math


def reference_state(path, fraction, start_yaw):
    if fraction not in (0.0, 0.5) or not path:
        raise ValueError("start/mid fractions and a prepared surface path are required")
    arcs = [0.0]
    for a, b in zip(path, path[1:]):
        arcs.append(arcs[-1] + math.dist(a, b))
    index = min(range(len(path)), key=lambda i: abs(arcs[i] - fraction * arcs[-1]))
    yaw = start_yaw
    if index:
        a, b = path[index - 1], path[index]
        yaw = math.atan2(-(b[0] - a[0]), -(b[2] - a[2]))
    return {
        "path_index": index,
        "world_position": path[index],
        "yaw": yaw,
        "requested_fraction": fraction,
        "arc_m": arcs[index],
        "path_length_m": arcs[-1],
        "source": "planned_smoothed_surface_path_not_raw_interpolation",
    }
