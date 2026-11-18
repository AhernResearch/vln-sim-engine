"""Geometric limits and path-scaled execution budgets; no model policy."""

import math


def validate_distance_limits(min_distance=0.0, max_distance=None):
    if (
        type(min_distance) not in (int, float)
        or not math.isfinite(min_distance)
        or min_distance < 0
        or (
            max_distance is not None
            and (
                type(max_distance) not in (int, float)
                or not math.isfinite(max_distance)
                or max_distance <= min_distance
            )
        )
    ):
        raise ValueError("invalid target distance limits")


def distance_allowed(distance, min_distance=0.0, max_distance=None):
    """No collection-length cap unless explicitly requested in a legacy profile."""
    return (
        math.isfinite(distance)
        and distance > 0
        and distance >= min_distance
        and (max_distance is None or distance <= max_distance)
    )


def execution_budget(
    path, profile, initial_yaw, base_steps, *, time_multiplier=3.0, settle_seconds=10.0
):
    """Derive a finite watchdog from the planned path; no fixed route-length cutoff."""
    if (
        type(base_steps) is not int
        or base_steps < 1
        or not math.isfinite(initial_yaw)
        or not math.isfinite(time_multiplier)
        or time_multiplier < 1
        or not math.isfinite(settle_seconds)
        or settle_seconds < 0
        or any(
            not math.isfinite(v) or v <= 0
            for v in [profile.dt_s, profile.v_max_mps, profile.omega_max_rps]
        )
    ):
        raise ValueError("invalid execution budget policy")
    points = [list(p) for p in path]
    if len(points) < 2 or any(len(p) != 3 or any(not math.isfinite(v) for v in p) for p in points):
        raise ValueError("invalid execution budget path")
    length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    headings = [
        math.atan2(-(b[0] - a[0]), -(b[2] - a[2]))
        for a, b in zip(points, points[1:])
        if math.hypot(b[0] - a[0], b[2] - a[2]) > 1e-6
    ]
    if not headings:
        raise ValueError("execution path has no translation")

    def wrap(v):
        return (v + math.pi) % (2 * math.pi) - math.pi

    turn = abs(wrap(headings[0] - initial_yaw)) + sum(
        abs(wrap(b - a)) for a, b in zip(headings, headings[1:])
    )
    nominal = length / profile.v_max_mps + turn / profile.omega_max_rps
    steps = max(base_steps, math.ceil((time_multiplier * nominal + settle_seconds) / profile.dt_s))
    return {
        "mode": "path_scaled",
        "base_steps": base_steps,
        "max_steps": steps,
        "control_dt_s": profile.dt_s,
        "path_length_m": length,
        "total_heading_change_rad": turn,
        "nominal_seconds": nominal,
        "time_multiplier": time_multiplier,
        "settle_seconds": settle_seconds,
        "budget_seconds": steps * profile.dt_s,
        "failure_is_not_visibility_or_distance_rejection": True,
    }
