"""Plan around ordered sparse guides, smooth, then sample the NavMesh surface."""

import math

from sim_engine.navigation.budget import checkpoint

from .errors import RejectedCandidate


def surface_point(mesh, point, *, horizontal_limit=0.03, vertical_limit=0.25):
    import numpy as np

    point = np.asarray(point, dtype=float)
    snapped = np.asarray(mesh.snap_point(point.astype(np.float32)), dtype=float)
    if (
        not np.isfinite(snapped).all()
        or np.linalg.norm((snapped - point)[[0, 2]]) > horizontal_limit
        or abs(snapped[1] - point[1]) > vertical_limit
    ):
        raise RejectedCandidate("reference cannot be projected to the local NavMesh surface")
    return snapped


def surface_segment(mesh, a, b, profile, spacing=0.025):
    """Read-only geometric connectivity, with no execution-clearance constraint.

    A virtual guide on the boundary may be relocated inward. The final planned
    path, rather than this virtual displacement, must meet execution clearance.
    """
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    count = max(1, math.ceil(float(np.linalg.norm(b - a)) / spacing))
    previous = a
    result = []
    for j in range(1, count + 1):
        checkpoint(query=1)
        desired = a + (b - a) * j / count
        point = surface_point(
            mesh, desired, horizontal_limit=1e-4, vertical_limit=profile.max_height_step_m
        )
        if abs(point[1] - previous[1]) > profile.max_height_step_m:
            raise RejectedCandidate("reference surface changes floor discontinuously")
        reached = np.asarray(
            mesh.try_step_no_sliding(previous.astype(np.float32), point.astype(np.float32))
        )
        if not np.isfinite(reached).all() or np.linalg.norm(reached - point) > 0.001:
            raise RejectedCandidate("reference adjustment crosses a blocked NavMesh segment")
        result.append(point)
        previous = point
    return result


def dense_surface_path(planner, points):
    """Densify only an already planned/smoothed route, retaining surface height."""
    import numpy as np

    profile = planner.profile
    output, mapping = [surface_point(planner.navmesh, points[0])], [0]
    for a, b in zip(points, points[1:]):
        for point in surface_segment(
            planner.navmesh, output[-1], surface_point(planner.navmesh, b), profile, profile.resample_m
        ):
            if not planner.edge_clear(output[-1], point):
                raise RejectedCandidate("surface-resampled planned path lacks clearance")
            if np.linalg.norm(point - output[-1]) > 1e-7:
                output.append(point)
        mapping.append(len(output) - 1)
    return [p.tolist() for p in output], mapping


def adjust_guide(planner, original, radius, label):
    import numpy as np

    mesh, profile = planner.navmesh, planner.profile
    original = np.asarray(original, dtype=float)
    snapped = surface_point(mesh, original, vertical_limit=profile.max_anchor_snap_m)
    before = planner.clearance(snapped)
    chosen = snapped
    if before + 1e-6 < profile.clearance_m:
        chosen = None
        rings = max(1, math.ceil(radius / 0.025))
        for ring in range(1, rings + 1):
            distance = radius * ring / rings
            options = []
            for angle_index in range(32):
                angle = 2 * math.pi * angle_index / 32
                proposal = snapped + [distance * math.cos(angle), 0, distance * math.sin(angle)]
                try:
                    point = surface_point(mesh, proposal, vertical_limit=profile.max_height_step_m)
                except RejectedCandidate:
                    continue
                if np.linalg.norm(point - snapped) > radius + 1e-6:
                    continue
                clearance = planner.clearance(point)
                if clearance + 1e-6 < profile.clearance_m:
                    continue
                try:
                    surface_segment(mesh, snapped, point, profile)
                except RejectedCandidate:
                    continue
                options.append((float(np.linalg.norm(point - snapped)), -clearance, angle_index, point))
            if options:
                chosen = min(options, key=lambda x: x[:3])[3]
                break
        if chosen is None:
            raise RejectedCandidate(f"{label} has no safe local adjustment within {radius}m")
    return chosen, {
        "original": original.tolist(),
        "adjusted": chosen.tolist(),
        "shift_m": float(np.linalg.norm(chosen - snapped)),
        "surface_projection_m": float(np.linalg.norm(snapped - original)),
        "original_clearance_m": before,
        "adjusted_clearance_m": planner.clearance(chosen),
    }


def prepare_guidance(planner, guard, guides, goal, radius, goal_adjustment_radius=0.0):
    import numpy as np

    mesh, profile = planner.navmesh, planner.profile
    records, adjusted = [], []
    for index, original in enumerate(guides):
        chosen, record = adjust_guide(planner, original, radius, f"guide {index}")
        adjusted.append(chosen.tolist())
        records.append(dict(record, guide_index=index))
    goal = np.asarray(goal, dtype=float)
    requested_goal = surface_point(
        mesh, goal, horizontal_limit=0.001, vertical_limit=profile.max_anchor_snap_m
    )
    snapped_goal = requested_goal
    goal_shift = 0.0
    if planner.clearance(snapped_goal) + 1e-6 < profile.clearance_m:
        if goal_adjustment_radius <= 0:
            raise RejectedCandidate("fixed goal has insufficient clearance")
        snapped_goal, adjustment = adjust_guide(planner, requested_goal, goal_adjustment_radius, "goal")
        goal_shift = adjustment["shift_m"]
    prepared = planner.prepare(adjusted + [snapped_goal.tolist()])
    path, mapping = dense_surface_path(planner, prepared["world_points"])
    if not all(guard.edge_clear(a, b) for a, b in zip(path, path[1:])):
        raise RejectedCandidate("final surface path fails execution clearance")
    anchor_indices = [mapping[i] for i in prepared["anchor_indices"]]
    return {
        "path": path,
        "guide_adjustments": records,
        "anchor_indices": anchor_indices,
        "requested_goal": goal.tolist(),
        "requested_goal_surface": requested_goal.tolist(),
        "fixed_goal": snapped_goal.tolist(),
        "goal_adjustment_m": goal_shift,
        "endpoint_review_required": goal_shift > 0.001,
        "adjustment_radius_m": radius,
        "required_clearance_m": profile.clearance_m,
        "planning_order": "sparse_guides>navmesh_clearance>smooth>dense_surface_path",
    }


def plan_with_preferences(planners, guard, guides, goal, radius, goal_adjustment_radius):
    """Prefer comfort clearance; fall back geometrically before any execution."""
    attempts = []
    # Preserve the exact goal if either comfort or minimum-clearance planning can.
    for index, planner in enumerate(planners):
        try:
            value = prepare_guidance(planner, guard, guides, goal, radius)
        except ValueError as error:
            attempts.append(
                {
                    "planner_index": index,
                    "clearance_m": planner.profile.clearance_m,
                    "goal_adjustment_allowed": False,
                    "error": str(error),
                }
            )
            continue
        return dict(
            value,
            status="feasible",
            planner_index=index,
            planning_attempts=attempts,
            comfort_fallback_used=index > 0,
        )
    planner = planners[-1]
    projected = surface_point(
        planner.navmesh, goal, horizontal_limit=0.001, vertical_limit=planner.profile.max_anchor_snap_m
    )
    if goal_adjustment_radius > 0 and planner.clearance(projected) + 1e-6 < planner.profile.clearance_m:
        try:
            value = prepare_guidance(planner, guard, guides, goal, radius, goal_adjustment_radius)
            return dict(
                value,
                status="feasible",
                planner_index=len(planners) - 1,
                planning_attempts=attempts,
                comfort_fallback_used=len(planners) > 1,
            )
        except ValueError as error:
            attempts.append(
                {
                    "planner_index": len(planners) - 1,
                    "clearance_m": planner.profile.clearance_m,
                    "goal_adjustment_allowed": True,
                    "error": str(error),
                }
            )
    return {"status": "infeasible", "reason": attempts[-1]["error"], "planning_attempts": attempts}
