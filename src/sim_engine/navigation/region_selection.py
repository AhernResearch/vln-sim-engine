"""Bounded endpoint refinement and controller previews before one movement attempt."""

import math

import numpy as np

from sim_engine.navigation.budget import checkpoint

from .guidance import surface_point, surface_segment
from .limits import distance_allowed, validate_distance_limits
from .visual_region import inside_bbox


def local_adjustments(candidate, mesh, guard, project, bbox, radius, target_valid=None):
    """Move away from a boundary only within the original visible floor patch."""
    origin = np.asarray(candidate["navmesh_point"], dtype=float)
    options = []
    rings = math.ceil(radius / 0.025)
    for ring in range(1, rings + 1):
        checkpoint(phase="goal_adjustments")
        distance = radius * ring / rings
        for index in range(16):
            angle = 2 * math.pi * index / 16
            desired = origin + [distance * math.cos(angle), 0, distance * math.sin(angle)]
            try:
                point = surface_point(
                    mesh, desired, horizontal_limit=1e-4, vertical_limit=guard.profile.max_height_step_m
                )
                shift = float(np.linalg.norm(point - origin))
                if shift > radius + 1e-6:
                    continue
                uv = project(point)
                clearance = guard.clearance(point)
                if (
                    not inside_bbox(uv, bbox)
                    or clearance <= candidate["clearance_m"] + 1e-6
                    or clearance + 1e-6 < guard.profile.clearance_m
                ):
                    continue
                surface = surface_segment(mesh, origin, point, guard.profile)
                if any(not inside_bbox(project(p), bbox) for p in surface):
                    continue
                if target_valid is not None and not target_valid(point):
                    continue
            except ValueError:
                continue
            grounding = {
                "source": candidate.get("grounding_checks"),
                "checked_source_navmesh_point": origin.tolist(),
                "adjustment_m": shift,
                "adjustment_radius_m": radius,
                "adjusted_surface_connection_verified": True,
                "adjusted_goal_inside_original_box": True,
            }
            options.append(
                dict(
                    candidate,
                    navmesh_point=point.tolist(),
                    projected_uv=uv,
                    clearance_m=clearance,
                    original_navmesh_point=origin.tolist(),
                    grounding_checks=grounding,
                    original_grounding={
                        key: candidate[key]
                        for key in ["world_point", "snap_distance_m", "grounding_checks", "geodesic_m"]
                        if key in candidate
                    },
                    adjustment_m=shift,
                    adjustment_mode="bounded_surface_clearance",
                )
            )
    options.sort(key=lambda c: (-min(c["clearance_m"], 0.15), c["adjustment_m"], c["navmesh_point"]))
    return options


def select_region_plan(
    region,
    *,
    mesh,
    guard,
    shortest,
    start,
    project,
    plan,
    preview,
    goal_tolerance,
    max_distance=None,
    min_distance=0.0,
    adjustment_radius=0.15,
    max_candidates=12,
    max_previews=64,
    adjustments_per_candidate=4,
    target_valid=None,
):
    """No environment execution callback is accepted; failures remain as evidence."""
    if (
        not math.isfinite(adjustment_radius)
        or not 0 <= adjustment_radius <= 0.3
        or any(
            type(n) is not int or n < 1
            for n in [max_candidates, max_previews, adjustments_per_candidate]
        )
        or not math.isfinite(goal_tolerance)
        or goal_tolerance <= 0
    ):
        raise ValueError("invalid bounded region selection limits")
    validate_distance_limits(min_distance, max_distance)
    report = {
        "bbox": region["bbox"],
        "selected": None,
        "planning_attempts": [],
        "box_expanded": False,
        "movement_performed_during_selection": False,
        "adjustment_radius_m": adjustment_radius,
        "max_previews": max_previews,
        "max_candidates": max_candidates,
        "adjustments_per_candidate": adjustments_per_candidate,
        "distance_policy": {"min_geodesic_m": min_distance, "max_geodesic_m": max_distance},
    }

    def evaluate(candidate, rank):
        checkpoint(phase="region_candidate")
        record = {"candidate_rank": rank, "candidate": candidate, "status": "rejected"}
        report["planning_attempts"].append(record)
        try:
            goal = candidate["navmesh_point"]
            if not inside_bbox(project(goal), region["bbox"]):
                raise ValueError("endpoint leaves original box")
            if target_valid is not None and not target_valid(goal):
                raise ValueError("endpoint lacks visible safe floor support inside original box")
            if not mesh.is_navigable(goal) or guard.clearance(goal) + 1e-6 < guard.profile.clearance_m:
                raise ValueError("endpoint lacks execution clearance")
            route = shortest(start, goal)
            distance = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
            if not distance_allowed(distance, min_distance, max_distance):
                raise ValueError("outside configured distance or disconnected")
            candidate = dict(candidate, geodesic_m=distance)
            if candidate.get("adjustment_m") and "world_point" in candidate:
                candidate["snap_distance_m"] = math.dist(candidate["world_point"], goal)
            record["candidate"] = candidate
            prepared = plan(goal)
            record["plan"] = prepared
            if prepared["status"] != "feasible":
                raise ValueError("endpoint planning infeasible: " + prepared["reason"])
            # Region selection owns adjustment; planners must preserve this exact target.
            if math.dist(prepared["fixed_goal"], goal) > 1e-6:
                raise ValueError("planner changed region endpoint")
            forecast = preview(goal, prepared["path"])
            record["preview"] = forecast
            record["execution_budget_exhausted"] = forecast["termination"] == "budget_exhausted"
            if forecast["termination"] != "expert_stop" or forecast["collision_steps"]:
                raise ValueError("read-only preview rejected execution")
            rows = forecast["rows"]
            if not rows or math.dist(rows[0]["world_position"], start) > 0.001:
                raise ValueError("preview does not start at requested pose")
            if not all(
                guard.edge_clear(a["world_position"], b["world_position"])
                for a, b in zip(rows, rows[1:])
            ):
                raise ValueError("preview lacks swept execution clearance")
            end = rows[-1]["world_position"]
            record["predicted_stop_uv"] = project(end)
            if not inside_bbox(record["predicted_stop_uv"], region["bbox"]):
                raise ValueError("predicted stop leaves original box")
            if math.dist(end, goal) > goal_tolerance:
                raise ValueError("predicted stop exceeds goal tolerance")
            record["status"] = "feasible"
            report.update(selected=candidate, plan=prepared, preview=forecast, status="feasible")
            return True
        except ValueError as error:
            record["reason"] = str(error)
            return False

    for rank, candidate in enumerate(region["candidates"][:max_candidates], 1):
        if len(report["planning_attempts"]) >= max_previews:
            break
        if evaluate(candidate, rank):
            return report
        for adjusted in local_adjustments(
            candidate, mesh, guard, project, region["bbox"], adjustment_radius, target_valid
        )[:adjustments_per_candidate]:
            if len(report["planning_attempts"]) >= max_previews:
                break
            if evaluate(adjusted, rank):
                return report
    report.update(
        status="infeasible", reason="no candidate passed bounded planning and controller preview"
    )
    return report
