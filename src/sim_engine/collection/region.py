"""Plan visible-region targets on the simulator; no language model dependency."""

import math

import numpy as np

from .guidance import plan_with_preferences
from .limits import execution_budget
from .region_selection import select_region_plan
from .safe_image import safe_floor_mask
from .storage import write
from .tracking import preview
from .visual_region import project_world, sample_region


def preview_target(backend, config, episode, goal, path):
    policy = config.get("execution_budget_policy", {})
    if policy.get("mode", "path_scaled") != "path_scaled":
        raise ValueError("unknown visual execution budget policy")
    rotation = episode["start_rotation"]
    yaw = 2 * math.atan2(rotation[1], rotation[3])
    budget = execution_budget(
        path,
        backend.expert_profile,
        yaw,
        config["backend_manifest"]["config"]["max_steps"],
        time_multiplier=policy.get("time_multiplier", 3.0),
        settle_seconds=policy.get("settle_seconds", 10.0),
    )
    result = preview(
        backend.pf,
        dict(episode, goals=[{"position": goal, "radius": 0.08}]),
        path,
        backend.expert_profile,
        max_steps=budget["max_steps"],
    )
    return dict(result, execution_budget=budget)


def ground_region(backend, task, depth, camera, views, start, folder, config, episode):
    if camera.get("depth_convention", "axial_minus_camera_z") != "axial_minus_camera_z":
        raise ValueError("depth projection convention must be validated before region planning")
    mask = None
    target_valid = None
    floor_policy = config.get("visible_floor_policy", {})
    if floor_policy.get("enabled", False):
        mask, visible_goals, clearance, stats = safe_floor_mask(
            depth,
            camera["sensors"]["depth"]["intrinsics_K"],
            views[task["target_view"]],
            start,
            backend.pf,
            backend.guard,
            min_distance=config.get("min_local_geodesic_m", 0.0),
            max_distance=config.get("max_local_geodesic_m"),
            horizontal_limit=floor_policy.get("horizontal_limit", 0.01),
            vertical_limit=floor_policy.get("vertical_limit", 0.2),
            slope_degrees=floor_policy.get("slope_degrees", 25.0),
            bbox=task["bbox"] if floor_policy.get("mask_scope", "full") == "target_bbox" else None,
        )
        extra = {}
        if "computed_pixel_bounds_xyxy" in stats:
            left, top, right, bottom = stats["computed_pixel_bounds_xyxy"]
            computed = np.zeros(mask.shape, dtype=bool)
            computed[top:bottom, left:right] = True
            extra["computed_mask"] = computed
        with (folder / "visible-floor-mask.npz").open("xb") as f:
            np.savez_compressed(f, safe_mask=mask, navmesh_points=visible_goals,
                                clearance_m=clearance, **extra)
        write(folder / "visible-floor-mask.json", stats)
        yy, xx = np.indices(mask.shape)
        x0, y0, x1, y1 = task["bbox"]
        inside = (
            mask & (xx / (depth.shape[1] - 1) * 1000 >= x0) & (xx / (depth.shape[1] - 1) * 1000 <= x1)
        )
        inside &= (yy / (depth.shape[0] - 1) * 1000 >= y0) & (yy / (depth.shape[0] - 1) * 1000 <= y1)
        support = visible_goals[inside]
        tolerance = floor_policy.get("adjustment_support_tolerance_m", 0.025)
        if not math.isfinite(tolerance) or not 0 < tolerance <= 0.05:
            raise ValueError("invalid visible support tolerance")

        def target_valid(p):
            return bool(
                len(support) and np.linalg.norm(support - np.asarray(p), axis=1).min() <= tolerance
            )

    region = sample_region(
        task["bbox"],
        depth,
        camera["sensors"]["depth"]["intrinsics_K"],
        views[task["target_view"]],
        start,
        backend.pf,
        backend.guard,
        backend.shortest,
        grid_size=config["region_grid_size"],
        max_distance=config.get("max_local_geodesic_m"),
        min_distance=config.get("min_local_geodesic_m", 0.0),
        safe_mask=mask,
    )
    write(folder / "region-candidates.json", region)
    selection = select_region_plan(
        region,
        mesh=backend.pf,
        guard=backend.guard,
        shortest=backend.shortest,
        start=start.tolist(),
        project=lambda p: project_world(
            p,
            camera["sensors"]["depth"]["intrinsics_K"],
            views[task["target_view"]],
            backend.camera.width,
            backend.camera.height,
        ),
        plan=lambda goal: plan_with_preferences(
            backend.planners, backend.guard, [start.tolist()], goal, 0.0, 0.0
        ),
        preview=lambda goal, path: preview_target(backend, config, episode, goal, path),
        goal_tolerance=config["backend_manifest"]["config"]["goal_tolerance_m"],
        max_distance=config.get("max_local_geodesic_m"),
        min_distance=config.get("min_local_geodesic_m", 0.0),
        max_candidates=config["region_plan_candidates"],
        adjustment_radius=config.get("region_adjustment_radius_m", 0.15),
        max_previews=config.get("region_max_previews", 64),
        adjustments_per_candidate=config.get("region_adjustments_per_candidate", 4),
        target_valid=target_valid,
    )
    write(folder / "region-selection.json", selection)
    if selection["status"] != "feasible":
        raise ValueError(selection["reason"])
    candidate = selection["selected"]
    goal = candidate["navmesh_point"]
    return (
        dict(candidate, requested_bbox=task["bbox"]),
        np.asarray(goal),
        candidate["grounding_checks"],
        selection["plan"],
        selection["preview"],
    )
