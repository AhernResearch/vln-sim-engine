"""Train-only NavMesh start pools. CPU PathFinder only; no Simulator or movement."""

from __future__ import annotations

import math
import random
import uuid
from pathlib import Path

from sim_engine.data import EpisodeStore, sha256
from sim_engine.scenes import resolve_scene

from .storage import write

MIN_CLEARANCE_M = 0.15
MIN_ISLAND_AREA_M2 = 2.0
POSE_CELL_M = 0.25
YAW_BINS = 12


def allocate(areas, total, minimum):
    if not areas or total < minimum * len(areas) or any(a <= 0 for a in areas):
        raise ValueError("invalid areas or insufficient total for scene floor")
    remaining = total - minimum * len(areas)
    exact = [remaining * a / sum(areas) for a in areas]
    result = [minimum + math.floor(v) for v in exact]
    order = sorted(range(len(areas)), key=lambda i: (-(exact[i] % 1), i))
    for i in order[: total - sum(result)]:
        result[i] += 1
    return result


def sample(pathfinder, count, seed, *, clearance=MIN_CLEARANCE_M, cell_m=POSE_CELL_M,
           yaw_bins=YAW_BINS, minimum_island_area=MIN_ISLAND_AREA_M2):
    rng = random.Random(seed)
    pathfinder.seed(seed)
    areas = [pathfinder.island_area(i) for i in range(pathfinder.num_islands)]
    seen = set()
    accepted = []
    for draw in range(max(10000, count * 100)):
        p = pathfinder.get_random_navigable_point()
        if not all(math.isfinite(float(x)) for x in p) or not pathfinder.is_navigable(p):
            continue
        island = pathfinder.get_island(p)
        if island < 0 or areas[island] < minimum_island_area:
            continue
        if pathfinder.distance_to_closest_obstacle(p, 0.5) < clearance:
            continue
        yaw_bin = rng.randrange(yaw_bins)
        cell = tuple(math.floor(float(x) / cell_m) for x in p)
        key = (*cell, yaw_bin)
        if key in seen:
            continue
        seen.add(key)
        yaw = -math.pi + (yaw_bin + rng.random()) * (2 * math.pi / yaw_bins)
        accepted.append(
            {
                "position": [float(x) for x in p],
                "rotation": [0.0, math.sin(yaw / 2), 0.0, math.cos(yaw / 2)],
                "island": island,
                "draw": draw,
                "pose_cell_yaw_bin": list(key),
            }
        )
        if len(accepted) == count:
            return accepted
    raise ValueError(
        f"start pool exhausted: accepted {len(accepted)}/{count}; no replacement or movement"
    )


def eligible_area(pathfinder):
    areas = (pathfinder.island_area(i) for i in range(pathfinder.num_islands))
    return sum(a for a in areas if a >= MIN_ISLAND_AREA_M2)


def build_pool(settings, config_sha256, dataset, camera, output: Path, owner, *, successes,
               minimum_per_scene, attempt_factor, seed):
    """Freeze per-scene start pools and their allocation into a new, empty directory."""
    import habitat_sim as hs

    uuid.UUID(hex=owner)
    if settings.collection is None or dataset not in settings.collection.training_datasets:
        raise ValueError("start pools require a configured collection training dataset")
    if camera not in settings.cameras:
        raise ValueError("unknown camera")
    store = EpisodeStore(settings)
    scenes = sorted({r["scene_id"] for r in store.rows[dataset].values()})
    output.mkdir(parents=True, exist_ok=False)
    write(output / "owner.json", {"owner": owner, "scope": "train_navmesh_start_pool",
                                  "config_sha256": config_sha256, "model_requests": 0,
                                  "movement_commands": 0})
    finders = []
    for scene in scenes:
        navmesh = resolve_scene(settings, dataset, scene).navmesh
        pf = hs.PathFinder()
        if not pf.load_nav_mesh(str(navmesh)):
            raise ValueError("cannot load pinned NavMesh")
        finders.append((scene, navmesh, pf))
    areas = [eligible_area(pf) for _, _, pf in finders]
    goals = allocate(areas, successes, minimum_per_scene)
    rows = []
    for i, ((scene, navmesh, pf), goal, area) in enumerate(zip(finders, goals, areas)):
        cap = math.ceil(goal * attempt_factor)
        pool = output / f"scene-{i:03d}.json"
        write(pool, {"owner": owner, "scene_id": scene, "seed": seed + i,
                     "navmesh_sha256": sha256(navmesh), "minimum_clearance_m": MIN_CLEARANCE_M,
                     "minimum_island_area_m2": MIN_ISLAND_AREA_M2, "pose_cell_m": POSE_CELL_M,
                     "yaw_bins": YAW_BINS, "poses": sample(pf, cap, seed + i)})
        rows.append({"dataset": dataset, "scene_id": scene, "camera": camera, "success_goal": goal,
                     "attempts": cap, "eligible_navmesh_area_m2": area, "pool": pool.name,
                     "pool_sha256": sha256(pool)})
    manifest = {
        "owner": owner, "split": "train", "total_success_goal": sum(goals),
        "total_attempt_cap": sum(r["attempts"] for r in rows),
        "allocation": "minimum_per_scene_plus_eligible_navmesh_area",
        "minimum_per_scene": minimum_per_scene, "attempt_factor": attempt_factor, "seed": seed,
        "scenes": rows, "dataset_identity": store.scene_hashes[dataset],
        "habitat_version": hs.__version__, "model_requests": 0, "movement_commands": 0,
    }
    write(output / "manifest.json", manifest)
    return {key: manifest[key] for key in ("total_success_goal", "total_attempt_cap")}
