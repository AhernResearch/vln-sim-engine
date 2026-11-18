"""Train-only NavMesh start pools. CPU PathFinder only; no Simulator or movement."""

from __future__ import annotations

import argparse
import math
import random
import uuid
from pathlib import Path

from sim_engine.config import Settings
from sim_engine.data import EpisodeStore, sha256
from sim_engine.scenes import resolve_scene

from .storage import write


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


def sample(
    pathfinder, count, seed, *, clearance=0.15, cell_m=0.25, yaw_bins=12, minimum_island_area=2.0
):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--successes", type=int, default=100000)
    parser.add_argument("--minimum-per-scene", type=int, default=500)
    parser.add_argument("--attempt-factor", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()
    uuid.UUID(hex=args.owner)
    settings = Settings.read(args.config)
    if settings.collection is None or settings.collection.training_datasets != ["train_pilot"]:
        raise ValueError("start pools require an explicit train_pilot collection deployment")
    import habitat_sim as hs

    store = EpisodeStore(settings)
    scenes = sorted({r["scene_id"] for r in store.rows["train_pilot"].values()})
    args.output.mkdir(parents=True, exist_ok=False)
    write(
        args.output / "owner.json",
        {
            "owner": args.owner,
            "scope": "train_navmesh_start_pool",
            "config_sha256": sha256(args.config),
            "model_requests": 0,
            "movement_commands": 0,
        },
    )
    areas = []
    for scene in scenes:
        pf = hs.PathFinder()
        if not pf.load_nav_mesh(str(resolve_scene(settings, "train_pilot", scene).navmesh)):
            raise ValueError("cannot load pinned NavMesh")
        areas.append(sum(pf.island_area(i) for i in range(pf.num_islands) if pf.island_area(i) >= 2.0))
    goals = allocate(areas, args.successes, args.minimum_per_scene)
    rows = []
    for i, (scene, goal, area) in enumerate(zip(scenes, goals, areas)):
        pf = hs.PathFinder()
        navmesh = resolve_scene(settings, "train_pilot", scene).navmesh
        if not pf.load_nav_mesh(str(navmesh)):
            raise ValueError("cannot reload pinned NavMesh")
        cap = math.ceil(goal * args.attempt_factor)
        poses = sample(pf, cap, args.seed + i)
        pool = args.output / f"scene-{i:03d}.json"
        write(
            pool,
            {
                "owner": args.owner,
                "scene_id": scene,
                "seed": args.seed + i,
                "navmesh_sha256": sha256(navmesh),
                "minimum_clearance_m": 0.15,
                "minimum_island_area_m2": 2.0,
                "pose_cell_m": 0.25,
                "yaw_bins": 12,
                "poses": poses,
            },
        )
        row = {
            "dataset": "train_pilot",
            "scene_id": scene,
            "camera": "visible_88cm_hfov120",
            "success_goal": goal,
            "attempts": cap,
            "eligible_navmesh_area_m2": area,
            "pool": pool.name,
            "pool_sha256": sha256(pool),
        }
        rows.append(row)
        print(f"{i + 1}/{len(scenes)} {scene} goal={goal} starts={cap}", flush=True)
    write(
        args.output / "manifest.json",
        {
            "owner": args.owner,
            "split": "train",
            "total_success_goal": sum(goals),
            "total_attempt_cap": sum(r["attempts"] for r in rows),
            "allocation": "minimum_per_scene_plus_eligible_navmesh_area",
            "minimum_per_scene": args.minimum_per_scene,
            "attempt_factor": args.attempt_factor,
            "seed": args.seed,
            "scenes": rows,
            "dataset_identity": store.scene_hashes["train_pilot"],
            "habitat_version": hs.__version__,
            "model_requests": 0,
            "movement_commands": 0,
        },
    )


if __name__ == "__main__":
    main()
