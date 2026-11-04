"""Derive benchmark-specific navmeshes without modifying source scene assets."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .atomic import atomic_json
from .data import dataset_rows, scene_path, sha256
from .habitat import HABITAT_VERSION, HabitatEnvironment


def prepare_navmeshes(settings, name, source_root: Path, output: Path):
    if source_root.resolve() == output.resolve() or (output.exists() and any(output.iterdir())):
        raise ValueError("derived navmesh output must be a separate empty directory")
    output.mkdir(parents=True, exist_ok=True)
    config = settings.model_copy(deep=True)
    dataset = config.simulation.datasets[name]
    dataset.scenes_root = source_root
    physics = dataset.physics
    rows = dataset_rows(dataset)
    examples = {r["scene_id"]: dict(r, _dataset=name) for r in rows}
    env = HabitatEnvironment(config)
    evidence = {
        "habitat_version": HABITAT_VERSION,
        "dataset": name,
        "annotations_sha256": sha256(dataset.episodes),
        "physics": physics.model_dump(),
        "scenes": {},
    }
    try:
        for scene, row in sorted(examples.items()):
            env.reset(row, next(iter(config.cameras.values())), 0)
            mesh = env.hs.NavMeshSettings()
            mesh.set_defaults()
            mesh.agent_height, mesh.agent_radius = physics.agent_height_m, physics.agent_radius_m
            if not env.sim.recompute_navmesh(env.sim.pathfinder, mesh):
                raise RuntimeError("navmesh reconstruction failed: " + scene)
            source = scene_path(source_root, scene)
            geometry = scene_path(output, scene)
            geometry.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, geometry)
            except OSError:
                shutil.copy2(source, geometry)
            target = geometry.with_suffix(".navmesh")
            if not env.sim.pathfinder.save_nav_mesh(str(target)):
                raise RuntimeError("failed to save derived navmesh: " + scene)
            evidence["scenes"][scene] = {
                "geometry_sha256": sha256(source),
                "source_navmesh_sha256": sha256(source.with_suffix(".navmesh")),
                "derived_navmesh_sha256": sha256(target),
                "settings": {
                    key: getattr(mesh, key)
                    for key in dir(mesh)
                    if not key.startswith("_") and isinstance(getattr(mesh, key), (int, float, bool))
                },
            }
            print("Prepared " + scene, flush=True)
    finally:
        env.close()
    atomic_json(output / "navmesh-provenance.json", evidence)
    return evidence


def audit_connectivity(settings, name):
    import habitat_sim as hs

    if hs.__version__ != HABITAT_VERSION:
        raise RuntimeError(f"habitat-sim {HABITAT_VERSION} is required")
    dataset = settings.simulation.datasets[name]
    root = dataset.scenes_root or settings.simulation.scenes_root
    rows = dataset_rows(dataset)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scene_id"], []).append(row)
    failures = []
    for scene, records in sorted(grouped.items()):
        finder = hs.PathFinder()
        if not finder.load_nav_mesh(str(scene_path(root, scene).with_suffix(".navmesh"))):
            raise RuntimeError("cannot load navmesh: " + scene)
        for row in records:
            path = hs.ShortestPath()
            path.requested_start = row["start_position"]
            path.requested_end = row["goals"][0]["position"]
            if not finder.find_path(path):
                failures.append({"episode_id": str(row["episode_id"]), "scene_id": scene})
    return {
        "dataset": name,
        "episodes": len(rows),
        "unreachable": failures,
        "ready": not failures,
        "physics": dataset.physics.model_dump(),
    }
