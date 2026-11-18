"""Derive benchmark-specific navmeshes without modifying source scene assets."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from sim_engine.core.io import file_sha256, replace_json

from .data import dataset_rows, scene_path
from .habitat import HABITAT_VERSION, HabitatEnvironment


def prepare_navmeshes(settings, name, source_root: Path, output: Path):
    if settings.simulation.backend != "habitat":
        raise ValueError("GS packages use bake-scan into a new immutable package")
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
        "annotations_sha256": file_sha256(dataset.episodes),
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
                "geometry_sha256": file_sha256(source),
                "source_navmesh_sha256": file_sha256(source.with_suffix(".navmesh")),
                "derived_navmesh_sha256": file_sha256(target),
                "settings": {
                    key: getattr(mesh, key)
                    for key in dir(mesh)
                    if not key.startswith("_") and isinstance(getattr(mesh, key), (int, float, bool))
                },
            }
            print("Prepared " + scene, flush=True)
    finally:
        env.close()
    replace_json(output / "navmesh-provenance.json", evidence)
    return evidence


def audit_connectivity(settings, name):
    import habitat_sim as hs

    from .backends import create_environment
    from .scenes import resolve_scene

    adapter = create_environment(settings)
    adapter.close()
    dataset = settings.simulation.datasets[name]
    rows = dataset_rows(dataset)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["scene_id"], []).append(row)
    failures = []
    for scene, records in sorted(grouped.items()):
        finder = hs.PathFinder()
        if not finder.load_nav_mesh(str(resolve_scene(settings, name, scene).navmesh)):
            raise RuntimeError("cannot load navmesh: " + scene)
        for row in records:
            if not row.get("goals"):
                continue
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
