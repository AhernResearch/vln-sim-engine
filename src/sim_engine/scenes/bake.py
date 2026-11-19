"""NavMesh baking adapted from open-nav-harness-tools scan_probe (Apache-2.0)."""
import json
import math
import uuid
from pathlib import Path

from sim_engine.backends.runtime import HABITAT_GS
from sim_engine.core.io import create_json, file_sha256

from .package import asset_path, verify_manifest

# Fixed Recast resolution and climb limits; agent geometry is a bake argument.
NAVMESH_SETTINGS = {"agent_max_climb": 0.2, "agent_max_slope": 45.0,
                    "cell_size": 0.05, "cell_height": 0.025}


def select_start(pathfinder, poses):
    """Choose a scan-supported start with the most NavMesh boundary clearance."""
    import numpy as np

    candidates, distances = [], []
    sampled = poses[::10]
    for position in sampled:
        # Magnum's Python sequence caster rejects mixed float/int coordinates.
        point = np.asarray(position, dtype=np.float32)
        snapped = np.asarray(pathfinder.snap_point(point), dtype=np.float32)
        if not np.isfinite(snapped).all():
            continue
        horizontal = math.hypot(snapped[0] - position[0], snapped[2] - position[2])
        height = position[1] - snapped[1]
        distances.append(horizontal)
        if horizontal <= 0.2 and 0.3 <= height <= 2.0:
            clearance = float(pathfinder.distance_to_closest_obstacle(snapped, 2.0))
            if math.isfinite(clearance) and clearance >= 0:
                candidates.append((clearance, snapped.tolist()))
    if not candidates:
        raise RuntimeError("No recorded scan position has a supported development start")
    clearance, start = max(candidates, key=lambda candidate: candidate[0])
    return start, {
        "scan_positions_sampled": len(sampled),
        "supported_start_candidates": len(candidates),
        "horizontal_snap_p90_m": float(np.quantile(distances, 0.9)),
        "start_selection": "maximum_navmesh_clearance_scan_supported",
        "start_clearance_m": clearance,
        "clearance_query_radius_m": 2.0,
    }


def bake(manifest_path, agent_radius_m=0.18, agent_height_m=1.5):
    import habitat_sim

    HABITAT_GS.validate(habitat_sim)

    path = Path(manifest_path).resolve()
    manifest = verify_manifest(path)
    destination = path.parent / "navigation"
    destination.mkdir(exist_ok=False)
    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(asset_path(path.parent, manifest["assets"]["geometry_stage"]["path"]))
    config.create_renderer = False
    config.enable_physics = False
    config.navmesh_settings = None
    settings = habitat_sim.NavMeshSettings()
    settings.set_defaults()
    for key, value in {**NAVMESH_SETTINGS, "agent_height": agent_height_m,
                       "agent_radius": agent_radius_m}.items():
        setattr(settings, key, value)
    settings_path = destination / "settings.json"
    settings.write_to_json(str(settings_path))
    frozen = json.loads(settings_path.read_text())
    with habitat_sim.Simulator(habitat_sim.Configuration(config, [habitat_sim.AgentConfiguration()])) as sim:
        if not sim.recompute_navmesh(sim.pathfinder, settings):
            raise RuntimeError("Cannot build a navigable surface from the scan mesh")
        navmesh = destination / "scene.navmesh"
        sim.pathfinder.save_nav_mesh(str(navmesh))
        poses = json.loads(asset_path(path.parent, manifest["assets"]["scan_poses"]["path"]).read_text())["positions"]
        start, start_diagnostics = select_start(sim.pathfinder, poses)
        episodes = {"purpose": "development", "episodes": [
            {"episode_id": "scan-dev-0", "scene_id": manifest["scene_id"],
             "start_position": start, "start_rotation": [0, 0, 0, 1],
             "instruction": {"instruction_text": "Observe the surroundings, explore a short clear section of the corridor, then stop."}}]}
        create_json(destination / "episodes.json", episodes)
        receipt = {"schema": "open_nav_scan_navigation_v1", "source_manifest_sha256": file_sha256(path),
                   "owner": uuid.uuid4().hex, "purpose": "development", "geometry_review": "required",
                   "settings": frozen, "habitat_version": habitat_sim.__version__,
                   "navigable_area_m2": float(sim.pathfinder.navigable_area),
                   **start_diagnostics,
                   "assets": {key: {"path": name, "sha256": file_sha256(destination / name)}
                              for key, name in (("navmesh", "scene.navmesh"), ("episodes", "episodes.json"))}}
        create_json(destination / "receipt.json", receipt)
    return receipt

