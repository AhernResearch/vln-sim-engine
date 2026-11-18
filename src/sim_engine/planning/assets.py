"""Export a portable training-only NavMesh registry for a CPU planning deployment."""

import shutil
import uuid
from pathlib import Path

from sim_engine.core.io import create_json, file_sha256
from sim_engine.data import EpisodeStore

from .contracts import NavMeshAsset, PlannerSettings


def registered_assets(settings, store):
    assets = []
    for dataset in settings.collection.training_datasets:
        from sim_engine.scenes import resolve_scene

        for scene in sorted(s for s in store.scene_hashes[dataset] if not s.endswith(".navmesh")):
            path = resolve_scene(settings, dataset, scene).navmesh
            assets.append(NavMeshAsset(dataset=dataset, scene_id=scene, path=path, sha256=file_sha256(path)))
    return assets


def export_assets(settings, output, *, store=None):
    if settings.collection is None:
        raise ValueError("asset export requires an explicit training collection configuration")
    source = registered_assets(settings, store or EpisodeStore(settings))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    owner = uuid.uuid4().hex
    create_json(output / "ownership.json", {"owner": owner, "operation": "export_training_navmeshes"})
    assets = []
    for asset in source:
        relative = Path("assets") / (asset.sha256 + ".navmesh")
        destination = output / relative
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copyfile(asset.path, destination)
        if file_sha256(destination) != asset.sha256:
            raise RuntimeError("copied NavMesh does not match pinned source")
        assets.append(asset.model_copy(update={"path": relative}))
    config = PlannerSettings(root=Path("jobs"), assets=assets)
    create_json(output / "planner.json", config.model_dump(mode="json"))
    return {"owner": owner, "scenes": len(assets), "config": str(output / "planner.json")}
