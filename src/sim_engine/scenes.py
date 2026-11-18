"""Resolve legacy scenes and immutable scan packages without loading CUDA."""
from dataclasses import dataclass
from pathlib import Path

from .core.io import file_sha256


@dataclass(frozen=True)
class Scene:
    render: Path
    navmesh: Path
    assets: dict[str, str]
    manifest: Path | None = None
    geometry_review: str = "legacy"
    navigation_settings: dict | None = None


def resolve_scene(settings, dataset_name, scene_id):
    from .data import scene_path

    dataset = settings.simulation.datasets.get(dataset_name)
    root = (dataset.scenes_root if dataset else None) or settings.simulation.scenes_root
    if dataset and scene_id in dataset.scenes:
        from .scan_assets import asset_path, load_scan_package

        manifest_path = dataset.scenes[scene_id]
        if not manifest_path.is_absolute():
            manifest_path = scene_path(root, str(manifest_path))
        path, manifest, receipt, _ = load_scan_package(manifest_path)
        if settings.simulation.backend != "habitat_gs" or manifest["scene_id"] != scene_id:
            raise ValueError("scene package and runtime backend/scene_id differ")
        render = asset_path(path.parent, manifest["assets"]["stage"]["path"])
        navmesh = asset_path(path.parent / "navigation", receipt["assets"]["navmesh"]["path"])
        assets = {"manifest": file_sha256(path), "navigation": file_sha256(path.parent / "navigation/receipt.json")}
        assets.update({key: value["sha256"] for key, value in manifest["assets"].items()})
        assets["navmesh"] = file_sha256(navmesh)
        return Scene(render, navmesh, assets, path, "required", receipt.get("settings"))
    if settings.simulation.backend == "habitat_gs":
        raise ValueError("GS requires an explicit frozen scene package")
    render = scene_path(root, scene_id)
    navmesh = render.with_suffix(".navmesh")
    return Scene(render, navmesh, {})
