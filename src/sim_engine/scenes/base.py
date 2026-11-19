"""Scene sources resolve a dataset's scene ID to immutable render and NavMesh assets.

Two sources exist: plain mesh scenes under a scenes root, and verified scan packages.
The configured runtime chooses which one a dataset must use; resolution never loads CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sim_engine.backends.runtime import MESH_SCENES, SCAN_PACKAGES, runtime
from sim_engine.core.io import file_sha256


@dataclass(frozen=True)
class Scene:
    render: Path
    navmesh: Path
    # Content identity recorded in health and receipts, keyed as the source defines it.
    assets: dict[str, str] = field(default_factory=dict)
    geometry_review: str = "legacy"
    # Agent geometry the NavMesh was baked for, when the source records it.
    navigation_settings: dict | None = None

    def identity(self, scene_id: str) -> dict[str, str]:
        return self.assets or {scene_id: file_sha256(self.render),
                               str(Path(scene_id).with_suffix(".navmesh")): file_sha256(self.navmesh)}


def scene_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("scene must be relative to configured scenes_root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("scene escapes configured scenes_root")
    return resolved


class MeshSource:
    """``<root>/<scene_id>`` with a sibling ``.navmesh``."""

    kind = MESH_SCENES

    def __init__(self, root: Path):
        self.root = root

    def resolve(self, scene_id: str) -> Scene:
        render = scene_path(self.root, scene_id)
        return Scene(render, render.with_suffix(".navmesh"))


class ScanPackageSource:
    """Explicit scene ID -> verified ``scene.json`` with a baked ``navigation/`` receipt."""

    kind = SCAN_PACKAGES

    def __init__(self, root: Path, manifests: dict[str, Path]):
        self.root, self.manifests = root, manifests

    def resolve(self, scene_id: str) -> Scene:
        from .package import asset_path, load_scan_package

        if scene_id not in self.manifests:
            raise ValueError("scan scene is not registered for this dataset: " + scene_id)
        manifest_path = self.manifests[scene_id]
        if not manifest_path.is_absolute():
            manifest_path = scene_path(self.root, str(manifest_path))
        path, manifest, receipt, _ = load_scan_package(manifest_path)
        if manifest["scene_id"] != scene_id:
            raise ValueError("scene package and registered scene_id differ")
        navigation = path.parent / "navigation"
        render = asset_path(path.parent, manifest["assets"]["stage"]["path"])
        navmesh = asset_path(navigation, receipt["assets"]["navmesh"]["path"])
        assets = {"manifest": file_sha256(path), "navigation": file_sha256(navigation / "receipt.json")}
        assets.update({key: value["sha256"] for key, value in manifest["assets"].items()})
        assets["navmesh"] = file_sha256(navmesh)
        return Scene(render, navmesh, assets, "required", receipt.get("settings"))


def scene_source(settings, dataset_name: str):
    """The single source a dataset uses under the configured runtime."""
    dataset = settings.simulation.datasets[dataset_name]
    root = dataset.scenes_root or settings.simulation.scenes_root
    kind = runtime(settings.simulation.backend).scene_source
    if kind == SCAN_PACKAGES:
        if not dataset.scenes:
            raise ValueError(f"{settings.simulation.backend} requires explicit frozen scene packages")
        return ScanPackageSource(root, dataset.scenes)
    if dataset.scenes:
        raise ValueError(f"scene packages require a {SCAN_PACKAGES} runtime, not "
                         + settings.simulation.backend)
    return MeshSource(root)


def resolve_scene(settings, dataset_name: str, scene_id: str) -> Scene:
    return scene_source(settings, dataset_name).resolve(scene_id)
