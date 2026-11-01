from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from .config import Settings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_file(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return json.load(stream)


def scene_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("scene must be relative to configured scenes_root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("scene escapes configured scenes_root")
    return resolved


class EpisodeStore:
    def __init__(self, settings: Settings):
        self.rows = {}
        self.metadata = {}
        self.scene_hashes = {}
        for name, dataset in settings.simulation.datasets.items():
            root = dataset.scenes_root or settings.simulation.scenes_root
            scene_hashes = self.scene_hashes[name] = {}
            rows = json_file(dataset.episodes)["episodes"]
            if dataset.language:
                rows = [
                    r
                    for r in rows
                    if r["instruction"].get("language", "").split("-")[0] == dataset.language
                ]
            references = json_file(dataset.reference_paths) if dataset.reference_paths else None
            records = {}
            for row in rows:
                row["_dataset"] = name
                key = str(row["episode_id"])
                if key in records:
                    raise ValueError("duplicate episode identity")
                if references is not None:
                    row["reference_path"] = references[key]["locations"]
                geometry = scene_path(root, row["scene_id"])
                if not geometry.is_file() or not geometry.with_suffix(".navmesh").is_file():
                    raise FileNotFoundError(f"missing scene geometry/navmesh: {geometry}")
                for asset in (row["scene_id"], str(Path(row["scene_id"]).with_suffix(".navmesh"))):
                    if asset not in scene_hashes:
                        scene_hashes[asset] = sha256(scene_path(root, asset))
                records[key] = row
            if not records:
                raise ValueError("empty dataset")
            self.rows[name] = records
            self.metadata[name] = {
                "episodes": len(records),
                "annotations_sha256": sha256(dataset.episodes),
                "reference_sha256": sha256(dataset.reference_paths) if references else None,
                "physics": dataset.physics.model_dump(),
            }

    def get(self, dataset: str, episode_id: str) -> dict:
        return self.rows[dataset][episode_id]


def model_identity(root: Path):
    weights = sorted(root.glob("*.safetensors"))
    if not weights:
        raise ValueError("checkpoint has no safetensors weights")
    return {
        "directory": str(root.resolve()),
        "weights": [{"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in weights],
        "processor_files": {p.name: sha256(p) for p in sorted(root.glob("*.json"))},
    }
