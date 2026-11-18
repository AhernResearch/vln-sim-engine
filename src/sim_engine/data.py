from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from .config import Dataset, Settings


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


def dataset_rows(dataset: Dataset) -> list[dict]:
    """Episodes of one dataset, narrowed to the configured instruction language."""
    rows = json_file(dataset.episodes)["episodes"]
    if dataset.language:
        rows = [
            row
            for row in rows
            if row["instruction"].get("language", "").split("-")[0] == dataset.language
        ]
    if not rows:
        raise ValueError("empty dataset")
    return rows


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
            scene_hashes = self.scene_hashes[name] = {}
            references = json_file(dataset.reference_paths) if dataset.reference_paths else None
            records = {}
            for row in dataset_rows(dataset):
                row["_dataset"] = name
                row.pop("_metric_reference_path", None)
                key = str(row["episode_id"])
                if key in records:
                    raise ValueError("duplicate episode identity")
                if references is not None:
                    row["reference_path"] = references[key]["locations"]
                    row["_metric_reference_path"] = references[key]["locations"]
                from .scenes import resolve_scene

                if row["scene_id"] not in scene_hashes:
                    scene = resolve_scene(settings, name, row["scene_id"])
                    if scene.manifest:
                        scene_hashes[row["scene_id"]] = scene.assets
                    else:
                        scene_hashes.update({row["scene_id"]: sha256(scene.render),
                                             str(Path(row["scene_id"]).with_suffix(".navmesh")):
                                             sha256(scene.navmesh)})
                row["_task"] = dataset.task
                if dataset.task == "development" and row.get("goals"):
                    raise ValueError("development episodes must not contain scoring goals")
                if dataset.task == "development" and settings.simulation.reward_mode != "none":
                    raise ValueError("development tasks cannot enable goal rewards")
                records[key] = row
            self.rows[name] = records
            self.metadata[name] = {
                "task": dataset.task,
                "episodes": len(records),
                "annotations_sha256": sha256(dataset.episodes),
                "reference_sha256": sha256(dataset.reference_paths) if references else None,
                "physics": dataset.physics.model_dump(),
                "ndtw_reference": "explicit_gt_locations" if references is not None else "unavailable",
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
