from __future__ import annotations

from pathlib import Path

from .config import Dataset, Settings
from .core.io import file_sha256, read_json
from .scenes import scene_source
from .tasks import TASKS


def dataset_rows(dataset: Dataset) -> list[dict]:
    """Episodes of one dataset, narrowed to the configured instruction language."""
    rows = read_json(dataset.episodes)["episodes"]
    if dataset.language:
        rows = [
            row
            for row in rows
            if row["instruction"].get("language", "").split("-")[0] == dataset.language
        ]
    if not rows:
        raise ValueError("empty dataset")
    return rows


class EpisodeStore:
    def __init__(self, settings: Settings):
        self.rows = {}
        self.metadata = {}
        self.scene_hashes = {}
        for name, dataset in settings.simulation.datasets.items():
            scene_hashes = self.scene_hashes[name] = {}
            source = scene_source(settings, name)
            references = read_json(dataset.reference_paths) if dataset.reference_paths else None
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
                if row["scene_id"] not in scene_hashes:
                    scene = source.resolve(row["scene_id"])
                    identity = scene.identity(row["scene_id"])
                    # Scan packages report a nested asset identity; mesh scenes are flat.
                    if scene.assets:
                        scene_hashes[row["scene_id"]] = identity
                    else:
                        scene_hashes.update(identity)
                TASKS[dataset.task].validate(row, settings.simulation.reward_mode)
                records[key] = row
            self.rows[name] = records
            self.metadata[name] = {
                "task": dataset.task,
                "episodes": len(records),
                "annotations_sha256": file_sha256(dataset.episodes),
                "reference_sha256": file_sha256(dataset.reference_paths) if references else None,
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
        "weights": [{"file": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in weights],
        "processor_files": {p.name: file_sha256(p) for p in sorted(root.glob("*.json"))},
    }
