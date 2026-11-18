"""Versioned jobs contain logical scene IDs and verified blobs, never host paths."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from sim_engine.collection.protocol import RegionRequest, TaskRequest
from sim_engine.config import CURRENT_PLANNING_PROFILE, Camera, PlanningVersion, StrictModel

Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Identity = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]


def fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


class Blob(StrictModel):
    sha256: Digest
    size: int = Field(ge=0, le=32 * 1024**2)
    media_type: Literal["application/json", "application/octet-stream"]
    data: str = Field(max_length=44 * 1024**2)

    def decode(self):
        content = base64.b64decode(self.data, validate=True)
        if len(content) != self.size or hashlib.sha256(content).hexdigest() != self.sha256:
            raise ValueError("artifact digest or size mismatch")
        return content

    @classmethod
    def encode(cls, content, name):
        return cls(sha256=hashlib.sha256(content).hexdigest(), size=len(content),
                   media_type="application/json" if name.endswith(".json") else "application/octet-stream",
                   data=base64.b64encode(content).decode())


class PlanningBudget(StrictModel):
    wall_s: float = Field(default=120, gt=0, le=900, allow_inf_nan=False)
    max_expansions: int = Field(default=80000, ge=1, le=1000000)
    max_queries: int = Field(default=2000000, ge=1, le=20000000)


class PlanningProfile(StrictModel):
    version: PlanningVersion = CURRENT_PLANNING_PROFILE
    base_steps: int = Field(default=160, ge=1, le=10000)

    def geometry(self, task):
        config = {
            "seed": task["seed"], "split": "train",
            "max_steps": self.base_steps if task["kind"] == "short_instructions" else 400,
            "minimum_planning_clearance_m": 0.05, "guide_adjustment_radius_m": 0.25,
            "endpoint_adjustment_radius_m": 0.15, "expert_goal_radius_m": 0.08,
            "goal_tolerance_m": 0.12 if task["kind"] == "short_instructions" else 0.3,
            "pose_sampling_attempts": 32, "start_radius_m": 0.5, "yaw_degrees": 45.0,
        }
        if self.version in {"collection-v3", "collection-v4", "collection-v5"}:
            config["planning_optimizations"] = True
        if self.version in {"collection-v4", "collection-v5"}:
            config["search_backend"] = "native"
        if self.version == "collection-v5":
            config.update(query_optimizations=True, native_path_optimizations=True)
        return config


INPUT_NAMES = {"start-state.json", "camera.json", "view-sensors.json",
               "front-depth.npz", "right-depth.npz", "back-depth.npz", "left-depth.npz"}
OUTPUT_NAMES = {"start-state.json", "private-geometry.json", "grounded-target.json",
                "grounding-check.json", "reference-check.json", "preflight.json",
                "execution-budget.json", "region-candidates.json", "region-selection.json",
                "visible-floor-mask.npz", "visible-floor-mask.json", "expert-profile.json",
                "reference-state.json", "pose-selection.json"}


class PlanRequest(StrictModel):
    schema_version: Literal[1] = 1
    attempt_id: Identity
    task: TaskRequest
    navmesh_sha256: Digest
    camera: Camera
    profile: PlanningProfile = Field(default_factory=PlanningProfile)
    budget: PlanningBudget = Field(default_factory=PlanningBudget)
    target: RegionRequest | None = None
    artifacts: dict[str, Blob] = Field(default_factory=dict, max_length=7)

    @model_validator(mode="after")
    def input_contract(self):
        if self.camera.width * self.camera.height > 1024 * 1024:
            raise ValueError("planning camera exceeds pixel budget")
        if set(self.artifacts) - INPUT_NAMES:
            raise ValueError("unsupported planning input artifact")
        if sum(b.size for b in self.artifacts.values()) > 16 * 1024**2:
            raise ValueError("planning input exceeds byte limit")
        if self.task.kind == "short_instructions":
            if self.target is None:
                raise ValueError("free-region planning requires a target")
            expected = {"start-state.json", "camera.json", "view-sensors.json",
                        self.target.target_view + "-depth.npz"}
            if set(self.artifacts) != expected:
                raise ValueError("free-region input must include only the chosen depth view")
        elif self.target or self.artifacts:
            raise ValueError("R2R/reference planning requires no observation artifacts")
        for blob in self.artifacts.values():
            blob.decode()
        return self

    def digest(self):
        value = self.model_dump(mode="json")
        value["artifacts"] = {k: {x: y for x, y in v.items() if x != "data"}
                              for k, v in value["artifacts"].items()}
        return fingerprint(value)


class NavMeshAsset(StrictModel):
    dataset: str
    scene_id: str
    path: Path
    sha256: Digest


class PlannerSettings(StrictModel):
    root: Path
    assets: list[NavMeshAsset] = Field(min_length=1)
    workers: int = Field(default=4, ge=1, le=64)
    capacity: int = Field(default=64, ge=1, le=4096)
    scene_cache_entries: int = Field(default=4, ge=1, le=64)
    hard_timeout_grace_s: float = Field(default=3, gt=0, le=30)

    @classmethod
    def read(cls, path):
        path = Path(path).resolve()
        value = cls.model_validate(json.loads(path.read_text()))
        value.root = (path.parent / value.root).resolve()
        for asset in value.assets:
            asset.path = (path.parent / asset.path).resolve()
        return value

    @model_validator(mode="after")
    def unique_assets(self):
        keys = [(x.dataset, x.scene_id) for x in self.assets]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate NavMesh identity")
        return self
