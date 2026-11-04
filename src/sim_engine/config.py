from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Camera(StrictModel):
    height_m: float = Field(default=1.25, gt=0, allow_inf_nan=False)
    pitch_down_rad: float = Field(default=math.pi / 6, ge=-math.pi / 2, le=math.pi / 2)
    view: str = "125cm_30deg"
    width: int = Field(default=640, ge=32)
    height: int = Field(default=480, ge=32)
    hfov: float = Field(default=79, gt=0, lt=180)

    def robot(self) -> dict:
        result = {
            "body_type": "simulated_ground_agent",
            "camera_view": self.view,
            "camera_height_m": self.height_m,
        }
        if self.pitch_down_rad:
            result["camera_pitch_down_rad"] = self.pitch_down_rad
        return result


class VelocityLimits(StrictModel):
    """habitat-lab ``velocity_control`` settings, in its native units."""

    linear_mps: tuple[float, float] = (0.0, 2.5)
    angular_dps: tuple[float, float] = (-300.0, 300.0)
    min_abs_linear_mps: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    min_abs_angular_dps: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    time_step_s: float = Field(default=0.1, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered_ranges(self):
        for low, high in (self.linear_mps, self.angular_dps):
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError("velocity ranges must be finite and increasing")
        return self


class Physics(StrictModel):
    agent_height_m: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    agent_radius_m: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    forward_m: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    turn_degrees: float = Field(default=15, gt=0, le=90, allow_inf_nan=False)
    allow_sliding: bool = True
    velocity: VelocityLimits | None = None


class Dataset(StrictModel):
    episodes: Path
    language: str | None = None
    reference_paths: Path | None = None
    scenes_root: Path | None = None
    physics: Physics = Field(default_factory=Physics)


class Simulation(StrictModel):
    scenes_root: Path
    datasets: dict[str, Dataset]
    gpu: int = Field(default=2, ge=0)
    workers: int = Field(default=2, ge=1, le=32)
    queue_capacity: int = Field(default=32, ge=1, le=1024)
    max_actions: int = Field(default=500, ge=1, le=2000)
    rpc_timeout_seconds: float = Field(default=60, gt=0)
    reward_mode: str = Field(default="none", pattern="^(none|progress_v1)$")


class Policy(StrictModel):
    model: Path
    prompt_profile: Literal["trajectory_v1", "navigation_v2"] = "trajectory_v1"
    sim_url: str = "http://127.0.0.1:18810"
    gpu_memory_utilization: float = Field(default=0.65, gt=0, lt=1)
    max_model_len: int = Field(default=8192, ge=1024)
    max_num_seqs: int = Field(default=4, ge=1)
    max_num_batched_tokens: int = Field(default=4096, ge=1024)
    enforce_eager: bool = True
    max_pixels: int = Field(default=262144, ge=1024)
    image_cache_entries: int = Field(default=128, ge=8)
    max_pending_requests: int = Field(default=32, ge=1)
    max_images: int = Field(default=8, ge=1, le=16)
    lora_adapter: Path | None = None
    max_lora_rank: int = Field(default=16, ge=1, le=256)
    seed: int = 0
    return_logprobs: bool = True


class Settings(StrictModel):
    simulation: Simulation
    policy: Policy
    cameras: dict[str, Camera] = Field(default_factory=lambda: {"trained_ce30": Camera()})
    policy_version: str = "epoch1"

    @classmethod
    def read(cls, path: Path) -> Settings:
        text = os.path.expandvars(path.read_text())
        if "${" in text:
            raise ValueError("unresolved environment variable in configuration")
        return cls.model_validate(json.loads(text))
