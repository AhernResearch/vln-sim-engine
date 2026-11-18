"""Bounded, typed task input shared by collection and planning. No paths, argv or credentials."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from sim_engine.config import StrictModel
from sim_engine.navigation.visual_region import validate_bbox
from sim_engine.protocol import FiniteFloat, Identifier

Point = tuple[FiniteFloat, FiniteFloat, FiniteFloat]


class TaskRequest(StrictModel):
    request_id: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    owner: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    kind: Literal["short_instructions", "r2r_chunks", "reference_geometry"]
    dataset: Identifier
    scene_id: Annotated[str, Field(min_length=1, max_length=512)]
    camera: Identifier
    seed: StrictInt = 0
    instruction: Annotated[str, Field(max_length=32000)] = ""
    start_position: Point
    start_rotation: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    reference_guides: Annotated[list[Point], Field(max_length=4096)] = Field(default_factory=list)
    goal_position: Point | None = None
    reference_fraction: Literal[0.0, 0.5] = 0.0
    offset: Point = (0.0, 0.0, 0.0)
    yaw_delta: FiniteFloat = 0.0

    @model_validator(mode="after")
    def geometry_contract(self):
        if abs(sum(v * v for v in self.start_rotation) - 1.0) > 1e-5:
            raise ValueError("unit xyzw start rotation required")
        if self.kind == "short_instructions":
            if self.reference_guides or self.goal_position is not None or self.instruction:
                raise ValueError("free collection cannot have an assigned route, goal or instruction")
        elif self.kind == "reference_geometry":
            if len(self.reference_guides) < 2:
                raise ValueError("reference geometry requires at least two ordered points")
        elif not self.reference_guides or self.goal_position is None or not self.instruction.strip():
            raise ValueError("R2R collection requires private guides, goal and local instruction")
        if math.sqrt(sum(v * v for v in self.offset)) > 0.5 or abs(self.yaw_delta) > math.pi / 4:
            raise ValueError("R2R initial perturbation exceeds the frozen sampling bounds")
        return self


class RegionRequest(StrictModel):
    target_view: Literal["front", "right", "back", "left"]
    bbox: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]

    @model_validator(mode="after")
    def region_contract(self):
        validate_bbox(self.bbox)
        return self
