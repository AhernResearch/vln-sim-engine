from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator

from .config import StrictModel

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]


class ResetRequest(StrictModel):
    dataset: Identifier
    episode_id: Identifier
    camera: Identifier = "trained_ce30"
    seed: StrictInt = 0


class StepRequest(StrictModel):
    request_id: Identifier
    expected_event_index: Annotated[StrictInt, Field(ge=0)]
    trajectory: Annotated[list[tuple[float, float, float]], Field(min_length=8, max_length=8)]
    action_budget: Annotated[StrictInt, Field(ge=1, le=4)] = 4
    policy_version: str = ""

    @field_validator("trajectory", mode="before")
    @classmethod
    def finite_numeric_points(cls, value):
        if not isinstance(value, list) or any(
            not isinstance(p, (list, tuple))
            or len(p) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in p)
            for p in value
        ):
            raise ValueError("trajectory must contain finite numeric SE(2) points")
        return value


class FinishRequest(StrictModel):
    reason: Literal["policy_stop", "invalid_output", "cancelled"]


class Observation(StrictModel):
    session_id: Identifier
    event_index: Annotated[StrictInt, Field(ge=0)]


class GenerateRequest(StrictModel):
    request_id: Identifier
    instruction: Annotated[str, Field(min_length=1, max_length=32000)]
    camera: Identifier = "trained_ce30"
    observations: Annotated[list[Observation], Field(min_length=1, max_length=8)]
    temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    seed: StrictInt = 0
    max_tokens: Annotated[StrictInt, Field(ge=32, le=512)] = 512

    @field_validator("observations")
    @classmethod
    def causal_history(cls, values):
        if len({v.session_id for v in values}) != 1:
            raise ValueError("history must come from one session")
        indices = [v.event_index for v in values]
        if indices != sorted(set(indices)):
            raise ValueError("history indices must be unique and increasing")
        return values
