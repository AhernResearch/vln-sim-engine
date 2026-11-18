from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, StrictInt, field_validator, model_validator

from .config import StrictModel

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:-]+$")]


class ResetRequest(StrictModel):
    dataset: Identifier
    episode_id: Identifier
    camera: Identifier = "trained_ce30"
    seed: StrictInt = 0


class ExpertRequest(StrictModel):
    """Run the simulator-side expert on one episode; outputs stay on the server's disk."""

    dataset: Identifier
    episode_id: Identifier
    camera: Identifier = "trained_ce30"


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class WaypointCommand(StrictModel):
    kind: Literal["waypoints"] = "waypoints"
    trajectory: Annotated[list[tuple[float, float, float]], Field(min_length=8, max_length=8)]
    action_budget: Annotated[StrictInt, Field(ge=1, le=4)] = 4

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


class VelocityCommand(StrictModel):
    """Body-frame unicycle velocity held for the dataset's velocity time step."""

    kind: Literal["velocity"]
    linear_mps: FiniteFloat
    angular_rps: FiniteFloat


class StepRequest(StrictModel):
    request_id: Identifier
    expected_event_index: Annotated[StrictInt, Field(ge=0)]
    command: Annotated[WaypointCommand | VelocityCommand, Field(discriminator="kind")]
    policy_version: str = ""

    @model_validator(mode="before")
    @classmethod
    def legacy_waypoints(cls, value):
        """Accept the original top-level trajectory/action_budget body as a waypoint command."""
        if isinstance(value, dict) and "command" not in value and "trajectory" in value:
            value = dict(value)
            command = {"kind": "waypoints", "trajectory": value.pop("trajectory")}
            if "action_budget" in value:
                command["action_budget"] = value.pop("action_budget")
            value["command"] = command
        return value


class FinishRequest(StrictModel):
    """``forced_stop`` is a caller-imposed STOP at its own decision budget, not a model STOP;
    both are scored at the stopped pose, and the reason stays distinguishable."""

    reason: Literal["policy_stop", "forced_stop", "invalid_output", "cancelled"]


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


class RecordedImage(StrictModel):
    sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    data_base64: Annotated[str, Field(min_length=1, max_length=8_000_000)]


class InputMessage(StrictModel):
    role: Literal["system", "user"]
    content: Annotated[str, Field(min_length=1, max_length=32000)]


class RecordedGenerateRequest(StrictModel):
    """Recorded causal observations, using the same vLLM and image preparation."""

    request_id: Identifier
    messages: Annotated[list[InputMessage], Field(min_length=1, max_length=2)]
    images: Annotated[list[RecordedImage], Field(min_length=1, max_length=16)]
    prepared: bool = False
    expected_visual_tokens: Annotated[StrictInt, Field(ge=1, le=16384)] | None = None
    temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    seed: StrictInt = 0
    max_tokens: Annotated[StrictInt, Field(ge=32, le=512)] = 512

    @model_validator(mode="after")
    def causal_prefix(self):
        roles = [m.role for m in self.messages]
        if roles not in (["user"], ["system", "user"]):
            raise ValueError("expected an optional system message followed by one user message")
        if sum(m.content.count("<image>") for m in self.messages) != len(self.images):
            raise ValueError("image placeholder/cardinality mismatch")
        if self.messages[0].role == "system" and "<image>" in self.messages[0].content:
            raise ValueError("images belong in the user message")
        return self
