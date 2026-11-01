"""Checkpoint-specific output contracts; stationary trajectories never imply STOP."""

import json

from pydantic import StrictBool, field_validator, model_validator

from .config import StrictModel
from .protocol import StepRequest


class NavigationPrediction(StrictModel):
    stop: StrictBool
    trajectory: list[tuple[float, float, float]]

    @field_validator("trajectory", mode="before")
    @classmethod
    def finite_points(cls, value):
        return StepRequest.finite_numeric_points(value)

    @model_validator(mode="after")
    def stop_and_horizon_agree(self):
        if len(self.trajectory) != (0 if self.stop else 8):
            raise ValueError("STOP requires an empty trajectory; movement requires eight poses")
        return self


def parse_prediction(text, profile):
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("prediction must be a JSON object")
    if profile == "trajectory_v1":
        if set(value) != {"trajectory"}:
            raise ValueError("v1 checkpoint requires a trajectory-only output")
        value = dict(stop=False, **value)
    elif profile != "navigation_v2":
        raise ValueError("unknown checkpoint output profile")
    return NavigationPrediction.model_validate(value)
