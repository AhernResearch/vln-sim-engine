import json
import math

import pytest

from sim_engine.config import Camera
from sim_engine.metrics import navigation_metrics
from sim_engine.outputs import parse_prediction
from sim_engine.prompt import PromptBuilder


def test_v2_explicit_stop_and_stationary_predictions_have_different_meanings():
    assert parse_prediction('{"stop":true,"trajectory":[]}', "navigation_v2").stop
    stationary = parse_prediction(
        json.dumps({"stop": False, "trajectory": [[0, 0, 0]] * 8}), "navigation_v2"
    )
    assert not stationary.stop
    assert not parse_prediction(json.dumps({"trajectory": [[0, 0, 0]] * 8}), "trajectory_v1").stop
    with pytest.raises(ValueError):
        parse_prediction('{"stop":true,"trajectory":[]}', "trajectory_v1")


@pytest.mark.parametrize(
    "value",
    [
        {"stop": 1, "trajectory": []},
        {"stop": "true", "trajectory": []},
        {"stop": True, "trajectory": [[0, 0, 0]] * 8},
        {"stop": False, "trajectory": []},
        {"trajectory": [[0, 0, 0]] * 8},
        {"stop": False, "trajectory": [[True, 0, 0]] * 8},
        {"stop": False, "trajectory": [[float("nan"), 0, 0]] * 8},
        {"stop": True, "trajectory": [], "reason": "near goal"},
    ],
)
def test_v2_rejects_ambiguous_or_invalid_completion(value):
    with pytest.raises(ValueError):
        parse_prediction(json.dumps(value), "navigation_v2")


def test_v2_prompt_has_causal_images_without_event_or_completion_hints():
    builder = PromptBuilder("navigation_v2")
    a = builder.messages("Walk past <image>.", Camera(), [0, 1, 3])
    b = builder.messages("Walk past <image>.", Camera(), [0, 20, 40])
    assert a == b
    assert a[1]["content"].count("<image>") == 3
    assert "Current: <image>" in a[1]["content"]
    assert "event_index" not in a[1]["content"]
    assert '"robot":"ground"' in a[1]["content"]
    assert '"camera_pitch_deg":30' in a[1]["content"]
    assert "stop=true, trajectory=[]" in a[0]["content"]


def test_success_requires_both_explicit_stop_and_goal_proximity():
    positions = [[0, 0, 0], [4, 0, 0]]
    assert navigation_metrics(positions, positions, [4, 0, 0], math.dist)["success"] == 0
    correct = navigation_metrics(positions, positions, [4, 0, 0], math.dist, stopped=True)
    assert correct["success"] == correct["spl"] == 1
    assert navigation_metrics(positions, positions, [10, 0, 0], math.dist, stopped=True)["success"] == 0
