import pytest
from pydantic import ValidationError

from sim_engine.client import history_indices
from sim_engine.config import Camera
from sim_engine.prompt import PromptBuilder
from sim_engine.protocol import GenerateRequest, StepRequest


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "1"])
def test_trajectory_rejects_non_numeric_or_non_finite_values(value):
    with pytest.raises(ValidationError):
        StepRequest(request_id="a", expected_event_index=0, trajectory=[[value, 0, 0]] * 8)


def test_policy_request_excludes_truth_and_mixed_sessions():
    request = {
        "request_id": "a",
        "instruction": "forward",
        "observations": [{"session_id": "one", "event_index": 0}],
    }
    with pytest.raises(ValidationError):
        GenerateRequest(**request, goal=[1, 2, 3])
    with pytest.raises(ValidationError):
        GenerateRequest(
            **{
                **request,
                "observations": [
                    {"session_id": "one", "event_index": 0},
                    {"session_id": "two", "event_index": 1},
                ],
            }
        )


def test_causal_history_and_training_prompt_contract():
    assert history_indices(4) == [0, 1, 2, 3, 4]
    assert history_indices(64) == [0, 1, 30, 60, 61, 62, 63, 64]
    for index in range(500):
        selected = history_indices(index)
        assert selected == sorted(set(selected)) and selected[-1] == index and len(selected) <= 8
    messages = PromptBuilder().messages("literal <image>", Camera(), [0, 1])
    assert messages[1]["content"].count("<image>") == 2
    assert "camera_pitch_down_rad" in messages[1]["content"]
    assert "not incremental displacements" in messages[0]["content"]
