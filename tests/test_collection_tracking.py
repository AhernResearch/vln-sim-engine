from types import SimpleNamespace

import pytest

from sim_engine.collection.tracking import EndpointTracker


def test_final_translation_cannot_overshoot_remaining_route():
    tracker = SimpleNamespace(
        path=SimpleNamespace(length=1.0),
        max_progress=0.91,
        command=lambda pose: ({"linear_mps": 0.6, "angular_rps": 0.2}, 0.01),
    )
    command, error = EndpointTracker(tracker, 0.25).command(None)
    assert command["linear_mps"] == pytest.approx(0.36)
    assert command["angular_rps"] == 0.2 and error == 0.01


def test_rotation_and_stop_are_preserved():
    tracker = SimpleNamespace(
        path=SimpleNamespace(length=1.0),
        max_progress=0.91,
        command=lambda pose: ({"linear_mps": 0.0, "angular_rps": 1.26}, 0.01),
    )
    command, _ = EndpointTracker(tracker, 0.25).command(None)
    assert command == {"linear_mps": 0.0, "angular_rps": 1.26}
    tracker.command = lambda pose: (None, 0.01)
    assert EndpointTracker(tracker, 0.25).command(None) == (None, 0.01)
