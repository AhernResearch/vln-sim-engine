import math

import pytest
from helpers import TestEnvironment, episode, request, settings
from pydantic import ValidationError

from sim_engine.config import Physics, VelocityLimits
from sim_engine.controllers import CommandRejected
from sim_engine.pool import SessionPool, WorkerError
from sim_engine.protocol import StepRequest
from sim_engine.session import Session


def open_session(tmp_path, physics=None):
    config = settings(tmp_path)
    environment = TestEnvironment(config)
    if physics is not None:
        environment.physics = physics
    camera = config.cameras["trained_ce30"]
    return Session.open("one", environment, episode(), "trained_ce30", camera, 0, config)


def velocity(key, expected, linear, angular):
    return {
        "request_id": key,
        "expected_event_index": expected,
        "command": {"kind": "velocity", "linear_mps": linear, "angular_rps": angular},
    }


def test_legacy_body_is_the_explicit_waypoint_command(tmp_path):
    legacy = request()
    explicit = {
        "request_id": "r1",
        "expected_event_index": 0,
        "command": {"kind": "waypoints", "trajectory": legacy["trajectory"], "action_budget": 4},
    }
    assert StepRequest.model_validate(legacy) == StepRequest.model_validate(explicit)
    a, b = open_session(tmp_path), open_session(tmp_path)
    assert a.step(legacy)["executed_actions"] == b.step(explicit)["executed_actions"]
    assert [list(p) for p in a.predictions[0]["trajectory"]] == legacy["trajectory"]


@pytest.mark.parametrize(
    "command",
    [
        {"kind": "teleport", "linear_mps": 1, "angular_rps": 0},
        {"kind": "velocity", "linear_mps": float("nan"), "angular_rps": 0},
        {"kind": "velocity", "linear_mps": 1},
        {"kind": "velocity", "linear_mps": 1, "angular_rps": 0, "time_step_s": 1},
    ],
)
def test_malformed_commands_are_schema_errors(command):
    with pytest.raises(ValidationError):
        StepRequest.model_validate(
            {"request_id": "x", "expected_event_index": 0, "command": command}
        )


def test_velocity_is_clipped_to_dataset_limits_and_integrated_for_one_time_step(tmp_path):
    session = open_session(tmp_path, Physics(velocity=VelocityLimits()))
    reply = session.step(velocity("v1", 0, 9.0, 0.0))
    assert reply["event_index"] == 1 and len(reply["observations"]) == 1
    assert reply["executed_actions"] == [{"linear_mps": 2.5, "angular_rps": 0.0}]
    assert session.env.pose().x == pytest.approx(0.25)
    reply = session.step(velocity("v2", 1, -1.0, 100.0))
    assert reply["executed_actions"][0]["linear_mps"] == 0.0
    assert reply["executed_actions"][0]["angular_rps"] == pytest.approx(math.radians(300))
    assert session.env.pose().yaw == pytest.approx(math.radians(30))
    assert session.predictions[-1]["command"]["kind"] == "velocity"


def test_rejected_commands_leave_the_session_unchanged(tmp_path):
    discrete = open_session(tmp_path)
    with pytest.raises(CommandRejected, match="does not define velocity"):
        discrete.step(velocity("v1", 0, 1.0, 0.0))
    session = open_session(tmp_path, Physics(velocity=VelocityLimits()))
    with pytest.raises(CommandRejected, match="STOP"):
        session.step(velocity("v1", 0, 0.001, 0.0))
    assert len(session.actions) == 0 and len(session.frames) == 1
    assert session.step(velocity("v2", 0, 1.0, 0.0))["event_index"] == 1


def test_forced_stop_is_scored_at_the_stopped_pose_but_kept_distinct(tmp_path):
    session = open_session(tmp_path, Physics(velocity=VelocityLimits()))
    session.step(velocity("v1", 0, 1.0, 0.0))
    session.termination = "forced_stop"
    metrics = session.metrics()
    assert metrics["termination"] == "forced_stop"
    assert metrics["metrics"]["success"] == 1.0


async def test_worker_reports_rejection_as_422_without_invalidating_the_session(tmp_path):
    pool = SessionPool(settings(tmp_path, workers=1), factory=TestEnvironment)
    try:
        reset = await pool.reset(episode(), "trained_ce30", 0)
        sid = reset["session_id"]
        with pytest.raises(WorkerError) as error:
            payload = StepRequest.model_validate(velocity("v1", 0, 1.0, 0.0)).model_dump()
            await pool.call(sid, "step", request=payload)
        assert error.value.status == 422
        payload = StepRequest.model_validate(request()).model_dump()
        assert (await pool.call(sid, "step", request=payload))["event_index"] == 4
    finally:
        pool.close()
