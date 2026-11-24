"""Planner-delegated legs: observe snapshots, point goals and retrace on one continuous session."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import ExpertTestEnvironment, episode, settings

from sim_engine import controllers
from sim_engine.backends.contracts import Capture
from sim_engine.config import Legs, Physics, VelocityLimits
from sim_engine.core.process import WorkerError
from sim_engine.navigation import point_goal
from sim_engine.navigation.point_goal import TargetRejected
from sim_engine.pool import SessionPool
from sim_engine.protocol import StepRequest
from sim_engine.session import Conflict, Session

VIEWS = ("front", "right", "back", "left")


class LegEnvironment(ExpertTestEnvironment):
    __test__ = False
    pathfinder = None

    def observe_views(self, yaw):
        image = self.image()
        return {name: Capture({"rgb": image, "depth": np.ones(image.shape[:2], np.float32)},
                              self.position(), [0, 0, 0, 1],
                              {"rgb": {}, "depth": {"position": self.position(),
                                                    "rotation_xyzw": [0, 0, 0, 1]}},
                              {"projections": {"depth": [[1, 0, 0, 0], [0, 1, 0, 0]]}})
                for name in VIEWS}


@pytest.fixture
def session(tmp_path, monkeypatch):
    config = settings(tmp_path)
    config.simulation.max_actions = 200
    geometry = SimpleNamespace(pf=None, guard=None, planners=[], clear=lambda: None)
    targets = {}

    def ground(depth, k, sensor, start, mesh, guard, point, legs):
        if point not in targets:
            raise TargetRejected("that spot is not visible floor")
        return {"goal": targets[point]}

    def plan(planners, guard, guides, goal, radius, goal_radius):
        return {"status": "feasible", "path": [guides[0], goal], "fixed_goal": goal,
                "planner_index": 0, "comfort_fallback_used": False, "guide_adjustments": []}

    monkeypatch.setattr(controllers, "ground_point", ground)
    monkeypatch.setattr(controllers, "plan_with_preferences", plan)
    value = Session.open("one", LegEnvironment(config), episode(), "trained_ce30",
                         config.cameras["trained_ce30"], 0, config, geometry=lambda: geometry)
    value.targets = targets
    return value


def step(session, key, command):
    return session.step({"request_id": key, "expected_event_index": len(session.actions),
                         "command": command})


def point(session, key, xy, view="front"):
    observation = session.observe(key + "-observe")["observation_id"]
    return step(session, key, {"kind": "point_goal", "observation_id": observation,
                               "view": view, "point": list(xy)})


def test_observe_returns_png_views_and_repeats_by_request_id(session):
    reply = session.observe("o1")
    assert set(reply["views"]) == set(VIEWS) and reply["event_index"] == 0
    assert session.observe("o1") == reply
    png, digest = session.observation_image(reply["observation_id"], "left")
    assert png.startswith(b"\x89PNG") and digest == reply["views"]["left"]["sha256"]
    assert session.actions == []


def test_point_goal_walks_to_the_grounded_goal_and_hides_private_geometry(session):
    session.targets[(500, 800)] = [0.0, 0.0, -1.0]
    reply = point(session, "p1", (500, 800))
    leg = reply["leg"]
    assert leg["status"] == "reached" and leg["leg_id"] == 1
    assert leg["end_event_index"] == reply["event_index"] > 0
    assert 0.9 < leg["net_displacement_m"] <= 1.0 and leg["collision_steps"] == 0
    assert "private" not in leg and "goal" not in leg
    assert session.metrics()["legs"][0]["private"]["goal"] == [0.0, 0.0, -1.0]


def test_rejected_targets_do_not_move_or_count_as_stalls(session):
    for key in ("a", "b", "c", "d"):
        leg = point(session, key, (10, 10))["leg"]
        assert leg["status"] == "invalid_target" and leg["reason"] == "that spot is not visible floor"
    assert session.actions == [] and session.termination is None and len(session.legs) == 4


def test_motion_makes_earlier_observations_stale(session):
    session.targets[(500, 800)] = [0.0, 0.0, -1.0]
    old = session.observe("before")["observation_id"]
    step(session, "p1", {"kind": "point_goal", "observation_id": old, "view": "front",
                         "point": [500, 800]})
    with pytest.raises(Conflict, match="stale"):
        step(session, "p2", {"kind": "point_goal", "observation_id": old, "view": "front",
                             "point": [500, 800]})


def test_retrace_returns_along_the_executed_path_to_an_earlier_leg(session):
    session.targets[(500, 800)] = [0.0, 0.0, -1.0]
    session.targets[(500, 700)] = [0.0, 0.0, -2.0]
    point(session, "p1", (500, 800))
    point(session, "p2", (500, 700))
    assert session.env.position()[2] == pytest.approx(-2.0, abs=0.15)
    leg = step(session, "r1", {"kind": "retrace", "leg_id": 1})["leg"]
    assert leg["kind"] == "retrace" and leg["leg_id"] == 3
    assert session.env.position()[2] == pytest.approx(-1.0, abs=0.15)
    assert step(session, "r2", {"kind": "retrace", "leg_id": 9})["leg"]["status"] == "invalid_target"


def velocity(session, key, linear=0.5, angular=0.0):
    return step(session, key, {"kind": "velocity", "linear_mps": linear, "angular_rps": angular})


def test_retrace_to_an_event_follows_the_recorded_trajectory_of_any_motion(session):
    for index in range(4):
        velocity(session, f"v{index}")
    session.targets[(500, 800)] = [0.0, 0.0, -4.0]
    point(session, "p1", (500, 800))
    leg = step(session, "r1", {"kind": "retrace", "event_index": 2})["leg"]
    assert leg["kind"] == "retrace" and leg["status"] == "reached" and leg["leg_id"] == 2
    assert session.env.position()[2] == pytest.approx(session.positions[2][2], abs=0.15)
    assert session.metrics()["legs"][1]["private"]["retrace_to_event"] == 2


def test_retrace_to_an_event_rejects_future_events_and_the_current_pose_without_moving(session):
    velocity(session, "v0")
    for key, event in (("future", 9), ("here", 1)):
        leg = step(session, key, {"kind": "retrace", "event_index": event})["leg"]
        assert leg["status"] == "invalid_target"
    assert len(session.actions) == 1


def test_retrace_names_exactly_one_target():
    for command in ({"kind": "retrace"}, {"kind": "retrace", "leg_id": 1, "event_index": 0}):
        with pytest.raises(ValueError, match="exactly one"):
            StepRequest.model_validate({"request_id": "r", "expected_event_index": 0,
                                        "command": command})


def test_legs_follow_the_dataset_time_step_instead_of_assuming_four_hertz(session):
    session.targets[(500, 800)] = [0.0, 0.0, -1.0]
    session.env.physics = session.physics = Physics(velocity=VelocityLimits(time_step_s=0.1))
    leg = point(session, "p1", (500, 800))["leg"]
    assert leg["status"] == "reached" and leg["end_event_index"] > 15
    assert controllers.at_time_step(controllers.expert_profile({"expert_goal_radius_m": 0.08}),
                                    0.1).stall_steps == 40  # a 180-degree turn is 25 steps
    back = step(session, "r1", {"kind": "retrace", "event_index": 0})["leg"]
    assert back["status"] == "reached" and session.env.position()[2] == pytest.approx(0, abs=0.15)


def test_leg_budget_is_reported_separately_from_the_episode_budget(session):
    session.targets[(500, 800)] = [0.0, 0.0, -5.0]
    observation = session.observe("o")["observation_id"]
    leg = step(session, "p", {"kind": "point_goal", "observation_id": observation, "view": "front",
                              "point": [500, 800], "max_actions": 3})["leg"]
    assert leg["status"] == "leg_budget" and leg["end_event_index"] == 3


async def test_worker_serves_observe_and_view_images(tmp_path):
    pool = SessionPool(settings(tmp_path, workers=1), factory=LegEnvironment)
    try:
        opened = await pool.reset(episode(), "trained_ce30", 0)
        reply = await pool.call(opened["session_id"], "observe", request_id="o1")
        png, digest = await pool.call(opened["session_id"], "observation",
                                      observation_id=reply["observation_id"], view="back")
        assert png.startswith(b"\x89PNG") and digest == reply["views"]["back"]["sha256"]
        depth = await pool.call(opened["session_id"], "observation", sensor="depth",
                                observation_id=reply["observation_id"], view="back")
        assert depth["encoding"] == "depth_npz" and "position" in json.dumps(depth["pose"])
        with np.load(io.BytesIO(depth["raw"])) as stored:
            assert stored["depth_m"].ndim == 2
        with pytest.raises(WorkerError) as error:
            await pool.call(opened["session_id"], "observation", observation_id="obs-9", view="back")
        assert error.value.status == 404
    finally:
        await pool.close()


def test_ground_point_snaps_within_radius_and_names_the_rejection(monkeypatch):
    depth = np.ones((101, 201), np.float32)
    safe = np.zeros(depth.shape, bool)
    safe[50, 103] = True
    goals = np.zeros((*depth.shape, 3), np.float32)
    goals[50, 103] = [1, 0, -2]
    stats = {"visible_floor_candidates": 5, "rejection_counts": {"not connected to start": 4}}
    monkeypatch.setattr(point_goal, "safe_floor_mask",
                        lambda *a, **k: (safe, goals, np.full(depth.shape, 0.3), stats))
    legs = Legs()  # radius 0.02 * 200 = 4 px
    near = point_goal.ground_point(depth, None, None, [0, 0, 0], None, None, (500, 500), legs)
    assert near["grounded_pixel_xy"] == [103, 50] and near["snap_px"] == 3
    assert near["goal"] == [1.0, 0.0, -2.0]
    with pytest.raises(TargetRejected, match="not reachable from here"):
        point_goal.ground_point(depth, None, None, [0, 0, 0], None, None, (450, 500), legs)
    with pytest.raises(TargetRejected, match="not reachable"):
        point_goal.ground_point(depth, None, None, [0, 0, 0], None, None, (500, 500),
                                Legs(snap_radius=0))


def test_metrics_carry_the_goal_and_reference_route_for_evaluation(session):
    reference = session.metrics()["reference"]
    assert reference["goal"] == session.episode["goals"][0]["position"]
    assert reference["path"] == session.episode.get("reference_path")
