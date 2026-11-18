import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import ExpertTestEnvironment, settings

from sim_engine.expert import ExpertRollout, PathPlanner
from sim_engine.navigation.pose import Pose
from sim_engine.navigation.tracking import ControllerProfile, Polyline, UnicycleTracker
from sim_engine.pool import SessionPool
from sim_engine.sim_server import create_app

PROFILE = ControllerProfile()


def straight(length=3.0, step=0.05):
    return Polyline([(i * step, 0.0, 0.0) for i in range(int(length / step) + 1)])


class Planner:
    """Straight route along +x from the origin to ``goal_x``."""

    def __init__(self, goal_x=2.0, fail=False):
        self.goal_x, self.fail = goal_x, fail

    def densify(self, points, goal):
        if self.fail:
            raise ValueError("not connected")
        return [(i * 0.05, 0.0, 0.0) for i in range(int(self.goal_x / 0.05) + 1)]


class Environment:
    def __init__(self, physics, collisions=0):
        from sim_engine.config import Physics, VelocityLimits

        self.physics = physics or Physics(velocity=VelocityLimits(time_step_s=PROFILE.dt_s))
        self.x = self.y = self.yaw = 0.0
        self.collisions, self.camera = collisions, None

    def reset(self, episode, camera, seed):
        self.camera, self.x, self.y, self.yaw = camera, 0.0, 0.0, 0.0
        return self.image()

    def image(self):
        return np.full((self.camera.height, self.camera.width, 3), int(self.x * 40) % 255, np.uint8)

    def pose(self):
        return Pose(self.x, self.y, self.yaw)

    def position(self):
        return [self.x, 0.0, self.y]

    def distance(self, start, end):
        return float(np.hypot(end[0] - start[0], end[2] - start[2]))

    def velocity_step(self, linear_mps, angular_rps, dt, allow_sliding):
        self.x += linear_mps * dt * np.cos(self.yaw)
        self.y += linear_mps * dt * np.sin(self.yaw)
        self.yaw += angular_rps * dt
        self.collisions += 0
        return self.image(), self.collisions > 0


def episode():
    return {"episode_id": "e1", "reference_path": [[0, 1.5, 0], [2, 1.5, 0]],
            "goals": [{"position": [2, 1.5, 0]}]}


def camera():
    from sim_engine.config import Camera

    return Camera(width=32, height=32)


def encode(image):
    return image.tobytes()


def run(tmp_path, physics=None, profile=PROFILE, planner=None, max_steps=200, collisions=0):
    env = Environment(physics, collisions)
    route = planner or Planner()
    rollout = ExpertRollout(env, profile, planner_factory=lambda _env: route)
    return rollout.run(episode(), camera(), tmp_path, encode, max_steps)


def test_polyline_arc_length_and_projection():
    path = Polyline([(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    assert path.length == pytest.approx(2.0)
    assert path.point_at(0.5) == pytest.approx((0.5, 0.0))
    assert path.point_at(1.5) == pytest.approx((1.0, 0.5))
    arc, distance, heading = path.project(1.2, 0.5)
    assert (arc, distance) == pytest.approx((1.5, 0.2))
    assert heading == pytest.approx(np.pi / 2)


def test_resample_keeps_length_and_endpoints():
    path = straight(1.0, 0.01).resample(0.05)
    assert path.length == pytest.approx(1.0, abs=1e-9)
    assert path.point_at(path.length) == pytest.approx((1.0, 0.0))
    assert len(path.points) == 21


def test_profile_geometry_is_015_m_and_18_deg_per_step():
    command = PROFILE.command(0.0)
    assert command["linear_mps"] * PROFILE.dt_s == pytest.approx(0.15)
    assert PROFILE.omega_max_rps * PROFILE.dt_s == pytest.approx(np.radians(18), abs=1e-3)
    turn = PROFILE.command(np.radians(90))
    assert turn["linear_mps"] == 0.0
    assert turn["angular_rps"] * PROFILE.dt_s == pytest.approx(np.radians(18), abs=1e-3)


def test_tracker_moves_one_row_per_step_and_stops_at_the_goal():
    tracker = UnicycleTracker(straight(2.0), PROFILE)
    pose, rows = Pose(0.0, 0.0, 0.0), 0
    while True:
        command, _ = tracker.command(pose)
        if command is None:
            break
        pose = Pose(pose.x + command["linear_mps"] * PROFILE.dt_s * np.cos(pose.yaw),
                    pose.y + command["linear_mps"] * PROFILE.dt_s * np.sin(pose.yaw),
                    pose.yaw + command["angular_rps"] * PROFILE.dt_s)
        rows += 1
    assert rows == pytest.approx((2.0 - PROFILE.goal_radius_m) / 0.15, abs=1)


CROSSING = [(0, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0), (1, -1, 0)]  # passes (1, 0) at arc 1 and 5


def test_projection_window_ignores_other_passes_of_the_route():
    path = Polyline(CROSSING)
    assert path.project(1.0, 0.0)[0] == pytest.approx(1.0)
    assert path.project(1.0, 0.0, start=4.5, end=5.5)[0] == pytest.approx(5.0)


def test_tracker_keeps_going_straight_through_its_own_earlier_pass():
    path = Polyline(CROSSING).resample(0.05)
    tracker = UnicycleTracker(path, PROFILE)
    for arc in np.arange(0.0, 4.9, 0.15):  # an ideal follower reaching the second pass
        (x, y), (nx, ny) = path.point_at(arc), path.point_at(arc + 0.01)
        tracker.command(Pose(x, y, np.arctan2(ny - y, nx - x)))
    command, _ = tracker.command(Pose(1.0, 0.0, -np.pi / 2))
    assert command["linear_mps"] == pytest.approx(PROFILE.v_max_mps, rel=1e-3)
    assert tracker.max_progress == pytest.approx(5.0, abs=0.01)


def test_tracker_rejects_cross_track_beyond_limit():
    tracker = UnicycleTracker(straight(2.0), PROFILE)
    with pytest.raises(ValueError, match="cross-track"):
        tracker.command(Pose(0.0, 0.5, 0.0))


def test_tracker_allows_a_full_turn_but_rejects_a_wedged_agent():
    tracker = UnicycleTracker(straight(2.0), PROFILE)
    tracker.command(Pose(0.5, 0.0, 0.0))
    for _ in range(10):  # 180 degrees in place: no progress, still legal
        tracker.command(Pose(0.5, 0.0, np.pi))
    with pytest.raises(ValueError, match="stalled"):
        for _ in range(PROFILE.stall_steps):
            tracker.command(Pose(0.5, 0.0, 0.0))


def test_rollout_writes_frames_states_and_receipt(tmp_path):
    result = run(tmp_path)
    assert result.termination == "expert_stop"
    assert result.steps == pytest.approx((2.0 - PROFILE.goal_radius_m) / 0.15, abs=1)
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    states = json.loads((tmp_path / "states.json").read_text())["rows"]
    assert receipt["control_steps"] == result.steps and receipt["episode_id"] == "e1"
    assert len(states) == result.steps + 1 and receipt["frames"] == len(states)
    assert (tmp_path / "frames" / f"{result.steps:05d}.jpg").is_file()
    assert states[1]["t_s"] == pytest.approx(0.25)
    assert states[1]["command"]["linear_mps"] == pytest.approx(0.6)
    assert result.stats["goal_distance_m"] < PROFILE.goal_radius_m + 0.05


def test_rollout_rejects_missing_velocity_physics(tmp_path):
    from sim_engine.config import Physics

    with pytest.raises(ValueError, match="velocity physics"):
        run(tmp_path, physics=Physics(velocity=None), max_steps=1)


def test_rollout_rejects_profile_physics_mismatch(tmp_path):
    from sim_engine.config import Physics, VelocityLimits

    with pytest.raises(ValueError, match="does not match profile dt"):
        run(tmp_path, physics=Physics(velocity=VelocityLimits(time_step_s=0.1)), max_steps=1)


def test_rollout_planner_failure_is_reported(tmp_path):
    with pytest.raises(ValueError, match="path planning failed"):
        run(tmp_path, planner=Planner(fail=True))


def test_rollout_budget_exhaustion_is_marked(tmp_path):
    result = run(tmp_path, max_steps=3)
    assert result.termination == "budget_exhausted" and result.steps == 3


def test_rollout_is_deterministic(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    run(first)
    run(second)
    assert (first / "states.json").read_text() == (second / "states.json").read_text()
    assert sorted(p.name for p in (first / "frames").iterdir()) == sorted(
        p.name for p in (second / "frames").iterdir())


def test_planner_maps_habitat_world_to_the_base_frame():
    env = ExpertTestEnvironment(None)
    points = PathPlanner(env).densify([[0, 0, 0], [0, 0, -1]], [-1, 0, -1])
    # Habitat -z is forward (+x), Habitat -x is left (+y).
    assert points == [(0, 0, 0.0), (1, 0, 0.0), (1, 1, 0.0)]


def test_planner_reports_disconnected_reference():
    class Disconnected(ExpertTestEnvironment):
        def shortest_path(self, start, end):
            raise RuntimeError("unreachable geodesic path")

    with pytest.raises(ValueError, match="not connected"):
        PathPlanner(Disconnected(None)).densify([[0, 0, 0]], [0, 0, -1])


def expert_app(tmp_path, root):
    config = settings(tmp_path, workers=1)
    config.simulation.max_actions = 40
    config.simulation.expert_root = root

    def get(dataset, key):
        if key != "a":
            raise KeyError(key)
        return {"_dataset": dataset, "episode_id": key, "scene_id": "mp3d/one/one.glb",
                "instruction": {"instruction_text": "forward"},
                "reference_path": [[0, 0, 0], [0, 0, -2]], "goals": [{"position": [0, 0, -2]}]}

    store = type("Store", (), {"metadata": {}, "scene_hashes": {}, "get": staticmethod(get)})()
    return create_app(config, store=store,
                      pool_factory=lambda cfg: SessionPool(cfg, factory=ExpertTestEnvironment))


def test_http_expert_job_writes_once_and_frees_the_worker(tmp_path):
    root = tmp_path / "experts"
    with TestClient(expert_app(tmp_path, root)) as client:
        body = {"dataset": "test", "episode_id": "a"}
        reply = client.post("/v1/experts", json=body)
        assert reply.status_code == 200, reply.text
        value = reply.json()
        assert value["termination"] == "expert_stop" and value["output"] == str(root / "test" / "a")
        assert not {"frames", "states"} & value.keys()
        assert (root / "test" / "a" / "receipt.json").is_file()
        assert [p.name for p in (root / "test").iterdir()] == ["a"]
        assert client.post("/v1/experts", json=body).status_code == 409
        assert client.post("/v1/experts", json={**body, "episode_id": "b"}).status_code == 404
        assert client.post("/v1/sessions", json=body).status_code == 200


def test_http_expert_requires_a_configured_root(tmp_path):
    with TestClient(expert_app(tmp_path, None)) as client:
        reply = client.post("/v1/experts", json={"dataset": "test", "episode_id": "a"})
        assert reply.status_code == 422 and "expert_root" in reply.text
