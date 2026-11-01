import asyncio
import math

import numpy as np
import pytest

from sim_engine.config import Settings
from sim_engine.geometry import Pose
from sim_engine.pool import SessionPool, WorkerError
from sim_engine.session import Conflict, Session


class TestEnvironment:
    __test__ = False

    def __init__(self, settings):
        self.x = self.yaw = 0

    def reset(self, episode, camera, seed):
        self.x = self.yaw = 0
        self.camera = camera
        return self.image()

    def image(self):
        return np.full(
            (self.camera.height, self.camera.width, 3), int(self.x * 20) % 255, dtype=np.uint8
        )

    def pose(self):
        return Pose(self.x, 0, self.yaw)

    def position(self):
        return [0, 0, -self.x]

    def step(self, action):
        if action == "MOVE_FORWARD":
            self.x += 0.25
        elif action == "TURN_LEFT":
            self.yaw += math.pi / 12
        else:
            self.yaw -= math.pi / 12
        return self.image(), False

    def distance(self, a, b):
        return math.dist(a, b)

    def metrics(self, episode, positions, stopped=False):
        return {"success": float(stopped), "path_length": len(positions) * 0.25}

    def close(self):
        pass


def settings(tmp_path, workers=2):
    return Settings.model_validate(
        {
            "simulation": {
                "scenes_root": str(tmp_path),
                "datasets": {},
                "workers": workers,
                "rpc_timeout_seconds": 10,
                "max_actions": 8,
            },
            "policy": {"model": str(tmp_path)},
        }
    )


def episode(key="a"):
    return {
        "episode_id": key,
        "scene_id": "mp3d/one/one.glb",
        "instruction": {"instruction_text": "forward"},
        "goals": [{"position": [0, 0, -5]}],
    }


def request(key="r1", expected=0):
    return {
        "request_id": key,
        "expected_event_index": expected,
        "trajectory": [[i * 0.25, 0, 0] for i in range(1, 9)],
        "action_budget": 4,
    }


def test_step_retry_is_exactly_once_and_observations_are_immutable(tmp_path):
    config = settings(tmp_path)
    session = Session(
        "one",
        TestEnvironment(config),
        episode(),
        "trained_ce30",
        config.cameras["trained_ce30"],
        0,
        config,
    )
    initial = session.frames[0]
    result = session.step(request())
    assert session.step(request()) == result
    assert len(session.actions) == 4
    assert result["event_index"] == 4
    assert session.frames[0] == initial
    assert session.frames[0] != session.frames[4]
    with pytest.raises(Conflict, match="different content"):
        session.step(request(expected=4))
    with pytest.raises(Conflict, match="out-of-order"):
        session.step(request("other", 0))
    session.step(request("second", 4))
    assert session.termination == "timeout"
    assert session.metrics()["metrics"]["success"] == 0


async def test_worker_processes_isolate_sessions_and_release_capacity(tmp_path):
    config = settings(tmp_path)
    pool = SessionPool(config, factory=TestEnvironment)
    try:
        a, b = await asyncio.gather(
            pool.reset(episode("a"), "trained_ce30", 0), pool.reset(episode("b"), "trained_ce30", 0)
        )
        with pytest.raises(WorkerError) as error:
            await pool.reset(episode("c"), "trained_ce30", 0)
        assert error.value.status == 429
        await pool.call(a["session_id"], "step", request=request())
        assert len((await pool.call(b["session_id"], "metrics"))["actions"]) == 0
        await pool.release(a["session_id"])
        c = await pool.reset(episode("c"), "trained_ce30", 0)
        assert c["session_id"] != a["session_id"]
        with pytest.raises(WorkerError):
            await pool.call(a["session_id"], "metrics")
    finally:
        pool.close()


async def test_dead_worker_invalidates_session_and_other_worker_remains_usable(tmp_path):
    pool = SessionPool(settings(tmp_path), factory=TestEnvironment)
    try:
        reset = await pool.reset(episode(), "trained_ce30", 0)
        worker = pool.sessions[reset["session_id"]]
        worker.process.terminate()
        worker.process.join(timeout=5)
        with pytest.raises(WorkerError) as error:
            await pool.call(reset["session_id"], "metrics")
        assert error.value.status == 503
        assert reset["session_id"] not in pool.sessions
        other = await pool.reset(episode("other"), "trained_ce30", 0)
        assert other["episode_id"] == "other"
    finally:
        pool.close()
