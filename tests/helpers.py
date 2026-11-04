"""Shared test doubles: a deterministic environment and minimal settings."""

import math

import numpy as np

from sim_engine.config import Physics, Settings
from sim_engine.geometry import Pose


class TestEnvironment:
    __test__ = False

    def __init__(self, settings):
        self.x = self.y = self.yaw = 0
        self.physics = Physics()

    def reset(self, episode, camera, seed):
        self.x = self.y = self.yaw = 0
        self.camera = camera
        return self.image()

    def image(self):
        return np.full(
            (self.camera.height, self.camera.width, 3), int(self.x * 20) % 255, dtype=np.uint8
        )

    def pose(self):
        return Pose(self.x, self.y, self.yaw)

    def position(self):
        return [-self.y, 0, -self.x]

    def step(self, action):
        if action == "MOVE_FORWARD":
            self.x += 0.25 * math.cos(self.yaw)
            self.y += 0.25 * math.sin(self.yaw)
        elif action == "TURN_LEFT":
            self.yaw += math.pi / 12
        else:
            self.yaw -= math.pi / 12
        return self.image(), False

    def velocity_step(self, linear_mps, angular_rps, dt, allow_sliding):
        self.x += linear_mps * dt * math.cos(self.yaw)
        self.y += linear_mps * dt * math.sin(self.yaw)
        self.yaw += angular_rps * dt
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
