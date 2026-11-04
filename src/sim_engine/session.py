"""Worker-owned state. Ground truth is never included in policy observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from .geometry import next_action, reached, to_world
from .protocol import StepRequest


class Conflict(ValueError):
    pass


class Session:
    def __init__(self, session_id, environment, episode, camera_name, camera, settings):
        self.id, self.env, self.episode = session_id, environment, episode
        self.camera_name, self.camera, self.settings = camera_name, camera, settings
        self.physics = environment.physics
        self.frames, self.actions, self.positions, self.predictions = [], [], [], []
        self.collisions = self.stalled = 0
        self.termination = None
        self.replies = {}

    @classmethod
    def open(cls, session_id, environment, episode, camera_name, camera, seed, settings):
        """Reset the episode, then capture the initial frame and position."""
        image = environment.reset(episode, camera, seed)
        session = cls(session_id, environment, episode, camera_name, camera, settings)
        session._store(image)
        session.positions.append(environment.position())
        return session

    def _store(self, image):
        if image.shape != (self.camera.height, self.camera.width, 3):
            raise RuntimeError("renderer returned unexpected image shape")
        # Immutable snapshot, independent of renderer buffer reuse.
        raw = image.tobytes()
        self.frames.append((raw, hashlib.sha256(raw).hexdigest()))

    def observation(self, index):
        return {
            "session_id": self.id,
            "event_index": index,
            "width": self.camera.width,
            "height": self.camera.height,
            "sha256": self.frames[index][1],
            "encoding": "rgb8",
        }

    def reset_reply(self):
        return {
            "session_id": self.id,
            "episode_id": str(self.episode["episode_id"]),
            "instruction": self.episode["instruction"]["instruction_text"],
            "camera": self.camera_name,
            "robot": self.camera.robot(),
            "observations": [self.observation(0)],
            "event_index": 0,
            "done": False,
        }

    def step(self, payload):
        request = StepRequest.model_validate(payload)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        if request.request_id in self.replies:
            previous_digest, reply = self.replies[request.request_id]
            if previous_digest != digest:
                raise Conflict("request_id was already used with different content")
            return reply
        if request.expected_event_index != len(self.actions):
            raise Conflict("stale or out-of-order event index")
        if self.termination:
            raise Conflict("episode has already ended")
        origin = self.env.pose()
        targets = [to_world(origin, point) for point in request.trajectory]
        start, cursor = len(self.actions), 0
        before = (
            self.env.distance(self.positions[-1], self.episode["goals"][0]["position"])
            if self.settings.simulation.reward_mode != "none"
            else None
        )
        old_collisions = self.collisions
        for _ in range(min(request.action_budget, self.settings.simulation.max_actions - start)):
            pose = self.env.pose()
            while cursor < len(targets) and reached(
                pose, targets[cursor], self.physics.forward_m, self.physics.turn_degrees
            ):
                cursor += 1
            if cursor == len(targets):
                break
            action = next_action(
                pose, targets[cursor], self.physics.forward_m, self.physics.turn_degrees
            )
            if action is None:
                break
            image, collided = self.env.step(action)
            self._store(image)
            self.actions.append(action)
            self.positions.append(self.env.position())
            self.collisions += int(collided)
        end = len(self.actions)
        self.stalled = self.stalled + 1 if start == end else 0
        if end >= self.settings.simulation.max_actions:
            self.termination = "timeout"
        elif self.stalled >= 3:
            self.termination = "stalled"
        self.predictions.append(
            {
                "request_id": request.request_id,
                "origin": asdict(origin),
                "start_event_index": start,
                "end_event_index": end,
                "trajectory": request.trajectory,
                "policy_version": request.policy_version,
            }
        )
        reply = {
            "session_id": self.id,
            "request_id": request.request_id,
            "event_index": end,
            "executed_actions": self.actions[start:end],
            "observations": [self.observation(i) for i in range(start + 1, end + 1)],
            "collisions": self.collisions - old_collisions,
            "done": self.termination is not None,
            "termination": self.termination,
        }
        if before is not None:
            after = self.env.distance(self.positions[-1], self.episode["goals"][0]["position"])
            components = {
                "geodesic_progress": before - after,
                "collision_penalty": -0.01 * (self.collisions - old_collisions),
                "action_penalty": -0.001 * (end - start),
            }
            reply["training_reward"] = {
                "version": "experimental_progress_v1",
                "components": components,
                "total": sum(components.values()),
            }
        self.replies[request.request_id] = digest, reply
        return reply

    def metrics(self):
        result = self.env.metrics(
            self.episode, self.positions, stopped=self.termination == "policy_stop"
        )
        result["collisions"] = self.collisions
        return {
            "episode_id": str(self.episode["episode_id"]),
            "scene_id": self.episode["scene_id"],
            "session_id": self.id,
            "done": self.termination is not None,
            "termination": self.termination,
            "metrics": result,
            "positions": self.positions,
            "actions": self.actions,
            "predictions": self.predictions,
        }
