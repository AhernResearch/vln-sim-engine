"""Short model-free session acceptance through the public HTTP API.

One session runs observe, point goal, observe, retrace to an event, one velocity step and finish.
Every reply a policy-side caller can see is scanned for privileged fields. Motion is never retried:
a point goal that grounds to nothing executes nothing, so the next candidate pixel is not a retry.
"""

import uuid

# Goals, reference routes and global poses are simulator-side only.
PRIVILEGED = frozenset({
    "goal", "goals", "reference", "reference_path", "position", "positions", "pose", "poses",
    "start_position", "global_pose", "rotation_xyzw", "goal_distance", "geodesic_distance",
})
UNSTARTED = frozenset({"invalid_target", "unreachable"})
# Normalized (x, y) pixels in the lower middle of a view, where the floor is nearest.
GOAL_PIXELS = ((500, 800), (500, 700), (500, 900), (300, 800), (700, 800))


def privileged_keys(value, path="$") -> list[str]:
    if isinstance(value, dict):
        return [found for key, item in value.items()
                for found in ([f"{path}.{key}"] if key in PRIVILEGED else []) + privileged_keys(item, f"{path}.{key}")]
    if isinstance(value, list):
        return [found for index, item in enumerate(value) for found in privileged_keys(item, f"{path}[{index}]")]
    return []


class SessionFlow:
    """Runs against any client with ``get/post/delete``: ``httpx.Client`` or a ``TestClient``."""

    def __init__(self, client, dataset, camera, episode_id, goal_pixels=GOAL_PIXELS):
        self.client, self.dataset, self.camera = client, dataset, camera
        self.episode_id, self.goal_pixels = str(episode_id), goal_pixels
        self.sid = None
        self.replies = {}

    def _json(self, name, response, expected=200):
        assert response.status_code == expected, f"{name}: {response.status_code} {response.text}"
        value = response.json()
        leaked = privileged_keys(value)
        assert not leaked, f"{name} reply exposes privileged fields: {leaked}"
        self.replies[name] = value
        return value

    def _step(self, name, event, command):
        body = {"request_id": uuid.uuid4().hex, "expected_event_index": event, "command": command}
        return self._json(name, self.client.post(f"/v1/sessions/{self.sid}/step", json=body))

    def _observe(self, name, event):
        view = self._json(name, self.client.post(f"/v1/sessions/{self.sid}/observe",
                                                 json={"request_id": uuid.uuid4().hex}))
        assert view["event_index"] == event and set(view["views"]) == {"front", "right", "back", "left"}
        return view["observation_id"]

    def _point_goal(self, event, observation):
        for pixel in self.goal_pixels:
            reply = self._step("point_goal", event, {"kind": "point_goal", "observation_id": observation,
                                                     "view": "front", "point": list(pixel)})
            if reply["leg"]["status"] not in UNSTARTED:
                return pixel, reply
        raise AssertionError(f"no candidate pixel grounded to a walkable goal: {reply['leg']}")

    def run(self) -> dict:
        reset = self._json("reset", self.client.post("/v1/sessions", json={
            "dataset": self.dataset, "camera": self.camera, "episode_id": self.episode_id}))
        self.sid = reset["session_id"]
        try:
            assert reset["event_index"] == 0
            first = self._observe("observe_start", 0)
            pixel, goal = self._point_goal(0, first)
            moved = goal["event_index"]
            assert moved > 0 and goal["leg"]["end_event_index"] == moved, goal["leg"]
            second = self._observe("observe_after_goal", moved)
            assert second != first
            target = moved // 2
            retrace = self._step("retrace", moved, {"kind": "retrace", "event_index": target})
            assert retrace["leg"]["kind"] == "retrace" and retrace["event_index"] > moved, retrace["leg"]
            after = retrace["event_index"]
            velocity = self._step("velocity", after, {"kind": "velocity", "linear_mps": 0.2, "angular_rps": 0.0})
            assert velocity["event_index"] == after + 1 and len(velocity["executed_actions"]) == 1
            finish = self.client.post(f"/v1/sessions/{self.sid}/finish", json={"reason": "forced_stop"})
            assert finish.status_code == 200, finish.text
            assert self.client.delete(f"/v1/sessions/{self.sid}").status_code == 200
            self.sid = None
        finally:
            if self.sid is not None:
                self.client.delete(f"/v1/sessions/{self.sid}")
        legs = {name: self.replies[name]["leg"] for name in ("point_goal", "retrace")}
        return {"passed": True, "dataset": self.dataset, "camera": self.camera,
                "episode_id": self.episode_id, "goal_pixel": list(pixel), "legs": legs,
                "retrace_event_index": target,
                "events": {"start": 0, "after_point_goal": moved, "after_retrace": after,
                           "after_velocity": velocity["event_index"]},
                "velocity_actions": velocity["executed_actions"],
                "replies_scanned": sorted(self.replies), "privileged_keys": sorted(PRIVILEGED),
                "finish": finish.json().get("termination"), "model_inference": False}
