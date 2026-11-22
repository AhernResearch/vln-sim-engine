"""Expert rollout: follow a privileged reference path with velocity control.

Lives on the simulator side because reference paths are privileged data. Path
planning and tracking are pure SE(2) geometry (unit-testable without Habitat);
``ExpertRollout`` is the only part that touches the environment.
"""

from __future__ import annotations

import hashlib
import shutil
import uuid
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from sim_engine.core.io import replace_json

from .controllers import CONTROLLERS
from .navigation.tracking import ControllerProfile, Polyline, UnicycleTracker
from .protocol import VelocityCommand


class PathPlanner:
    """Reference path -> dense navmesh polyline in the base SE(2) frame, ending at the goal.

    Consecutive waypoints are joined by navmesh geodesics, so the route never cuts
    through obstacles; the environment supplies ``snap`` and ``shortest_path``.
    """

    def __init__(self, environment):
        self.env = environment

    def densify(self, points, goal):
        anchors = [self.env.snap(p) for p in [*points, goal]]
        route = [anchors[0]]
        for start, end in zip(anchors, anchors[1:]):
            try:
                route.extend(self.env.shortest_path(start, end)[1:])
            except RuntimeError as error:
                raise ValueError("reference path is not connected on the navmesh") from error
        # Habitat (x, y, z) -> base frame used by ``pose()``: forward = -z, left = -x.
        return [(-p[2], -p[0], 0.0) for p in route]


@dataclass(frozen=True)
class RolloutResult:
    termination: str
    steps: int
    stats: dict


def jpeg_encoder(quality=95):
    """Payload encoder for the writer: RGB8 array -> JPEG bytes."""

    def encode(image):
        buffer = BytesIO()
        Image.fromarray(np.ascontiguousarray(image)).save(buffer, "JPEG", quality=quality)
        return buffer.getvalue()

    return encode


class RolloutWriter:
    """Appends frames and states to an episode directory, then seals it."""

    def __init__(self, root, camera, profile, encode):
        self.root, self.camera, self.profile, self.encode = root, camera, profile, encode
        self.states = []
        self.frames_sha256 = hashlib.sha256()

    def add(self, image, pose, command, collided, cross_track, progress):
        index = len(self.states)
        payload = self.encode(image)
        self.frames_sha256.update(hashlib.sha256(payload).digest())
        target = self.root / f"frames/{index:05d}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        self.states.append(
            {
                "step": index,
                "t_s": round(index * self.profile.dt_s, 6),
                "base_pose_se2": [pose.x, pose.y, pose.yaw],
                "command": command,
                "collided": bool(collided),
                "cross_track_m": round(cross_track, 4),
                "path_progress_m": round(progress, 4),
            }
        )

    def seal(self, receipt):
        replace_json(self.root / "states.json", {"schema": "resim_states_v1", "rows": self.states})
        replace_json(
            self.root / "receipt.json",
            {**receipt, "frames": len(self.states), "frames_sha256": self.frames_sha256.hexdigest()},
        )


class ExpertRollout:
    """Drives one episode along its reference path; never reads a policy output."""

    def __init__(self, environment, profile=None, planner_factory=PathPlanner):
        self.env, self.profile, self.planner_factory = (
            environment,
            profile or ControllerProfile(),
            planner_factory,
        )

    def run(self, episode, camera, writer_root, encode, max_steps):
        """Execute the episode and return a result; raises ValueError on failure."""
        writer_root = Path(writer_root)
        image = self.env.reset(episode, camera, 0)
        self._check_physics()
        pose = self.env.pose()
        try:
            points = self.planner_factory(self.env).densify(
                episode["reference_path"], episode["goals"][0]["position"]
            )
        except ValueError as error:
            raise ValueError(f"path planning failed: {error}") from error
        path = Polyline(points).resample(self.profile.resample_m)
        tracker = UnicycleTracker(path, self.profile)
        controller = CONTROLLERS["velocity"]
        writer = RolloutWriter(writer_root, camera, self.profile, encode)
        writer.add(image, pose, None, False, 0.0, 0.0)
        collisions = 0
        goal = episode["goals"][0]["position"]
        for step in range(1, max_steps + 1):
            command, cross_track = tracker.command(pose)
            if command is None:
                writer.seal(self._receipt(episode, "expert_stop", step - 1))
                return RolloutResult("expert_stop", step - 1, self._stats(tracker, collisions, goal))
            try:
                executed = controller.execute(
                    self.env, self.env.physics, VelocityCommand(kind="velocity", **command), 1, None
                ).executed
            except Exception as error:  # sub-threshold or missing physics
                raise ValueError(f"command rejected: {error}") from error
            if not executed:
                raise ValueError("controller executed nothing")
            pose, collisions = self.env.pose(), collisions + int(executed[0].collided)
            writer.add(
                executed[0].image, pose, command, executed[0].collided, cross_track, tracker.max_progress
            )
        writer.seal(self._receipt(episode, "budget_exhausted", max_steps))
        return RolloutResult("budget_exhausted", max_steps, self._stats(tracker, collisions, goal))

    def _check_physics(self):
        """The recorded dt must be the dt the controller integrates, or rows lie."""
        limits = self.env.physics.velocity
        if limits is None:
            raise ValueError("expert rollout requires dataset velocity physics")
        if abs(limits.time_step_s - self.profile.dt_s) > 1e-9:
            raise ValueError(
                f"physics time step {limits.time_step_s} s does not match profile dt {self.profile.dt_s} s"
            )

    def _stats(self, tracker, collisions, goal):
        """Small summary only: the frames stay on the worker's disk, not on the wire."""
        return {
            "path_length_m": round(tracker.path.length, 3),
            "max_progress_m": round(tracker.max_progress, 3),
            "goal_distance_m": round(self.env.distance(self.env.position(), goal), 4),
            "collisions": collisions,
        }

    def _receipt(self, episode, termination, steps):
        return {
            "schema": "resim_receipt_v1",
            "episode_id": str(episode["episode_id"]),
            "termination": termination,
            "control_steps": steps,
            "controller_profile": asdict(self.profile),
        }


def expert_job(environment, episode, camera, root: Path, max_steps: int, encode=None) -> dict:
    """Run one episode into ``root/<dataset>/<episode_id>`` atomically; never overwrites."""
    target = Path(root) / episode["_dataset"] / str(episode["episode_id"])
    if target.exists():
        raise FileExistsError("expert output already exists")
    partial = target.parent / f".partial-{uuid.uuid4().hex}"
    try:
        result = ExpertRollout(environment).run(
            episode, camera, partial, encode or jpeg_encoder(), max_steps
        )
        partial.rename(target)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    return {
        "episode_id": str(episode["episode_id"]),
        "output": str(target),
        "termination": result.termination,
        "control_steps": result.steps,
        **result.stats,
    }
