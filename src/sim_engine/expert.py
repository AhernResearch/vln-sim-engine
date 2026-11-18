"""Expert rollout: follow a privileged reference path with velocity control.

Lives on the simulator side because reference paths are privileged data. Path
planning and tracking are pure SE(2) geometry (unit-testable without Habitat);
``ExpertRollout`` is the only part that touches the environment.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .atomic import atomic_json
from .controllers import CONTROLLERS
from .geometry import Pose, wrap
from .protocol import VelocityCommand


@dataclass(frozen=True)
class ControllerProfile:
    """Versioned control profile; its fields are recorded with every rollout."""

    name: str = "path_follow_4hz_v2"
    dt_s: float = 0.25
    v_max_mps: float = 0.6
    omega_max_rps: float = 1.26
    lookahead_m: float = 0.5
    turn_in_place_deg: float = 35.0
    goal_radius_m: float = 0.25
    max_cross_track_m: float = 0.3
    turn_gain: float = 2.0
    resample_m: float = 0.05
    # Projection only searches this far past the progress made so far, so routes that
    # revisit a place (RxR) cannot snap the agent onto a later or earlier pass.
    search_window_m: float = 1.0
    # Longer than a 180-degree in-place turn (10 steps), so only wedged agents trip it.
    stall_steps: int = 16

    def command(self, heading_error: float) -> dict:
        """One unicycle command from the signed heading error to the lookahead point."""
        error = math.degrees(abs(heading_error))
        if error > self.turn_in_place_deg:
            return {"linear_mps": 0.0, "angular_rps": math.copysign(self.omega_max_rps, heading_error)}
        return {
            "linear_mps": self.v_max_mps * max(math.cos(heading_error), 0.0),
            "angular_rps": max(
                min(self.turn_gain * heading_error, self.omega_max_rps), -self.omega_max_rps
            ),
        }


class Polyline:
    """Dense, arc-length parametrized path in world SE(2)."""

    def __init__(self, points):
        if len(points) < 2:
            raise ValueError("path needs at least two points")
        self.points = [(float(x), float(y)) for x, y, *_ in points]
        self.cumulative = [0.0]
        for (ax, ay), (bx, by) in zip(self.points, self.points[1:]):
            self.cumulative.append(self.cumulative[-1] + math.hypot(bx - ax, by - ay))
        if self.length <= 0:
            raise ValueError("path has zero length")

    @property
    def length(self) -> float:
        return self.cumulative[-1]

    def point_at(self, arc: float):
        arc = max(0.0, min(arc, self.length))
        low, high = 0, len(self.cumulative) - 1
        while low + 1 < high:
            middle = (low + high) // 2
            if self.cumulative[middle] <= arc:
                low = middle
            else:
                high = middle
        span = self.cumulative[high] - self.cumulative[low]
        ratio = 0.0 if span <= 0 else (arc - self.cumulative[low]) / span
        (ax, ay), (bx, by) = self.points[low], self.points[high]
        return ax + ratio * (bx - ax), ay + ratio * (by - ay)

    def project(self, x: float, y: float, start: float = 0.0, end: float = math.inf):
        """Nearest arc length, distance and path heading among segments overlapping [start, end]."""
        best = (-1.0, math.inf, 0.0)
        for index, ((ax, ay), (bx, by)) in enumerate(zip(self.points, self.points[1:])):
            if self.cumulative[index + 1] < start or self.cumulative[index] > end:
                continue
            dx, dy = bx - ax, by - ay
            span = dx * dx + dy * dy
            ratio = 0.0 if span <= 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / span))
            distance = math.hypot(x - (ax + ratio * dx), y - (ay + ratio * dy))
            if distance < best[1]:
                best = (self.cumulative[index] + ratio * math.sqrt(span), distance, math.atan2(dy, dx))
        return best

    def resample(self, spacing: float) -> Polyline:
        steps = max(1, math.ceil(self.length / spacing))
        return Polyline([(*self.point_at(i * self.length / steps), 0.0) for i in range(steps + 1)])


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


class UnicycleTracker:
    """Pure pursuit over a polyline; emits one velocity command per control step."""

    def __init__(self, path: Polyline, profile: ControllerProfile):
        self.path, self.profile, self.max_progress, self.idle = path, profile, 0.0, 0

    def command(self, pose: Pose):
        """Next command plus the cross-track error at this pose."""
        arc, cross_track, _ = self.path.project(
            pose.x, pose.y, self.max_progress, self.max_progress + self.profile.search_window_m
        )
        if cross_track > self.profile.max_cross_track_m:
            raise ValueError(f"cross-track error {cross_track:.3f} m exceeds limit")
        self.idle = 0 if arc > self.max_progress + 1e-3 else self.idle + 1
        if self.idle > self.profile.stall_steps:
            raise ValueError(f"stalled: no path progress for {self.idle} steps")
        self.max_progress = max(self.max_progress, arc)
        if self.path.length - arc <= self.profile.goal_radius_m:
            return None, cross_track
        target = self.path.point_at(min(arc + self.profile.lookahead_m, self.path.length))
        heading = math.atan2(target[1] - pose.y, target[0] - pose.x)
        return self.profile.command(wrap(heading - pose.yaw)), cross_track


@dataclass(frozen=True)
class RolloutResult:
    termination: str
    steps: int
    stats: dict


def jpeg_encoder(quality=95):
    """Payload encoder for the writer: RGB8 array -> JPEG bytes."""
    from io import BytesIO

    import numpy as np
    from PIL import Image

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
        atomic_json(self.root / "states.json", {"schema": "resim_states_v1", "rows": self.states})
        atomic_json(
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
                    self.env, self.env.physics, VelocityCommand(kind="velocity", **command), 1
                )
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
