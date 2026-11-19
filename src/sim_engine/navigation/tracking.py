"""Pure-pursuit path tracking shared by read-only preview and actual 4Hz execution.

Preview integration follows the pinned kinematic_preview recipe. The source
snapshot and its provenance remain frozen; no neighboring runtime is modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from sim_engine.core.optional import optional_module

from .budget import checkpoint
from .pose import Pose, wrap


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


def expert_profile(config):
    return replace(
        ControllerProfile(), name="pose_path_follow_4hz_v3", goal_radius_m=config["expert_goal_radius_m"]
    )


def make_tracker(world_path, profile):
    path = Polyline([(-p[2], -p[0], 0) for p in world_path]).resample(profile.resample_m)
    return EndpointTracker(UnicycleTracker(path, profile), profile.dt_s)


class EndpointTracker:
    """Cap the last translation so one control step cannot overshoot the endpoint."""

    def __init__(self, tracker, dt):
        self.tracker, self.dt = tracker, dt

    def command(self, pose):
        command, error = self.tracker.command(pose)
        if command is not None:
            remaining = max(0.0, self.tracker.path.length - self.tracker.max_progress)
            command = dict(command, linear_mps=min(command["linear_mps"], remaining / self.dt))
        return command, error


def preview(pf, episode, world_path, profile, max_steps):
    hs = optional_module("habitat_sim")
    mn = optional_module("magnum")

    tracker = make_tracker(world_path, profile)
    rotation = mn.Quaternion(mn.Vector3(*episode["start_rotation"][:3]), episode["start_rotation"][3])
    current = hs.RigidState(rotation, np.asarray(episode["start_position"], dtype=np.float32))
    velocity = hs.physics.VelocityControl()
    velocity.controlling_lin_vel = velocity.controlling_ang_vel = True
    velocity.lin_vel_is_local = velocity.ang_vel_is_local = True
    rows = []

    def record(command, collided):
        pos = np.asarray(current.translation)
        forward = current.rotation.transform_vector(mn.Vector3(0, 0, -1))
        yaw = math.atan2(-forward[0], -forward[2])
        rows.append(
            {
                "frame": len(rows),
                "world_position": pos.tolist(),
                "yaw": yaw,
                "command": command,
                "collided": collided,
                "boundary_clearance_m": float(pf.distance_to_closest_obstacle(pos, 2)),
            }
        )
        return Pose(-float(pos[2]), -float(pos[0]), yaw)

    pose = record(None, False)
    termination = "budget_exhausted"
    for _ in range(max_steps):
        checkpoint(phase="preview")
        try:
            command, _ = tracker.command(pose)
        except ValueError as error:
            termination = str(error)
            break
        if command is None:
            termination = "expert_stop"
            break
        velocity.linear_velocity = np.asarray([0, 0, -command["linear_mps"]])
        velocity.angular_velocity = np.asarray([0, command["angular_rps"], 0])
        desired = velocity.integrate_transform(profile.dt_s, current)
        final = pf.try_step(current.translation, desired.translation)
        before = (desired.translation - current.translation).dot()
        after = (final - current.translation).dot()
        current = hs.RigidState(desired.rotation, final)
        pose = record(command, bool(after + 1e-5 < before))
    return {
        "termination": termination,
        "frames": len(rows),
        "rows": rows,
        "collision_steps": sum(r["collided"] for r in rows),
        "mode": "read_only_pathfinder_preview_with_shared_expert_profile",
    }
