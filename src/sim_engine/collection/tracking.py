"""One expert profile for read-only prediction and actual 4Hz execution.

Preview integration follows the pinned kinematic_preview recipe. The source
snapshot and its provenance remain frozen; no neighboring runtime is modified.
"""

import math

from sim_engine.planning.context import checkpoint


def expert_profile(config):
    from dataclasses import replace

    from sim_engine.expert import ControllerProfile

    return replace(
        ControllerProfile(), name="pose_path_follow_4hz_v3", goal_radius_m=config["expert_goal_radius_m"]
    )


def make_tracker(world_path, profile):
    from sim_engine.expert import Polyline, UnicycleTracker

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
    import habitat_sim as hs
    import magnum as mn
    import numpy as np

    from sim_engine.geometry import Pose

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
