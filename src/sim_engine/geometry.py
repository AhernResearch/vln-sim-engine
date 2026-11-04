"""Pure SE(2) receding-horizon control, independent of the simulator."""

import math
from dataclasses import dataclass


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float


def to_world(origin: Pose, point: list[float]) -> Pose:
    c, s = math.cos(origin.yaw), math.sin(origin.yaw)
    return Pose(
        origin.x + c * point[0] - s * point[1],
        origin.y + s * point[0] + c * point[1],
        wrap(origin.yaw + point[2]),
    )


def next_action(current: Pose, target: Pose, forward_m=0.25, turn_degrees=15) -> str | None:
    """One-step kinematic descent; no scene or task-goal access."""

    def error(pose):
        return ((pose.x - target.x) ** 2 + (pose.y - target.y) ** 2) / forward_m**2 + (
            wrap(pose.yaw - target.yaw) / math.radians(turn_degrees)
        ) ** 2

    candidates = {
        "MOVE_FORWARD": Pose(
            current.x + forward_m * math.cos(current.yaw),
            current.y + forward_m * math.sin(current.yaw),
            current.yaw,
        ),
        "TURN_LEFT": Pose(current.x, current.y, wrap(current.yaw + math.radians(turn_degrees))),
        "TURN_RIGHT": Pose(current.x, current.y, wrap(current.yaw - math.radians(turn_degrees))),
    }
    action = min(candidates, key=lambda name: error(candidates[name]))
    return action if error(candidates[action]) < error(current) - 1e-9 else None


def reached(current: Pose, target: Pose, forward_m=0.25, turn_degrees=15) -> bool:
    # Quaternion conversion and angle wrapping perturb exact half-action boundaries.
    # At a boundary neither native action strictly reduces error, so it must count
    # as reached rather than leaving the controller with no admissible action.
    return math.hypot(current.x - target.x, current.y - target.y) <= forward_m / 2 + 1e-6 and abs(
        wrap(current.yaw - target.yaw)
    ) <= math.radians(turn_degrees / 2) + 1e-6
