"""Control interfaces: each maps one validated step command to simulator primitives.

A controller only sees the local command, its own pose and the dataset physics. It
never reads goals, reference paths or rewards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sim_engine.navigation.pose import next_action, reached, to_world


class CommandRejected(ValueError):
    """A well-formed command that this session's physics cannot execute."""


@dataclass(frozen=True)
class Executed:
    action: Any
    image: Any
    collided: bool
    snapshot: Any = None
    position: Any = None


class Controller:
    kind: str

    def execute(self, env, physics, command, remaining: int) -> list[Executed]:
        raise NotImplementedError


class WaypointController(Controller):
    """Receding-horizon descent over native discrete actions (the original interface)."""

    kind = "waypoints"

    def execute(self, env, physics, command, remaining):
        origin = env.pose()
        targets = [to_world(origin, point) for point in command.trajectory]
        executed, cursor = [], 0
        for _ in range(min(command.action_budget, remaining)):
            pose = env.pose()
            while cursor < len(targets) and reached(
                pose, targets[cursor], physics.forward_m, physics.turn_degrees
            ):
                cursor += 1
            if cursor == len(targets):
                break
            action = next_action(pose, targets[cursor], physics.forward_m, physics.turn_degrees)
            if action is None:
                break
            image, collided = env.step(action)
            executed.append(Executed(action, image, collided,
                                     env.capture() if hasattr(env, "capture") else None,
                                     env.position()))
        return executed


class VelocityController(Controller):
    """One unicycle velocity command integrated for the dataset time step.

    Follows habitat-lab 0.3.1 ``VelocityAction``: velocities are clipped to the
    configured ranges; a command below both minimum speeds is a STOP, which must be
    sent through ``finish`` instead of being executed as motion.
    """

    kind = "velocity"

    def execute(self, env, physics, command, remaining):
        limits = physics.velocity
        if limits is None:
            raise CommandRejected("dataset physics does not define velocity control")
        if remaining < 1:
            return []
        linear = min(max(command.linear_mps, limits.linear_mps[0]), limits.linear_mps[1])
        angular_dps = min(
            max(math.degrees(command.angular_rps), limits.angular_dps[0]), limits.angular_dps[1]
        )
        if abs(linear) < limits.min_abs_linear_mps and abs(angular_dps) < limits.min_abs_angular_dps:
            raise CommandRejected("sub-threshold velocity is a STOP; call finish instead")
        image, collided = env.velocity_step(
            linear, math.radians(angular_dps), limits.time_step_s, physics.allow_sliding
        )
        action = {"linear_mps": linear, "angular_rps": math.radians(angular_dps)}
        return [Executed(action, image, collided,
                                     env.capture() if hasattr(env, "capture") else None,
                                     env.position())]


CONTROLLERS: dict[str, Controller] = {
    controller.kind: controller for controller in (WaypointController(), VelocityController())
}
