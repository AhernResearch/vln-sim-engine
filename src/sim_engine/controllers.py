"""Control interfaces: each maps one validated step command to simulator primitives.

A controller only sees the local command, its own pose, the dataset physics and the
session's own observations and executed path. It never reads goals, reference paths
or rewards.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from sim_engine.navigation.budget import PlanningContext, PlanningStopped
from sim_engine.navigation.guidance import plan_with_preferences
from sim_engine.navigation.point_goal import TargetRejected, Unreachable, ground_point, intrinsics
from sim_engine.navigation.pose import next_action, reached, to_world
from sim_engine.navigation.tracking import expert_profile, make_tracker


class CommandRejected(ValueError):
    """A well-formed command that this session's physics cannot execute."""


@dataclass(frozen=True)
class Executed:
    action: Any
    image: Any
    collided: bool
    snapshot: Any = None
    position: Any = None


@dataclass(frozen=True)
class Outcome:
    executed: list[Executed]
    leg: dict | None = None  # set by leg controllers; the session numbers and records it


class Controller:
    kind: str

    def execute(self, env, physics, command, remaining: int, session) -> Outcome:
        raise NotImplementedError


class WaypointController(Controller):
    """Receding-horizon descent over native discrete actions (the original interface)."""

    kind = "waypoints"

    def execute(self, env, physics, command, remaining, session):
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
                                     env.capture(),
                                     env.position()))
        return Outcome(executed)


class VelocityController(Controller):
    """One unicycle velocity command integrated for the dataset time step.

    Follows habitat-lab 0.3.1 ``VelocityAction``: velocities are clipped to the
    configured ranges; a command below both minimum speeds is a STOP, which must be
    sent through ``finish`` instead of being executed as motion.
    """

    kind = "velocity"

    def execute(self, env, physics, command, remaining, session):
        limits = physics.velocity
        if limits is None:
            raise CommandRejected("dataset physics does not define velocity control")
        if remaining < 1:
            return Outcome([])
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
        return Outcome([Executed(action, image, collided, env.capture(), env.position())])


class LegController(Controller):
    """Plan a path, then track it with velocity commands until arrival or a budget.

    A rejected target or an infeasible plan is an ordinary outcome with no motion.
    Public leg fields describe only what the robot did; goals and paths are private.
    """

    def plan(self, env, command, session) -> tuple[list, list, dict]:
        raise NotImplementedError

    def execute(self, env, physics, command, remaining, session):
        legs = session.settings.simulation.legs
        profile = expert_profile({"expert_goal_radius_m": legs.goal_radius_m})
        limits = physics.velocity
        if limits is None or not math.isclose(limits.time_step_s, profile.dt_s):
            raise CommandRejected("legs need velocity physics at the tracker time step")
        origin = env.position()
        try:
            path, goal, private = self.plan(env, command, session)
        except TargetRejected as error:
            return Outcome([], {"kind": self.kind, "status": error.status, "reason": str(error),
                                "private": error.private})
        budget = min(command.max_actions, remaining)
        executed, status, detail = drive(env, physics, path, profile, budget)
        if status == "arrived":
            status = ("reached" if math.dist(env.position(), goal) <= legs.reached_tolerance_m
                      else "stopped_short")
        elif status == "budget":
            status = "leg_budget" if budget == command.max_actions else "episode_budget"
        travelled = sum(math.dist(a, b) for a, b in
                        zip([origin] + [e.position for e in executed], [e.position for e in executed]))
        return Outcome(executed, {
            "kind": self.kind, "status": status,
            "reason": "the robot got stuck following the path" if status == "blocked" else None,
            "distance_m": round(travelled, 3),
            "net_displacement_m": round(math.dist(origin, env.position()), 3),
            "collision_steps": sum(e.collided for e in executed),
            "private": dict(private, goal=goal, path=path, controller_detail=detail,
                            final_goal_distance_m=math.dist(env.position(), goal)),
        })


def drive(env, physics, path, profile, budget):
    """Track ``path`` for at most ``budget`` control steps; slides are counted, not fatal."""
    tracker = make_tracker(path, profile)
    executed = []
    while len(executed) < budget:
        try:
            command, _ = tracker.command(env.pose())
        except ValueError as error:
            return executed, "blocked", str(error)
        if command is None:
            return executed, "arrived", None
        image, collided = env.velocity_step(
            command["linear_mps"], command["angular_rps"], profile.dt_s, physics.allow_sliding)
        action = {"linear_mps": command["linear_mps"], "angular_rps": command["angular_rps"]}
        executed.append(Executed(action, image, collided, env.capture(), env.position()))
    return executed, "budget", None


class PointGoalController(LegController):
    kind = "point_goal"

    def plan(self, env, command, session):
        legs = session.settings.simulation.legs
        view = session.snapshot(command.observation_id).views[command.view]
        depth = view.images["depth"]
        height, width = depth.shape[:2]
        k = intrinsics(view.calibration["projections"]["depth"], width, height)
        geometry = session.geometry()
        geometry.clear()
        start = env.position()
        budget = SimpleNamespace(wall_s=legs.planning_wall_s, max_queries=2_000_000,
                                 max_expansions=legs.planning_max_expansions)
        try:
            with PlanningContext(budget, threading.Event()).bind():
                grounded = ground_point(depth, k, view.sensors["depth"], start, geometry.pf,
                                        geometry.guard, command.point, legs)
                plan = plan_with_preferences(geometry.planners, geometry.guard, [start],
                                             grounded["goal"], legs.start_adjustment_m, 0.0)
        except PlanningStopped as error:
            raise Unreachable("planning ran out of time for that spot",
                              {"error": str(error)}) from error
        except TargetRejected:
            raise
        except ValueError as error:
            raise Unreachable("no path with enough clearance to that spot",
                              {"error": str(error)}) from error
        if plan["status"] != "feasible":
            raise Unreachable("no path with enough clearance to that spot",
                              {"grounding": grounded, "planning": plan})
        private = {"grounding": grounded, "planning": {
            k: plan[k] for k in ("planner_index", "comfort_fallback_used", "guide_adjustments")}}
        return plan["path"], plan["fixed_goal"], private


class RetraceController(LegController):
    kind = "retrace"

    def plan(self, env, command, session):
        if not 1 <= command.leg_id <= len(session.legs):
            raise TargetRejected("there is no leg with that id")
        end = session.legs[command.leg_id - 1]["end_event_index"]
        path = []
        for point in reversed(session.positions[end:]):
            if not path or math.dist(point, path[-1]) >= 0.05:
                path.append(list(point))
        if len(path) < 2:
            raise TargetRejected("the robot is already where that leg ended")
        return path, path[-1], {"retrace_of_leg": command.leg_id}


CONTROLLERS: dict[str, Controller] = {
    controller.kind: controller for controller in
    (WaypointController(), VelocityController(), PointGoalController(), RetraceController())
}
