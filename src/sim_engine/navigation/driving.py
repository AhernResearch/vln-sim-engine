"""One path-following loop for legs, expert rollouts and training collection.

Callers differ only in what they record around each control step (``StepHooks``) and in how a
command reaches the simulator (``execute``); the tracker, termination order and budget live here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Followed:
    status: str  # arrived | budget | blocked | a termination returned by a hook
    steps: int  # control steps executed
    detail: str | None = None


class StepHooks:
    """Recording around each control step. A returned string ends the walk with that status."""

    def observe(self, step: int, pose) -> str | None:
        """Before the tracker reads ``pose``; ``step`` commands have executed so far."""
        return None

    def dispatch(self, step: int, command: dict) -> None:
        """Right before ``command`` is executed."""

    def executed(self, step: int, command: dict, image, collided: bool, cross_track: float) -> str | None:
        """After the ``step``-th command (counting from 1) has executed."""
        return None


def follow(env, tracker, execute, budget: int, hooks: StepHooks | None = None, *,
           track_at_budget: bool = False) -> Followed:
    """Drive ``tracker`` until it arrives, fails, a hook ends the walk or ``budget`` steps ran.

    A tracker error (stall, cross-track) is ``blocked``. With ``track_at_budget`` the tracker is
    still consulted once the budget is spent, so arriving on the last step counts as arrival.
    """
    hooks = hooks or StepHooks()
    step = 0
    while True:
        if step == budget and not track_at_budget:
            return Followed("budget", step)
        pose = env.pose()
        if (end := hooks.observe(step, pose)) is not None:
            return Followed(end, step)
        try:
            command, cross_track = tracker.command(pose)
        except ValueError as error:
            return Followed("blocked", step, str(error))
        if command is None:
            return Followed("arrived", step)
        if step == budget:
            return Followed("budget", step)
        hooks.dispatch(step, command)
        image, collided = execute(command)
        step += 1
        if (end := hooks.executed(step, command, image, collided, cross_track)) is not None:
            return Followed(end, step)
