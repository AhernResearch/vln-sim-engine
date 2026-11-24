"""The shared path-following loop: termination order and hook control."""

from sim_engine.navigation.driving import StepHooks, follow


class Line:
    """A tracker that arrives after ``arrive_after`` commands, or fails at ``fail_at``."""

    def __init__(self, arrive_after=3, fail_at=None):
        self.calls, self.arrive_after, self.fail_at = 0, arrive_after, fail_at

    def command(self, pose):
        self.calls += 1
        if self.calls - 1 == self.fail_at:
            raise ValueError("stalled")
        return (None, 0.0) if pose >= self.arrive_after else ({"linear_mps": 1.0, "angular_rps": 0.0}, 0.0)


class Robot:
    def __init__(self):
        self.position = 0

    def pose(self):
        return self.position

    def step(self, command):
        self.position += 1
        return "image", False


def walk(tracker, budget, hooks=None, **kwargs):
    robot = Robot()
    return follow(robot, tracker, robot.step, budget, hooks, **kwargs)


def test_arrival_budget_and_blocked_are_distinct():
    assert walk(Line(3), 10).status == "arrived" and walk(Line(3), 10).steps == 3
    assert (walk(Line(3), 2).status, walk(Line(3), 2).steps) == ("budget", 2)
    blocked = walk(Line(3, fail_at=1), 10)
    assert (blocked.status, blocked.steps, blocked.detail) == ("blocked", 1, "stalled")


def test_arriving_on_the_last_step_counts_only_when_the_tracker_is_consulted_at_the_budget():
    assert walk(Line(3), 3).status == "budget"
    assert walk(Line(3), 3, track_at_budget=True).status == "arrived"


def test_hooks_see_every_step_in_order_and_can_end_the_walk():
    seen = []

    class Hooks(StepHooks):
        def observe(self, step, pose):
            seen.append(("observe", step))

        def dispatch(self, step, command):
            seen.append(("dispatch", step))

        def executed(self, step, command, image, collided, cross_track):
            seen.append(("executed", step))
            return "collision_or_clearance" if step == 2 else None

    result = walk(Line(5), 10, Hooks())
    assert (result.status, result.steps) == ("collision_or_clearance", 2)
    assert seen == [("observe", 0), ("dispatch", 0), ("executed", 1),
                    ("observe", 1), ("dispatch", 1), ("executed", 2)]
