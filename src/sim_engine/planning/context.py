"""One budget across grounding, all A* candidates, smoothing and previews."""

import contextvars
import time
from contextlib import contextmanager

_current = contextvars.ContextVar("planning_context", default=None)


class PlanningStopped(Exception):
    def __init__(self, outcome):
        self.outcome = outcome
        super().__init__(outcome)


class PlanningContext:
    def __init__(self, budget, stop, progress=None):
        self.budget, self.stop, self.progress = budget, stop, progress
        self.deadline = time.monotonic() + budget.wall_s
        self.queries = self.expansions = 0
        self.phase = "initializing"
        self.next_publish = 0.0
        self.geometry_queries = None

    def check(self, *, query=0, expansion=0, phase=None):
        self.queries += query
        self.expansions += expansion
        if phase is not None:
            self.phase = phase
        if self.progress is not None:
            # Publish at most ten times per second; avoid a shared-memory lock per query.
            now = time.time()
            if now >= self.next_publish:
                self.progress[:] = [self.queries, self.expansions, now]
                self.next_publish = now + 0.1
        if self.stop.is_set():
            raise PlanningStopped("cancelled")
        if (time.monotonic() >= self.deadline or self.queries > self.budget.max_queries
                or self.expansions > self.budget.max_expansions):
            raise PlanningStopped("budget_exhausted")

    @contextmanager
    def bind(self):
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)


def checkpoint(*, query=0, expansion=0, phase=None):
    context = _current.get()
    if context is not None:
        context.check(query=query, expansion=expansion, phase=phase)
