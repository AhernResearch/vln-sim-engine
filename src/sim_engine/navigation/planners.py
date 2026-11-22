"""Clearance planners over one NavMesh: comfort, minimum-clearance fallback and the guard."""

from dataclasses import replace

import numpy as np

from sim_engine.core.optional import optional_module

from .budget import checkpoint
from .clearance import ClearancePlanner, ClearanceProfile
from .errors import RejectedCandidate
from .queries import NavMeshQueries


class NavMeshPlanners:
    """``planners`` are tried in order by ``plan_with_preferences``; ``guard`` checks execution."""

    def __init__(self, pathfinder, minimum_clearance_m, profile=None):
        self.pf = pathfinder
        profile = profile or ClearanceProfile()
        # One exact query cache per NavMesh, shared by every planner of the current task.
        self.queries = NavMeshQueries(pathfinder)
        self.planner = self._planner(profile)
        self.planners = [self.planner]
        if minimum_clearance_m < profile.clearance_m:
            self.planners.append(self._planner(
                replace(profile, clearance_m=minimum_clearance_m, grid_m=min(profile.grid_m, 0.05))))
        self.guard = self._planner(
            replace(profile, clearance_m=minimum_clearance_m, edge_sample_m=0.0125))

    def _planner(self, profile):
        return ClearancePlanner(self.pf, self.shortest, profile, queries=self.queries)

    def shortest(self, a, b):
        checkpoint(query=1)
        path = optional_module("habitat_sim").ShortestPath()
        path.requested_start, path.requested_end = a, b
        if not self.pf.find_path(path):
            raise RejectedCandidate("anchors disconnected")
        return [np.asarray(p) for p in path.points]

    def clear(self):
        # Share queries between a task's candidate searches, never across tasks.
        for planner in [*self.planners, self.guard]:
            planner.clear_queries()
        self.queries.clear()
