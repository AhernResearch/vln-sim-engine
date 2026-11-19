"""Offline clearance-constrained expert-path preparation, before any movement.

Queries an existing agent-radius navmesh without changing it. Clearance here is
extra distance to its non-navigable boundary, not distance to a raw mesh surface.
This prepares a candidate path; controller rollout and execution need their own
validation. No movement, model inference, mesh rebuilding, or retry occurs here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from sim_engine.core.cache import MISSING, BoundedMemo
from sim_engine.navigation.budget import checkpoint

from .corridor import CorridorGeometry
from .errors import ClearanceError
from .native import as_point, native_fair, native_search, native_swept_edge
from .queries import NavMeshQueries


@dataclass(frozen=True)
class ClearanceProfile:
    name: str = "navmesh-clearance-astar-smooth-v1"
    clearance_m: float = 0.15
    grid_m: float = 0.10
    edge_sample_m: float = 0.025
    resample_m: float = 0.05
    corridor_m: float = 0.75
    max_anchor_snap_m: float = 0.20
    max_height_step_m: float = 0.25
    floor_bin_m: float = 0.05
    smoothing_passes: int = 8
    max_expansions: int = 20000

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key.endswith("_m") and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be positive and finite")
        if self.edge_sample_m > self.grid_m:
            raise ValueError("edge sampling must be no coarser than the search grid")
        for key in ("smoothing_passes", "max_expansions"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")


def resample(points, spacing):
    output = [points[0]]
    for a, b in zip(points, points[1:]):
        count = max(1, math.ceil(float(np.linalg.norm(b - a)) / spacing))
        output.extend(a + (b - a) * i / count for i in range(1, count + 1))
    return output


class ClearancePlanner:
    """Bounded corridor A*, followed by clearance-checked shortcutting and fairing.

    `navmesh` provides snap_point, distance_to_closest_obstacle and
    try_step_no_sliding; `shortest_path(a,b)` returns ordered world XYZ points.
    Anchors retain their order and snapped locations, including repeated visits.
    Search, swept-edge checks and fairing run in the native kernel over the
    task-owned exact query cache; directed edge certificates are memoized here.
    """

    def __init__(self, navmesh, shortest_path, profile=None, *, queries=None, cache_entries=65536):
        self.navmesh, self.shortest_path = navmesh, shortest_path
        self.profile = profile or ClearanceProfile()
        self.queries = queries or NavMeshQueries(navmesh)
        if self.queries.navmesh is not navmesh:
            raise ValueError("query cache belongs to a different NavMesh")
        self.edge_memo = BoundedMemo(cache_entries)

    def clear_queries(self):
        """Release edge certificates; the owner of the shared query cache clears it."""
        self.edge_memo.clear()

    def clearance(self, p):
        try:
            return self.queries.clearance(p, max(2.0, self.profile.clearance_m * 2))
        except ValueError as error:
            raise ClearanceError(str(error)) from error

    def edge_clear(self, a, b):
        """Check the swept segment, not just its endpoints; never slide a candidate."""
        checkpoint(phase="swept_clearance")
        a, b = as_point(a), as_point(b)
        key = (self.profile, a.tobytes(), b.tobytes())
        cached = self.edge_memo.get(key)
        if cached is not MISSING:
            value, cost = cached
            checkpoint(query=cost)
            return value
        value, cost = self._edge_uncached(a, b)
        self.edge_memo.put(key, (value, cost))
        return value

    def _edge_uncached(self, a, b):
        return native_swept_edge(self, a, b)

    def _corridor(self, seed):
        return CorridorGeometry(seed, capacity=self.edge_memo.capacity)

    def _in_corridor(self, p, corridor):
        checkpoint(query=1)
        return corridor.contains(p, self.profile.corridor_m)

    def _corridor_all(self, points, corridor):
        # Bound speculative work: an invalid shortcut often fails near its start.
        # Preserve the logical query order and stop at the first rejection.
        for begin in range(0, len(points), 32):
            for valid in corridor.contains_many(points[begin:begin + 32], self.profile.corridor_m):
                checkpoint(query=1)
                if not valid:
                    return False
        return True

    def _search(self, start, goal, seed, corridor):
        return native_search(self, start, goal, seed)

    def _fair(self, points, corridor):
        return native_fair(self, points, corridor)

    def _smooth(self, path, corridor):
        # Visibility shortcuts cannot cross a blocked corner or leave the corridor.
        reduced, i = [path[0]], 0
        while i < len(path) - 1:
            checkpoint(phase="smoothing")
            chosen = i + 1
            for j in range(len(path) - 1, i, -1):
                checkpoint()
                if self._corridor_all(resample([path[i], path[j]], self.profile.grid_m), corridor):
                    if self.edge_clear(path[i], path[j]):
                        chosen = j
                        break
            reduced.append(path[chosen])
            i = chosen
        return self._fair(np.asarray(resample(reduced, self.profile.resample_m)), corridor)

    def prepare(self, anchors):
        originals = [as_point(a) for a in anchors]
        if len(originals) < 2:
            raise ClearanceError("at least two ordered anchors are required")
        snapped = [as_point(self.navmesh.snap_point(a.astype(np.float32))) for a in originals]
        residuals = [float(np.linalg.norm(a - b)) for a, b in zip(originals, snapped)]
        if max(residuals) > self.profile.max_anchor_snap_m:
            raise ClearanceError("anchor snap exceeds the source geometry tolerance")
        if any(self.clearance(p) + 1e-6 < self.profile.clearance_m for p in snapped):
            raise ClearanceError("fixed anchor has insufficient clearance; do not silently move it")
        route, mappings, reports = [snapped[0]], [0], []
        for index, (a, b) in enumerate(zip(snapped, snapped[1:])):
            if np.linalg.norm(a - b) < 1e-6:
                mappings.append(len(route) - 1)
                reports.append({"segment": index, "repeated_anchor": True, "expanded": 0})
                continue
            seed = np.asarray([as_point(p) for p in self.shortest_path(a, b)])
            if (
                len(seed) < 2
                or np.linalg.norm(seed[0] - a) > 1e-3
                or np.linalg.norm(seed[-1] - b) > 1e-3
            ):
                raise ClearanceError("shortest path did not preserve segment endpoints")
            safe = all(self.edge_clear(p, q) for p, q in zip(seed, seed[1:]))
            corridor = self._corridor(seed)
            if safe:
                initial, expanded = list(seed), 0
            else:
                initial, expanded = self._search(a, b, seed, corridor)
            path = self._smooth(initial, corridor)
            if not all(self.edge_clear(p, q) for p, q in zip(path, path[1:])):
                raise ClearanceError("smoothed path failed swept-segment verification")
            route.extend(path[1:])
            mappings.append(len(route) - 1)
            reports.append({"segment": index, "seed_satisfied_clearance": safe, "expanded": expanded})
        samples = resample(route, self.profile.edge_sample_m)
        clearances = [
            self.clearance(as_point(self.navmesh.snap_point(p.astype(np.float32)))) for p in samples
        ]
        return {
            "profile": asdict(self.profile),
            "world_points": np.asarray(route).tolist(),
            "anchor_indices": mappings,
            "anchor_snap_residuals_m": residuals,
            "minimum_sampled_boundary_clearance_m": min(clearances),
            "segments": reports,
            "path_length_m": float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()),
            "verification": "navmesh swept-segment and sampled-clearance queries only",
            "controller_execution": "not_checked",
            "training_admission": "pending_review",
        }
