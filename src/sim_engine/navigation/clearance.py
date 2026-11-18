"""Offline clearance-constrained expert-path preparation, before any movement.

Queries an existing agent-radius navmesh without changing it. Clearance here is
extra distance to its non-navigable boundary, not distance to a raw mesh surface.
This prepares a candidate path; controller rollout and execution need their own
validation. No movement, model inference, mesh rebuilding, or retry occurs here.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np

from sim_engine.core.cache import MISSING, BoundedMemo
from sim_engine.navigation.budget import checkpoint

from .corridor import CorridorGeometry
from .errors import ClearanceError
from .native import as_point, native_fair, native_search, native_swept_edge, require_native


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


def _resample(points, spacing):
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
    """

    def __init__(self, navmesh, shortest_path, profile=None, *, optimize=True, cache_entries=65536,
                 search_backend="python", query_cache=None, native_path=False):
        self.navmesh, self.shortest_path = navmesh, shortest_path
        self.profile = profile or ClearanceProfile()
        self._clearance_cache = {}
        self.optimize = optimize
        self.edge_memo = BoundedMemo(cache_entries)
        if search_backend not in {"python", "native"}:
            raise ValueError("unknown clearance search backend")
        self.search_backend = search_backend
        if query_cache is not None and query_cache.navmesh is not navmesh:
            raise ValueError("query cache belongs to a different NavMesh")
        if native_path and query_cache is None:
            raise ValueError("native path processing requires a task query cache")
        self.query_cache, self.native_path = query_cache, native_path
        if search_backend == "native":

            require_native()

    def clear_queries(self):
        self._clearance_cache.clear()
        self.edge_memo.clear()

    def clearance(self, p):
        if self.query_cache is not None:
            try:
                return self.query_cache.clearance(p, max(2.0, self.profile.clearance_m*2))
            except ValueError as error:
                raise ClearanceError(str(error)) from error
        checkpoint(query=1)
        key = tuple(float(v) for v in p)
        if key not in self._clearance_cache:
            distance = float(
                self.navmesh.distance_to_closest_obstacle(
                    np.asarray(p, dtype=np.float32), max(2.0, self.profile.clearance_m * 2)
                )
            )
            if not math.isfinite(distance) or distance < 0:
                raise ClearanceError("invalid navmesh clearance")
            self._clearance_cache[key] = distance
        return self._clearance_cache[key]

    def edge_clear(self, a, b):
        """Check the swept segment, not just its endpoints; never slide a candidate."""
        checkpoint(phase="swept_clearance")
        a, b = as_point(a), as_point(b)
        if not self.optimize:
            return self._edge_uncached(a, b)[0]
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
        if self.native_path:

            return native_swept_edge(self, a, b)
        points = _resample([a, b], self.profile.edge_sample_m)
        queries = 0
        previous = a
        for p in points:
            try:
                snapped = as_point(self.navmesh.snap_point(np.asarray(p, dtype=np.float32)))
            except ClearanceError:
                return False, queries
            if np.linalg.norm(snapped[[0, 2]] - p[[0, 2]]) > 1e-4:
                return False, queries
            if abs(snapped[1] - p[1]) > self.profile.max_height_step_m:
                return False, queries
            queries += 1
            if self.clearance(snapped) + 1e-6 < self.profile.clearance_m:
                return False, queries
            try:
                reached = as_point(
                    self.navmesh.try_step_no_sliding(
                        np.asarray(previous, dtype=np.float32), np.asarray(snapped, dtype=np.float32)
                    )
                )
            except ClearanceError:
                return False, queries
            if np.linalg.norm(reached - snapped) > 1e-3:
                return False, queries
            previous = snapped
        return True, queries

    def _in_corridor(self, p, seed):
        checkpoint(query=1)
        if isinstance(seed, CorridorGeometry):
            return seed.contains(p, self.profile.corridor_m)
        a, b = seed[:-1], seed[1:]
        delta = b - a
        length2 = np.sum(delta * delta, axis=1)
        t = np.clip(np.sum((p - a) * delta, axis=1) / np.maximum(length2, 1e-12), 0, 1)
        return (
            float(np.linalg.norm(p - (a + t[:, None] * delta), axis=1).min()) <= self.profile.corridor_m
        )

    def _corridor_all(self, points, seed):
        if not isinstance(seed, CorridorGeometry):
            return all(self._in_corridor(p, seed) for p in points)
        # Bound speculative work: an invalid shortcut often fails near its start.
        # Preserve the old logical query order and stop at the first rejection.
        for begin in range(0, len(points), 32):
            checks = seed.contains_many(points[begin:begin + 32], self.profile.corridor_m)
            for valid in checks:
                checkpoint(query=1)
                if not valid:
                    return False
        return True

    def _search(self, start, goal, seed):
        profile = self.profile
        serial = itertools.count()
        first = (0, 0, round(float(start[1]) / profile.floor_bin_m))
        points, parent, costs = {first: start}, {}, {first: 0.0}
        queue = [(float(np.linalg.norm(goal - start)), next(serial), 0.0, first)]
        expanded = 0
        offsets = [(x, z) for x in (-1, 0, 1) for z in (-1, 0, 1) if x or z]
        while queue:
            checkpoint(expansion=1, phase="astar")
            _, _, cost, key = heapq.heappop(queue)
            if cost != costs[key]:
                continue
            current = points[key]
            expanded += 1
            if expanded > profile.max_expansions:
                raise ClearanceError("clearance search expansion budget exhausted")
            if np.linalg.norm(current - goal) <= profile.grid_m * 1.5 and self.edge_clear(current, goal):
                route = [goal, current]
                while key in parent:
                    key = parent[key]
                    route.append(points[key])
                return list(reversed(route)), expanded
            for dx, dz in offsets:
                ix, iz = key[0] + dx, key[1] + dz
                candidate = np.array(
                    [start[0] + ix * profile.grid_m, current[1], start[2] + iz * profile.grid_m]
                )
                if not self._in_corridor(candidate, seed):
                    continue
                try:
                    p = as_point(self.navmesh.snap_point(candidate.astype(np.float32)))
                except ClearanceError:
                    continue
                if np.linalg.norm(p[[0, 2]] - candidate[[0, 2]]) > 1e-4:
                    continue
                if abs(p[1] - current[1]) > profile.max_height_step_m or not self._in_corridor(p, seed):
                    continue
                next_key = (ix, iz, round(float(p[1]) / profile.floor_bin_m))
                next_cost = cost + float(np.linalg.norm(p - current))
                if next_cost >= costs.get(next_key, math.inf):
                    continue
                if not self.edge_clear(current, p):
                    continue
                points[next_key], parent[next_key], costs[next_key] = p, key, next_cost
                heapq.heappush(
                    queue,
                    (next_cost + float(np.linalg.norm(goal - p)), next(serial), next_cost, next_key),
                )
        raise ClearanceError("no route with requested clearance inside the reference corridor")

    def _smooth(self, path, seed):
        # Visibility shortcuts cannot cross a blocked corner or leave the corridor.
        reduced, i = [path[0]], 0
        while i < len(path) - 1:
            checkpoint(phase="smoothing")
            chosen = i + 1
            for j in range(len(path) - 1, i, -1):
                checkpoint()
                if self._corridor_all(
                    _resample([path[i], path[j]], self.profile.grid_m), seed
                ):
                    if self.edge_clear(path[i], path[j]):
                        chosen = j
                        break
            reduced.append(path[chosen])
            i = chosen
        points = np.asarray(_resample(reduced, self.profile.resample_m))
        if self.native_path:

            return native_fair(self, points, seed)
        for _ in range(self.profile.smoothing_passes):
            for i in range(1, len(points) - 1):
                candidate = 0.5 * points[i] + 0.25 * (points[i - 1] + points[i + 1])
                try:
                    p = as_point(self.navmesh.snap_point(candidate.astype(np.float32)))
                except ClearanceError:
                    continue
                if np.linalg.norm(p[[0, 2]] - candidate[[0, 2]]) > 1e-4 or not self._in_corridor(
                    p, seed
                ):
                    continue
                if abs(p[1] - candidate[1]) > self.profile.max_height_step_m:
                    continue
                if self.edge_clear(points[i - 1], p) and self.edge_clear(p, points[i + 1]):
                    points[i] = p
        return list(points)

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
            corridor = CorridorGeometry(seed, capacity=self.edge_memo.capacity) if self.optimize else seed
            if safe:
                initial, expanded = list(seed), 0
            elif self.search_backend == "native":

                initial, expanded = native_search(self, a, b, seed)
            else:
                initial, expanded = self._search(a, b, corridor)
            path = self._smooth(initial, corridor)
            if not all(self.edge_clear(p, q) for p, q in zip(path, path[1:])):
                raise ClearanceError("smoothed path failed swept-segment verification")
            route.extend(path[1:])
            mappings.append(len(route) - 1)
            reports.append({"segment": index, "seed_satisfied_clearance": safe, "expanded": expanded})
        samples = _resample(route, self.profile.edge_sample_m)
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
