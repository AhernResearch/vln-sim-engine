"""Python reference for the native planning kernels; used only as a test oracle.

It reproduces the original Python search, swept-edge check, scalar corridor and
fairing loop. Production planning always runs ``ClearancePlanner``'s native path.
"""

import heapq
import itertools
import math

import numpy as np

from sim_engine.core.cache import BoundedMemo
from sim_engine.navigation.budget import checkpoint
from sim_engine.navigation.clearance import ClearancePlanner, ClearanceProfile, resample
from sim_engine.navigation.errors import ClearanceError
from sim_engine.navigation.native import as_point


class ReferencePlanner(ClearancePlanner):
    def __init__(self, navmesh, shortest_path, profile=None, *, cache_entries=65536):
        self.navmesh, self.shortest_path = navmesh, shortest_path
        self.profile = profile or ClearanceProfile()
        self.edge_memo = BoundedMemo(cache_entries)

    def clearance(self, p):
        checkpoint(query=1)
        distance = float(self.navmesh.distance_to_closest_obstacle(
            np.asarray(p, dtype=np.float32), max(2.0, self.profile.clearance_m * 2)))
        if not math.isfinite(distance) or distance < 0:
            raise ClearanceError("invalid navmesh clearance")
        return distance

    def _snap(self, point):
        return as_point(self.navmesh.snap_point(np.asarray(point, dtype=np.float32)))

    def _edge_uncached(self, a, b):
        queries, previous = 0, a
        for p in resample([a, b], self.profile.edge_sample_m):
            try:
                snapped = self._snap(p)
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
                reached = as_point(self.navmesh.try_step_no_sliding(
                    np.asarray(previous, dtype=np.float32), np.asarray(snapped, dtype=np.float32)))
            except ClearanceError:
                return False, queries
            if np.linalg.norm(reached - snapped) > 1e-3:
                return False, queries
            previous = snapped
        return True, queries

    def _corridor(self, seed):
        return np.asarray(seed, dtype=float)

    def _in_corridor(self, p, seed):
        checkpoint(query=1)
        a, b = seed[:-1], seed[1:]
        delta = b - a
        length2 = np.sum(delta * delta, axis=1)
        t = np.clip(np.sum((p - a) * delta, axis=1) / np.maximum(length2, 1e-12), 0, 1)
        return float(np.linalg.norm(p - (a + t[:, None] * delta), axis=1).min()) <= self.profile.corridor_m

    def _corridor_all(self, points, seed):
        return all(self._in_corridor(p, seed) for p in points)

    def _search(self, start, goal, seed, corridor):
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
                    [start[0] + ix * profile.grid_m, current[1], start[2] + iz * profile.grid_m])
                if not self._in_corridor(candidate, corridor):
                    continue
                try:
                    p = self._snap(candidate)
                except ClearanceError:
                    continue
                if np.linalg.norm(p[[0, 2]] - candidate[[0, 2]]) > 1e-4:
                    continue
                if abs(p[1] - current[1]) > profile.max_height_step_m or not self._in_corridor(p, corridor):
                    continue
                next_key = (ix, iz, round(float(p[1]) / profile.floor_bin_m))
                next_cost = cost + float(np.linalg.norm(p - current))
                if next_cost >= costs.get(next_key, math.inf) or not self.edge_clear(current, p):
                    continue
                points[next_key], parent[next_key], costs[next_key] = p, key, next_cost
                heapq.heappush(
                    queue, (next_cost + float(np.linalg.norm(goal - p)), next(serial), next_cost, next_key))
        raise ClearanceError("no route with requested clearance inside the reference corridor")

    def _fair(self, points, corridor):
        for _ in range(self.profile.smoothing_passes):
            for i in range(1, len(points) - 1):
                candidate = 0.5 * points[i] + 0.25 * (points[i - 1] + points[i + 1])
                try:
                    p = self._snap(candidate)
                except ClearanceError:
                    continue
                if (np.linalg.norm(p[[0, 2]] - candidate[[0, 2]]) > 1e-4
                        or not self._in_corridor(p, corridor)
                        or abs(p[1] - candidate[1]) > self.profile.max_height_step_m):
                    continue
                if self.edge_clear(points[i - 1], p) and self.edge_clear(p, points[i + 1]):
                    points[i] = p
        return list(points)
