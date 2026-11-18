"""Task-owned NavMesh identity, exact native memoization and ordered batch APIs."""

import math

import numpy as np

from sim_engine.navigation.errors import ClearanceError

from .budget import checkpoint
from .native import require_native


def check(queries=0, expansions=0):
    checkpoint(query=queries, expansion=expansions)


class NavMeshQueries:
    """One immutable NavMesh, shared across task planners; never global or cross-task.

    Snap/island operations poll cancellation without charging extra logical queries.
    Clearance consumes one query per point, including cache hits. Public batch
    methods preserve point order and stop at the first exception or cancellation.
    """

    def __init__(self, navmesh, *, capacity=65536, batch_size=256):
        if type(capacity) is not int or not 0 <= capacity <= 1048576:
            raise ValueError("invalid geometry cache capacity")
        if type(batch_size) is not int or not 1 <= batch_size <= 4096:
            raise ValueError("invalid geometry batch size")
        self.navmesh, self.batch_size = navmesh, batch_size
        self.kernel = require_native()
        self.arena = self.kernel.query_cache(capacity)

    def _snap(self, point):

        try:
            return self.navmesh.snap_point(np.asarray(point, dtype=np.float32))
        except ClearanceError:
            return (math.nan, math.nan, math.nan)

    def _clearance(self, point, radius):
        return self.navmesh.distance_to_closest_obstacle(np.asarray(point, dtype=np.float32), radius)

    def _island(self, point):
        return self.navmesh.get_island(np.asarray(point, dtype=np.float32))

    def _step(self, a, b):

        try:
            return self.navmesh.try_step_no_sliding(np.asarray(a, dtype=np.float32),
                                                   np.asarray(b, dtype=np.float32))
        except ClearanceError:
            return (math.nan, math.nan, math.nan)

    def _batch(self, points, operation, *args):
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            if points.shape not in {(0,), (0, 3)}:
                raise ValueError("finite XYZ query points required")
            return []
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("finite XYZ query points required")
        rows = []
        for start in range(0, len(points), self.batch_size):
            checkpoint()
            rows.extend(operation(self.arena, points[start:start+self.batch_size], *args))
        return rows

    def snap_many(self, points):
        return np.asarray(self._batch(points, self.kernel.snap_many, self._snap, check),
                          dtype=float).reshape(-1, 3)

    def clearance_many(self, points, radius=2.0):
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("invalid clearance radius")
        return np.asarray(self._batch(points, self.kernel.clearance_many, radius,
                                      self._clearance, check), dtype=float)

    def island_many(self, points):
        return np.asarray(self._batch(points, self.kernel.island_many, self._island, check),
                          dtype=np.int64)

    def clearance(self, point, radius=2.0):
        return self.kernel.clearance_one(self.arena, point, radius, self._clearance, check)

    def clear(self):
        self.kernel.query_clear(self.arena)

    def stats(self):
        return self.kernel.query_stats(self.arena)
