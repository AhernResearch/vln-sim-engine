"""Immutable reference geometry, exact point memoization and bounded matrix queries."""

import numpy as np

from sim_engine.core.cache import MISSING, BoundedMemo
from sim_engine.navigation.budget import checkpoint


class CorridorGeometry:
    def __init__(self, seed, *, capacity=65536, max_pairs=32768):
        seed = np.array(seed, dtype=float, copy=True)
        if seed.ndim != 2 or seed.shape[1] != 3 or len(seed) < 2 or not np.isfinite(seed).all():
            raise ValueError("finite reference segments required")
        if type(max_pairs) is not int or max_pairs < 1:
            raise ValueError("matrix pair budget must be positive")
        self.a = seed[:-1].copy()
        self.delta = seed[1:] - self.a
        self.denominator = np.maximum(np.sum(self.delta * self.delta, axis=1), 1e-12)
        for values in (self.a, self.delta, self.denominator):
            values.flags.writeable = False
        self.memo = BoundedMemo(capacity)
        self.max_pairs = max_pairs

    def _scalar(self, p, radius):
        t = np.clip(np.sum((p - self.a) * self.delta, axis=1) / self.denominator, 0, 1)
        return float(np.linalg.norm(p - (self.a + t[:, None] * self.delta), axis=1).min()) <= radius

    def contains(self, point, radius):
        p = np.asarray(point, dtype=float)
        key = (radius, p.tobytes())
        cached = self.memo.get(key)
        if cached is not MISSING:
            return cached
        result = self._scalar(p, radius)
        self.memo.put(key, result)
        return result

    def contains_many(self, points, radius):
        """Compute point-by-segment distances without an unbounded N×M allocation.

        Budget charging belongs to the caller: short-circuit predicates must only
        charge the logical prefix they actually consume. This method still polls
        cancellation/deadline between bounded matrix blocks.
        """
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return np.empty(0, dtype=bool)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("finite XYZ query points required")
        result = np.empty(len(points), dtype=bool)
        missing = []
        for index, p in enumerate(points):
            checkpoint()
            key = (radius, p.tobytes())
            cached = self.memo.get(key)
            if cached is MISSING:
                missing.append((index, key))
            else:
                result[index] = cached
        segment_chunk = min(len(self.a), self.max_pairs)
        point_chunk = max(1, self.max_pairs // segment_chunk)
        for begin in range(0, len(missing), point_chunk):
            checkpoint()
            block = missing[begin : begin + point_chunk]
            p = points[[index for index, _ in block]]
            distances = np.full(len(block), np.inf)
            for start in range(0, len(self.a), segment_chunk):
                checkpoint()
                a = self.a[start : start + segment_chunk]
                delta = self.delta[start : start + segment_chunk]
                den = self.denominator[start : start + segment_chunk]
                offset = p[:, None, :] - a[None, :, :]
                t = np.clip(np.sum(offset * delta[None, :, :], axis=2) / den[None, :], 0, 1)
                gap = p[:, None, :] - (a[None, :, :] + t[:, :, None] * delta[None, :, :])
                distances = np.minimum(distances, np.linalg.norm(gap, axis=2).min(axis=1))
            for (index, key), distance in zip(block, distances):
                value = bool(distance <= radius)
                result[index] = value
                self.memo.put(key, value)
        return result
