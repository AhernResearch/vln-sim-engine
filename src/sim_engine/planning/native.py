"""Explicit native backend selection; never retry a failed search in Python."""

import importlib
import math

import numpy as np

from .context import checkpoint


def require_native():
    try:
        kernel = importlib.import_module("sim_engine.planning._astar")
    except ImportError as error:
        raise RuntimeError("native planning requires the compiled sim-engine geometry extension") from error
    if getattr(kernel, "KERNEL_API", None) != 2:
        raise RuntimeError("native planning requires the compiled geometry extension with kernel API 2")
    return kernel


def native_search(planner, start, goal, seed):
    from sim_engine.collection.clearance import ClearanceError, _point

    kernel = require_native()
    profile = planner.profile

    def snap(point):
        return planner.navmesh.snap_point(np.asarray(point, dtype=np.float32))

    def check(queries, expansions):
        checkpoint(query=queries, expansion=expansions, phase="astar" if expansions else None)

    try:
        optional = () if planner.query_cache is None else (planner.query_cache.arena,)
        path, expanded = kernel.search(
            start, goal, seed, profile.grid_m, profile.max_height_step_m,
            profile.floor_bin_m, profile.corridor_m, profile.max_expansions,
            snap if planner.query_cache is None else planner.query_cache._snap,
            planner.edge_clear, check, *optional,
        )
    except ValueError as error:
        raise ClearanceError(str(error)) from error
    return [_point(p) for p in path], expanded


def _norm_beyond(delta, threshold):
    return bool(np.linalg.norm(np.asarray(delta, dtype=float)) > threshold)


def native_swept_edge(planner, a, b):
    from sim_engine.collection.clearance import ClearanceError

    q, p = planner.query_cache, planner.profile
    # NumPy computes the old sample count, including integer-distance boundaries.
    count = max(1, math.ceil(float(np.linalg.norm(b-a)) / p.edge_sample_m))
    try:
        return q.kernel.swept_edge(a, b, count, p.max_height_step_m, p.clearance_m,
                                  max(2.0, p.clearance_m*2), q.arena, q._snap, q._clearance,
                                  q._step, _path_check, _norm_beyond)
    except ValueError as error:
        raise ClearanceError(str(error)) from error


def _path_check(queries=0, expansions=0):
    checkpoint(query=queries, expansion=expansions)


def native_fair(planner, points, seed):
    from sim_engine.collection.clearance import ClearanceError, _point

    q, p = planner.query_cache, planner.profile
    try:
        result = q.kernel.fair_path(
            points, p.smoothing_passes, p.max_height_step_m, q.arena, q._snap,
            lambda point: planner._in_corridor(point, seed), planner.edge_clear,
            _path_check, _norm_beyond,
        )
    except ValueError as error:
        raise ClearanceError(str(error)) from error
    return [_point(point) for point in result]
