"""Adapters from ClearancePlanner to the compiled geometry kernel; failures never fall back."""

import importlib
import math

import numpy as np

from sim_engine.navigation.errors import ClearanceError

from .budget import checkpoint


def as_point(value):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ClearanceError("navmesh returned a nonfinite or invalid world point")
    return point


def require_native():
    try:
        kernel = importlib.import_module("sim_engine.navigation._astar")
    except ImportError as error:
        raise RuntimeError("native planning requires the compiled sim-engine geometry extension") from error
    if getattr(kernel, "KERNEL_API", None) != 2:
        raise RuntimeError("native planning requires the compiled geometry extension with kernel API 2")
    return kernel


def native_search(planner, start, goal, seed):
    q, profile = planner.queries, planner.profile

    def check(queries, expansions):
        checkpoint(query=queries, expansion=expansions, phase="astar" if expansions else None)

    try:
        path, expanded = q.kernel.search(
            start, goal, seed, profile.grid_m, profile.max_height_step_m,
            profile.floor_bin_m, profile.corridor_m, profile.max_expansions,
            q._snap, planner.edge_clear, check, q.arena,
        )
    except ValueError as error:
        raise ClearanceError(str(error)) from error
    return [as_point(p) for p in path], expanded


def _norm_beyond(delta, threshold):
    return bool(np.linalg.norm(np.asarray(delta, dtype=float)) > threshold)


def native_swept_edge(planner, a, b):
    q, p = planner.queries, planner.profile
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


def native_fair(planner, points, corridor):
    q, p = planner.queries, planner.profile
    try:
        result = q.kernel.fair_path(
            points, p.smoothing_passes, p.max_height_step_m, q.arena, q._snap,
            lambda point: planner._in_corridor(point, corridor), planner.edge_clear,
            _path_check, _norm_beyond,
        )
    except ValueError as error:
        raise ClearanceError(str(error)) from error
    return [as_point(point) for point in result]
