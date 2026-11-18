"""Geometry equivalence, conservative ROI scope, cache identity and budget safety."""

from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from sim_engine.core.cache import MISSING, BoundedMemo
from sim_engine.navigation.budget import PlanningContext, PlanningStopped
from sim_engine.navigation.clearance import ClearancePlanner, ClearanceProfile
from sim_engine.navigation.corridor import CorridorGeometry
from sim_engine.navigation.safe_image import safe_floor_mask
from sim_engine.navigation.visual_region import sample_region
from sim_engine.planning.contracts import PlanningBudget, PlanningProfile


class Mesh:
    def __init__(self):
        self.snaps = self.steps = 0
        self.block_forward = False

    def snap_point(self, p):
        self.snaps += 1
        return np.asarray(p, dtype=np.float32)

    def distance_to_closest_obstacle(self, p, radius):
        return 0.10

    def try_step_no_sliding(self, a, b):
        self.steps += 1
        return a if self.block_forward and b[0] > a[0] else b

    def is_navigable(self, p):
        return True

    def get_island(self, p):
        return 0


def planner(mesh, optimize=True):
    return ClearancePlanner(
        mesh, lambda a, b: [a, b], ClearanceProfile(clearance_m=0.05), optimize=optimize
    )


def test_directed_edge_cache_preserves_false_and_profile_boundaries():
    mesh = Mesh()
    mesh.block_forward = True
    p = planner(mesh)
    a, b = [0, 0, 0], [0.1, 0, 0]
    assert p.edge_clear(a, b) is False
    calls = mesh.snaps
    assert p.edge_clear(a, b) is False and mesh.snaps == calls
    assert p.edge_clear(b, a) is True
    assert mesh.snaps > calls
    p.profile = replace(p.profile, clearance_m=0.15)
    assert p.edge_clear(b, a) is False  # A lower-clearance certificate cannot satisfy this profile.


def test_cache_preserves_logical_queries_budget_and_cancellation():
    p = planner(Mesh())
    stop = Event()
    context = PlanningContext(PlanningBudget(max_queries=100), stop)
    with context.bind():
        assert p.edge_clear([0, 0, 0], [0.1, 0, 0])
        first = context.queries
        assert first > 1
        assert p.edge_clear([0, 0, 0], [0.1, 0, 0])
        assert context.queries == 2 * first
        context.budget.max_queries = context.queries + first - 1
        with pytest.raises(PlanningStopped) as error:
            p.edge_clear([0, 0, 0], [0.1, 0, 0])
        assert error.value.outcome == "budget_exhausted"
        stop.set()
        with pytest.raises(PlanningStopped) as error:
            p.edge_clear([0, 0, 0], [0.1, 0, 0])
        assert error.value.outcome == "cancelled"


def test_query_clear_releases_task_certificates():
    mesh = Mesh()
    p = planner(mesh)
    assert p.edge_clear([0, 0, 0], [0.1, 0, 0])
    assert p.edge_memo.stats()["entries"] == 1
    p.clear_queries()
    assert not p._clearance_cache and not p.edge_memo.values
    mesh.block_forward = True
    assert p.edge_clear([0, 0, 0], [0.1, 0, 0]) is False


def test_bounded_memo_eviction_and_false_are_unambiguous():
    memo = BoundedMemo(1)
    memo.put("a", False)
    assert memo.get("a") is False
    memo.put("b", True)
    assert memo.get("a") is MISSING and memo.get("b") is True
    assert memo.stats()["evictions"] == 1
    disabled = BoundedMemo(0)
    disabled.put("a", True)
    assert disabled.get("a") is MISSING


@pytest.mark.parametrize("max_pairs", [1, 7, 32768])
def test_matrix_corridor_matches_scalar_at_boundaries_and_degenerate_segments(max_pairs):
    rng = np.random.default_rng(9401)
    seed = rng.normal(size=(19, 3))
    seed[5] = seed[4]
    seed[8] = seed[7] + 1e-9
    points = np.concatenate([rng.normal(size=(300, 3)), seed])
    # Include points exactly on, one ULP below and one ULP above a simple corridor boundary.
    straight = np.asarray([[0.0, 0, 0], [2.0, 0, 0]])
    edge = np.asarray([[1.0, y, 0] for y in [0.75, np.nextafter(0.75, 0), np.nextafter(0.75, 1)]])
    reference = planner(Mesh(), optimize=False)
    for path, samples in [(seed, points), (straight, edge)]:
        corridor = CorridorGeometry(path, max_pairs=max_pairs, capacity=16)
        expected = [reference._in_corridor(p, path) for p in samples]
        np.testing.assert_array_equal(corridor.contains_many(samples, 0.75), expected)
        assert [corridor.contains(p, 0.75) for p in samples] == expected
        assert corridor.memo.stats()["entries"] <= 16
        assert corridor.contains_many(np.empty((0, 3)), 0.75).shape == (0,)


def test_corridor_owns_immutable_geometry_and_radius_is_in_cache_identity():
    seed = np.asarray([[0.0, 0, 0], [1.0, 0, 0]])
    geometry = CorridorGeometry(seed)
    assert geometry.contains([0.5, 0.2, 0], 0.3)
    assert not geometry.contains([0.5, 0.2, 0], 0.1)
    seed[:] = 100
    assert geometry.contains([0.5, 0.2, 0], 0.3)


def test_matrix_short_circuit_charges_only_consumed_prefix():
    p = planner(Mesh())
    geometry = CorridorGeometry([[0, 0, 0], [1, 0, 0]])
    context = PlanningContext(PlanningBudget(max_queries=2), Event())
    with context.bind():
        assert not p._corridor_all([[0, 0, 0], [0, 2, 0], [0, 0, 0]], geometry)
        assert context.queries == 2


def floor_scene():
    depth = np.zeros((101, 101), dtype=np.float32)
    for y in range(51, 101):
        depth[y, :] = 50 / (y - 50)
    return (
        depth,
        [[50, 0, 50], [0, 50, 50], [0, 0, 1]],
        {"position": [0, 1, 0], "rotation_xyzw": [0, 0, 0, 1]},
    )


@pytest.mark.parametrize(
    "bbox", [[400, 700, 700, 900], [0, 700, 50, 1000], [499.9, 749.9, 500.1, 750.1]]
)
def test_roi_retains_exact_in_box_mask_world_mapping_and_candidate_order(bbox):
    depth, intrinsics, sensor = floor_scene()
    mesh = Mesh()
    guard = SimpleNamespace(clearance=lambda p: 0.1, profile=SimpleNamespace(clearance_m=0.05))
    args = (depth, intrinsics, sensor, [0, 0, 0], mesh, guard)
    full = safe_floor_mask(*args)
    calls = mesh.snaps
    roi = safe_floor_mask(*args, bbox=bbox)
    roi_calls = mesh.snaps - calls
    assert roi_calls < calls
    yy, xx = np.indices(depth.shape)
    x0, y0, x1, y1 = bbox
    inside = (
        (xx / 100 * 1000 >= x0)
        & (xx / 100 * 1000 <= x1)
        & (yy / 100 * 1000 >= y0)
        & (yy / 100 * 1000 <= y1)
    )
    for a, b in zip(full[:3], roi[:3]):
        np.testing.assert_array_equal(a[inside], b[inside])
    common = dict(
        bbox=bbox,
        depth=depth,
        intrinsics=intrinsics,
        sensor=sensor,
        start=[0, 0, 0],
        mesh=mesh,
        guard=guard,
        shortest=lambda a, b: [a, b],
    )
    assert sample_region(**common, safe_mask=full[0]) == sample_region(**common, safe_mask=roi[0])
    assert roi[3]["outside_computed_bounds"] == "unknown_not_obstacle"
    left, top, right, bottom = roi[3]["computed_pixel_bounds_xyxy"]
    assert 0 <= left < right <= 101 and 0 <= top < bottom <= 101


def test_profiles_pin_optimization_and_legacy_input_remains_available():
    task = {"seed": 1, "kind": "short_instructions"}
    assert "planning_optimizations" not in PlanningProfile(version="collection-v2").geometry(task)
    assert PlanningProfile().geometry(task)["planning_optimizations"] is True


@pytest.mark.parametrize(
    "anchors", [[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0.2, 0, 0.5], [0.8, 0, 0.5], [1, 0, 0]]]
)
def test_search_smoothing_and_ordered_anchors_match_legacy_around_obstacle(anchors):
    class BlockMesh(Mesh):
        @staticmethod
        def blocked(p):
            return 0.4 <= p[0] <= 0.6 and -0.1 <= p[2] <= 0.1

        def snap_point(self, p):
            self.snaps += 1
            return np.full(3, np.nan) if self.blocked(p) else np.asarray(p, dtype=np.float32)

        def distance_to_closest_obstacle(self, p, radius):
            dx = max(0.4 - p[0], 0, p[0] - 0.6)
            dz = max(-0.1 - p[2], 0, p[2] - 0.1)
            return min(radius, float(np.hypot(dx, dz)))

        def try_step_no_sliding(self, a, b):
            return a if any(self.blocked(a + (b - a) * t) for t in np.linspace(0, 1, 21)) else b

    old = planner(BlockMesh(), optimize=False).prepare(anchors)
    new = planner(BlockMesh(), optimize=True).prepare(anchors)
    assert old == new
    assert new["world_points"][0] == anchors[0]
    assert new["world_points"][-1] == anchors[-1]
    assert len(new["anchor_indices"]) == len(anchors)
