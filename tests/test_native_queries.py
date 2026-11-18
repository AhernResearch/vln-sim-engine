"""Task identity, exact cache semantics and native path verification parity."""

import math
from threading import Event

import numpy as np
import pytest
from test_native_astar import Mesh as ObstacleMesh

from sim_engine.navigation.budget import PlanningContext, PlanningStopped
from sim_engine.navigation.clearance import ClearancePlanner, ClearanceProfile
from sim_engine.navigation.native import native_search, require_native
from sim_engine.navigation.queries import NavMeshQueries
from sim_engine.planning.contracts import PlanningBudget, PlanningProfile


@pytest.fixture(autouse=True)
def native_kernel():
    require_native()


class Mesh:
    def __init__(self):
        self.snaps = self.clearances = self.islands = self.steps = 0
        self.invalid = self.fail = self.block = False
        self.distance = .2

    def snap_point(self, point):
        self.snaps += 1
        if self.fail:
            raise RuntimeError("NavMesh lost")
        return np.full(3, np.nan) if self.invalid else np.asarray(point, dtype=np.float32)

    def distance_to_closest_obstacle(self, point, radius):
        self.clearances += 1
        return self.distance

    def get_island(self, point):
        self.islands += 1
        return -1 if self.invalid else 0

    def try_step_no_sliding(self, a, b):
        self.steps += 1
        return a if self.block and b[0] > a[0] else b


def planner(mesh, native):
    return ClearancePlanner(
        mesh, lambda a, b: [a, b], ClearanceProfile(clearance_m=.05),
        search_backend="native", query_cache=NavMeshQueries(mesh) if native else None,
        native_path=native,
    )


def test_batch_order_float32_identity_copy_and_signed_zero():
    mesh = Mesh()
    q = NavMeshQueries(mesh, batch_size=1)
    points = [[1., 0, 0], [np.nextafter(1., 2.), 0, 0], [2., 0, 0], [1., 0, 0]]
    np.testing.assert_array_equal(q.snap_many(points), np.asarray(points, dtype=np.float32))
    assert mesh.snaps == 2 and q.stats()["snap"]["hits"] == 2
    changed = q.snap_many([[1, 0, 0]])
    changed[:] = 10
    assert q.snap_many([[1, 0, 0]])[0, 0] == 1
    q.clear()
    values = q.snap_many([[0., 0, 0], [-0., 0, 0]])
    assert q.stats()["snap"]["entries"] == 2
    assert np.signbit(values[:, 0]).tolist() == [False, True]


def test_shared_radius_identity_zero_island_and_no_cross_mesh_certificates():
    mesh = Mesh()
    q = NavMeshQueries(mesh)
    np.testing.assert_array_equal(q.clearance_many([[0, 0, 0]]*3), [.2]*3)
    assert mesh.clearances == 1
    assert q.clearance([0, 0, 0], 3.) == .2 and mesh.clearances == 2
    np.testing.assert_array_equal(q.island_many([[0, 0, 0]]*3), [0, 0, 0])
    assert mesh.islands == 1
    with pytest.raises(ValueError, match="different NavMesh"):
        ClearancePlanner(Mesh(), lambda a, b: [a, b], query_cache=q)


def test_geometric_failure_is_cached_but_runtime_exception_is_not():
    mesh = Mesh()
    q = NavMeshQueries(mesh)
    mesh.invalid = True
    assert np.isnan(q.snap_many([[0, 0, 0]]*2)).all() and mesh.snaps == 1
    np.testing.assert_array_equal(q.island_many([[0, 0, 0]]*2), [-1, -1])
    assert mesh.islands == 1
    q.clear()
    mesh.fail = True
    for _ in range(2):
        with pytest.raises(RuntimeError, match="NavMesh lost"):
            q.snap_many([[0, 0, 0]])
    assert q.stats()["snap"]["entries"] == 0 and mesh.snaps == 3
    mesh.fail = mesh.invalid = False
    assert np.isfinite(q.snap_many([[0, 0, 0]])).all()


def test_cache_lru_and_task_clear_release_all_query_kinds():
    mesh = Mesh()
    q = NavMeshQueries(mesh, capacity=2)
    q.snap_many([[0, 0, 0], [1, 0, 0], [0, 0, 0], [2, 0, 0]])
    assert q.stats()["snap"]["evictions"] == 1
    before = mesh.snaps
    q.snap_many([[0, 0, 0]])
    assert mesh.snaps == before
    q.snap_many([[1, 0, 0]])
    assert mesh.snaps == before+1
    q.clearance_many([[0, 0, 0]])
    q.island_many([[0, 0, 0]])
    q.clear()
    assert all(v["entries"] == v["hits"] == v["misses"] == 0 for v in q.stats().values())


def test_navmesh_context_releases_shared_arena_between_tasks():
    from types import SimpleNamespace

    from sim_engine.planning.geometry import NavMeshContext

    mesh = Mesh()
    q = NavMeshQueries(mesh)
    q.snap_many([[0, 0, 0]])
    context = NavMeshContext.__new__(NavMeshContext)
    context.queries = q
    context.planners = [ClearancePlanner(mesh, lambda a, b: [a, b], query_cache=q)]
    context.guard = ClearancePlanner(mesh, lambda a, b: [a, b], query_cache=q)
    context.references = SimpleNamespace(values={"old": True})
    context.tasks = {"old": True}
    context.clear_task()
    assert not context.tasks and not context.references.values
    assert all(v["entries"] == 0 for v in q.stats().values())


def test_batch_cached_queries_obey_prefix_budget_and_cancellation():
    mesh = Mesh()
    q = NavMeshQueries(mesh)
    q.clearance([0, 0, 0])
    stop = Event()
    context = PlanningContext(PlanningBudget(max_queries=3), stop)
    with context.bind(), pytest.raises(PlanningStopped) as error:
        q.clearance_many([[0, 0, 0]]*8)
    assert error.value.outcome == "budget_exhausted" and context.queries == 4
    assert mesh.clearances == 1
    stop.set()
    before = q.stats()["snap"]["misses"]
    with PlanningContext(PlanningBudget(), stop).bind(), pytest.raises(PlanningStopped):
        q.snap_many([[0, 0, 0]])
    assert q.stats()["snap"]["misses"] == before


def test_batch_mid_call_cancel_stops_remaining_queries():
    mesh = Mesh()
    stop = Event()
    original = mesh.snap_point

    def snap(point):
        value = original(point)
        if mesh.snaps == 2:
            stop.set()
        return value

    mesh.snap_point = snap
    q = NavMeshQueries(mesh, batch_size=3)
    with PlanningContext(PlanningBudget(), stop).bind(), pytest.raises(PlanningStopped) as error:
        q.snap_many([[i, 0, 0] for i in range(20)])
    assert error.value.outcome == "cancelled" and mesh.snaps == 2


@pytest.mark.parametrize("points", [[], np.empty((0, 3))])
def test_empty_batches(points):
    q = NavMeshQueries(Mesh())
    assert q.snap_many(points).shape == (0, 3)
    assert q.clearance_many(points).shape == q.island_many(points).shape == (0,)


@pytest.mark.parametrize("points", [[[math.nan, 0, 0]], [[1, 2]], [1, 2, 3], np.empty((0, 2))])
def test_invalid_batch_is_rejected_before_navmesh_calls(points):
    mesh = Mesh()
    with pytest.raises(ValueError):
        NavMeshQueries(mesh).snap_many(points)
    assert mesh.snaps == 0


@pytest.mark.parametrize("length", [0., .025, np.nextafter(.025, 0), np.nextafter(.025, 1), .1])
@pytest.mark.parametrize("block", [False, True])
def test_native_edge_sample_count_values_short_circuit_and_logical_budget(length, block):
    results = []
    for native in [False, True]:
        mesh = Mesh()
        mesh.block = block
        p = planner(mesh, native)
        context = PlanningContext(PlanningBudget(), Event())
        with context.bind():
            result = p._edge_uncached(np.zeros(3), np.asarray([length, 0, 0]))
        results.append((result, context.queries, mesh.steps))
    assert results[0] == results[1]


@pytest.mark.parametrize("offset", [np.nextafter(1e-4, 0), 1e-4, np.nextafter(1e-4, 1)])
def test_native_horizontal_boundary_uses_original_numpy_decision(offset):
    results = []
    for native in [False, True]:
        mesh = Mesh()
        mesh.snap_point = lambda p: np.asarray(p, dtype=float)+[offset, 0, 0]
        p = planner(mesh, native)
        p.edge_clear([0, 0, 0], [.1, 0, 0])
        results.append(p._edge_uncached(np.zeros(3), np.asarray([.1, 0, 0])))
    assert results[0] == results[1]


@pytest.mark.parametrize("anchors", [
    [[0, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [.2, 0, .5], [.8, 0, .5], [1, 0, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
])
def test_native_full_preparation_queries_and_sequential_fairing_match_v4(anchors):
    results = []
    for optimized in [False, True]:
        p = planner(ObstacleMesh(), optimized)
        context = PlanningContext(PlanningBudget(), Event())
        with context.bind():
            result = p.prepare(anchors)
        results.append((result, context.queries, context.expansions))
    assert results[0] == results[1]


def test_native_path_runtime_error_propagates_once_and_budget_stops_cached_edge():
    mesh = Mesh()
    p = planner(mesh, True)
    mesh.fail = True
    with pytest.raises(RuntimeError, match="NavMesh lost"):
        p.edge_clear([0, 0, 0], [.1, 0, 0])
    assert mesh.snaps == 1 and p.edge_memo.stats()["entries"] == 0
    mesh.fail = False
    p.edge_clear([0, 0, 0], [.1, 0, 0])
    with PlanningContext(PlanningBudget(max_queries=1), Event()).bind(), pytest.raises(PlanningStopped):
        p.edge_clear([0, 0, 0], [.1, 0, 0])


def test_searches_share_exact_snap_queries_without_sharing_corridor_or_budget():
    mesh = ObstacleMesh()
    p = planner(mesh, True)
    seed = np.asarray([[0., 0, 0], [1., 0, 0]])
    context = PlanningContext(PlanningBudget(), Event())
    with context.bind():
        first = native_search(p, seed[0], seed[-1], seed)
        first_queries, first_expansions = context.queries, context.expansions
        count = p.query_cache.stats()["snap"]["misses"]
        second = native_search(p, seed[0], seed[-1], seed)
    assert np.asarray(first[0]).tolist() == np.asarray(second[0]).tolist()
    assert first[1] == second[1] and context.queries == 2*first_queries
    assert context.expansions == 2*first_expansions
    assert p.query_cache.stats()["snap"]["misses"] == count


def test_v5_keeps_legacy_v4_flags_and_explicit_new_processing():
    task = {"seed": 1, "kind": "short_instructions"}
    old = PlanningProfile(version="collection-v4").geometry(task)
    new = PlanningProfile(version="collection-v5").geometry(task)
    assert "query_optimizations" not in old
    assert new.pop("query_optimizations") and new.pop("native_path_optimizations")
    assert old == new


def test_stale_compiled_kernel_is_rejected(monkeypatch):
    from types import SimpleNamespace

    from sim_engine.navigation import native

    monkeypatch.setattr(native.importlib, "import_module", lambda name: SimpleNamespace(KERNEL_API=1))
    with pytest.raises(RuntimeError, match="kernel API 2"):
        native.require_native()
