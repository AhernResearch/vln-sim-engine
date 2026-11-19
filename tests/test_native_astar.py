"""Native search parity, safety callbacks, cancellation and frozen implementation."""

from dataclasses import replace
from threading import Event

import numpy as np
import pytest
from helpers import settings
from reference_planner import ReferencePlanner

from sim_engine.collection import service as collection
from sim_engine.collection.service import CollectionService
from sim_engine.config import CURRENT_PLANNING_PROFILE, CollectionSettings
from sim_engine.core.io import implementation_sources
from sim_engine.navigation import native as native_module
from sim_engine.navigation.budget import PlanningContext, PlanningStopped
from sim_engine.navigation.clearance import ClearancePlanner, ClearanceProfile
from sim_engine.navigation.errors import ClearanceError
from sim_engine.navigation.native import native_search
from sim_engine.planning import service
from sim_engine.planning.contracts import PlanningBudget, PlanningProfile
from sim_engine.planning.service import PlanningService


class Mesh:
    def snap_point(self, p):
        if .4 <= p[0] <= .6 and -.1 <= p[2] <= .1:
            return np.full(3, np.nan)
        return np.asarray(p, dtype=np.float32)

    def distance_to_closest_obstacle(self, p, radius):
        return min(radius, float(np.hypot(max(.4-p[0], 0, p[0]-.6), max(-.1-p[2], 0, p[2]-.1))))

    def try_step_no_sliding(self, a, b):
        return a if any(not np.isfinite(self.snap_point(a+(b-a)*t)).all()
                        for t in np.linspace(0, 1, 21)) else b


def make_planner(kind=ClearancePlanner):
    return kind(Mesh(), lambda a, b: [a, b], ClearanceProfile(clearance_m=.05))


@pytest.fixture
def native():
    return pytest.importorskip("sim_engine.navigation._astar")


@pytest.mark.parametrize("height", [0., .025, -.025, .075, -.075])
def test_native_matches_python_path_expansions_queries_and_order(height, native):
    a, b = np.asarray([0., height, 0]), np.asarray([1., height, 0])
    seed = np.asarray([a, b])
    results = []
    for kind in [ReferencePlanner, ClearancePlanner]:
        p = make_planner(kind)
        context = PlanningContext(PlanningBudget(), Event())
        with context.bind():
            path, count = p._search(a, b, seed, p._corridor(seed))
        results.append((np.asarray(path).tolist(), count, context.queries, context.expansions))
    assert results[0] == results[1]


@pytest.mark.parametrize("anchors", [
    [[0, 0, 0], [1, 0, 0]],
    [[0, 0, 0], [.2, 0, .5], [.8, 0, .5], [1, 0, 0]],
    [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
])
def test_native_full_preparation_preserves_ordered_and_repeated_anchors(anchors, native):
    assert make_planner().prepare(anchors) == make_planner(ReferencePlanner).prepare(anchors)


@pytest.mark.parametrize("budget", [PlanningBudget(max_queries=3), PlanningBudget(max_expansions=1)])
def test_native_obeys_shared_budget_and_propagates_cancellation(budget, native):
    p = make_planner()
    stop = Event()
    context = PlanningContext(budget, stop)
    with context.bind(), pytest.raises(PlanningStopped) as error:
        native_search(p, [0, 0, 0], [1, 0, 0], [[0, 0, 0], [1, 0, 0]])
    assert error.value.outcome == "budget_exhausted"
    stop.set()
    with PlanningContext(PlanningBudget(), stop).bind(), pytest.raises(PlanningStopped) as error:
        native_search(p, [0, 0, 0], [1, 0, 0], [[0, 0, 0], [1, 0, 0]])
    assert error.value.outcome == "cancelled"


def test_native_per_search_limit_and_callback_errors_do_not_retry(native, monkeypatch):
    p = make_planner()
    p.profile = replace(p.profile, max_expansions=1)
    with pytest.raises(ClearanceError, match="expansion budget exhausted"):
        native_search(p, [0, 0, 0], [1, 0, 0], [[0, 0, 0], [1, 0, 0]])
    calls = []
    p = make_planner()  # A fresh task cache: the first search already cached its snaps.

    def fail(point):
        calls.append(point)
        raise RuntimeError("lost NavMesh")

    monkeypatch.setattr(p.navmesh, "snap_point", fail)
    with pytest.raises(RuntimeError, match="lost NavMesh"):
        native_search(p, [0, 0, 0], [1, 0, 0], [[0, 0, 0], [1, 0, 0]])
    assert len(calls) == 1


def test_mainline_defaults_match_and_missing_extension_fails_before_search(monkeypatch, tmp_path):
    assert PlanningProfile().version == CURRENT_PLANNING_PROFILE == "collection-v5"
    assert CollectionSettings(root=tmp_path, training_datasets=["train"]).planning_profile_version == (
        CURRENT_PLANNING_PROFILE)
    with pytest.raises(ValueError):
        PlanningProfile(version="collection-v4")

    def unavailable(name):
        raise ImportError(name)

    monkeypatch.setattr(native_module.importlib, "import_module", unavailable)
    with pytest.raises(RuntimeError, match="requires the compiled"):
        make_planner()


def test_collection_startup_rejects_missing_native_before_creating_attempts(tmp_path, monkeypatch):

    def unavailable():
        raise RuntimeError("native unavailable")

    for module in (collection, service):
        monkeypatch.setattr(module, "require_native", unavailable)
    config = settings(tmp_path)
    config.collection = CollectionSettings(root=tmp_path/"records", training_datasets=["train"])
    with pytest.raises(RuntimeError, match="native unavailable"):
        CollectionService(config, None)
    with pytest.raises(RuntimeError, match="native unavailable"):
        PlanningService(None)
    assert not config.collection.root.exists()


def test_native_binary_and_source_participate_in_implementation_identity(tmp_path):
    for name, content in {"a.py": b"python", "_astar.cpp": b"cpp", "_geometry.h": b"header", "_astar.abi3.so": b"binary",
                          "cache.pyc": b"ignored"}.items():
        (tmp_path/name).write_bytes(content)
    before = implementation_sources(tmp_path)
    assert set(before) == {"a.py", "_astar.cpp", "_geometry.h", "_astar.abi3.so"}
    (tmp_path/"_astar.abi3.so").write_bytes(b"different compiler output")
    assert implementation_sources(tmp_path) != before
