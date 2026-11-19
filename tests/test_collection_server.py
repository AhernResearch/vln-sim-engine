"""Real spawned warm processes and HTTP; fake physics, no GPU/model/service control."""

import asyncio
import hashlib
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import settings
from PIL import Image
from planning_fakes import fake_audit, fake_local_client

from sim_engine.collection import service
from sim_engine.collection.server import create_app
from sim_engine.collection.service import CollectionService
from sim_engine.config import (
    CollectionScene,
    CollectionSceneBudget,
    CollectionSettings,
    Dataset,
    Physics,
    VelocityLimits,
)
from sim_engine.core.io import create_json, file_sha256, read_json
from sim_engine.planning.tasks import TaskRequest


@pytest.fixture(autouse=True)
def planning_backend(monkeypatch):
    monkeypatch.setattr(service, "LocalPlanningClient", fake_local_client)
    monkeypatch.setattr(service, "audit_recording", fake_audit)


class FakeActor:
    def __init__(self, config, gpu):
        self.loads = 0
        self.scene = None
        self.observed = 0

    def observe(self, request, folder, stop):
        if request["scene_id"] != self.scene:
            self.scene = request["scene_id"]
            self.loads += 1
        self.observed += 1
        if request["kind"] == "reference_geometry":
            return {
                "reference_legs": [{"distance_m": 2.0, "initial_yaw": 0.0}],
                "source": "navmesh_geometry_not_executed_states",
            }
        public = {"images": {}}
        if request["kind"] == "short_instructions":
            prepared = {"episode": {k: request[k] for k in
                        ["start_position", "start_rotation", "scene_id"]}, "yaw": 0}
            create_json(folder / "start-state.json", prepared)
        create_json(folder / "camera.json", {})
        views = {}
        for view in ("front", "right", "back", "left"):
            depth = folder / (view + "-depth.npz")
            np.savez_compressed(depth, depth_m=np.ones((32, 32), dtype=np.float32))
            views[view] = {"depth_sha256": file_sha256(depth)}
        create_json(folder / "view-sensors.json", views)
        for view in ("front", "right", "back", "left"):
            path = folder / (view + ".png")
            Image.new("RGB", (32, 32), (self.observed, 0, 0)).save(path)
            public["images"][view] = {"path": str(path), "sha256": file_sha256(path)}
        if request["kind"] == "r2r_chunks":
            public["instruction"] = request["instruction"]
        create_json(
            folder / "observed.json",
            {"loads": self.loads, "observed": self.observed, "pid": os.getpid()},
        )
        if request["instruction"] == "observation-error":
            raise ValueError("invalid start")
        return {"public": public, "cache": {"scene_loads": self.loads}}

    def execute(self, request, folder, target, stop):
        assert (folder / "observation-ready.json").exists()
        if request["kind"] == "short_instructions":
            assert target == read_json(folder / "target.json")
        create_json(folder / "movement-attempted.json", {"source": "fake_physics"})
        if request["instruction"] == "crash":
            os._exit(74)
        time.sleep(0.02)
        create_json(folder / "fake-execution.json", {"pid": os.getpid(), "loads": self.loads})
        return {"status": "completed_audit", "geometry": {"passed": True, "source": "fake_physics"}}

    def close(self):
        pass


def config(tmp_path, capacity=8, workers=1):
    cfg = settings(tmp_path)
    cfg.simulation.datasets = {
        "train_pilot": Dataset(
            episodes=tmp_path / "train.json", physics=Physics(velocity=VelocityLimits(time_step_s=0.25))
        )
    }
    cfg.collection = CollectionSettings(
        root=tmp_path / "records",
        training_datasets=["train_pilot"],
        worker_gpus=[0] * workers,
        capacity=capacity,
        minimum_free_disk_bytes=0,
        operation_timeout_s=3,
    )
    cfg.collection.planning_workers = 1
    # Registered NavMesh bytes are hashed by the real adapter; the fake engine never opens them.
    for scene in ["room", *"abcd"]:
        path = cfg.simulation.scenes_root / (scene + ".navmesh")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test navmesh")
    return cfg


def app(cfg):
    return create_app(
        cfg,
        store=SimpleNamespace(scene_hashes={"train_pilot": {"room.glb": "hash"}}),
        actor_factory=FakeActor,
    )


def task(name, kind="short_instructions", instruction="go"):
    result = {
        "request_id": hashlib.sha256(name.encode()).hexdigest(),
        "owner": "a" * 32,
        "kind": kind,
        "dataset": "train_pilot",
        "scene_id": "room.glb",
        "camera": "trained_ce30",
        "start_position": [0.0, 0.0, 0.0],
        "start_rotation": [0.0, 0.0, 0.0, 1.0],
    }
    if kind == "r2r_chunks":
        result.update(
            instruction=instruction,
            reference_guides=[[0.0, 0.0, 0.0], [0.0, 0.0, -2.0]],
            goal_position=[0.0, 0.0, -2.0],
        )
    return result


def wait(client, cid, expected):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        response = client.get("/v1/collections/" + cid)
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] in expected:
            return result
        time.sleep(0.01)
    pytest.fail(str(result))


TARGET = {"target_view": "front", "bbox": [400, 600, 600, 800]}


def test_inode_reserve_rejects_before_creating_a_task_claim(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cfg.collection.minimum_free_inodes = 100
    request = task("inode-reserve")
    with TestClient(app(cfg)) as client:
        current = service.os.statvfs(cfg.collection.root)
        values = {name: getattr(current, name) for name in dir(current) if name.startswith("f_")}
        values["f_favail"] = 99
        monkeypatch.setattr(service.os, "statvfs", lambda path: SimpleNamespace(**values))
        response = client.post("/v1/collections", json=request)
        assert response.status_code == 507 and "inode reserve" in response.text
        assert not (cfg.collection.root / "tasks" / request["request_id"]).exists()


def test_blocked_claim_keeps_health_responsive_and_duplicate_does_not_replay(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    request = task("blocked-claim")
    entered, release = Event(), Event()
    original = service.create_json
    writes = []

    def blocked(path, value):
        if path.name == "request.json":
            writes.append(path)
            entered.set()
            assert release.wait(5)
        original(path, value)

    with TestClient(app(cfg)) as client, ThreadPoolExecutor(max_workers=3) as calls:
        monkeypatch.setattr(service, "create_json", blocked)
        first = calls.submit(client.post, "/v1/collections", json=request)
        try:
            assert entered.wait(2)
            health = calls.submit(client.get, "/v1/health").result(timeout=1).json()
            assert health["evidence_io"]["pending"] >= 1
            state = client.get("/v1/collections/" + request["request_id"]).json()
            assert state["status"] == "claiming"
            assert not (cfg.collection.root / "tasks" / request["request_id"] / "lease.json").exists()
            duplicate = calls.submit(client.post, "/v1/collections", json=request)
            release.set()
            assert first.result(timeout=3).status_code == duplicate.result(timeout=3).status_code == 200
            wait(client, request["request_id"], {"observation_ready"})
            assert len(writes) == 1
            assert client.get("/v1/health").json()["worker_state"][0]["operations"]["observe"] == 1
            client.delete("/v1/collections/" + request["request_id"])
        finally:
            release.set()


def test_blocked_terminal_receipt_does_not_publish_success_before_durable_write(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    request = task("blocked-receipt")
    entered, release = Event(), Event()
    original = service.create_json

    def blocked(path, value):
        if path.name == "receipt.json":
            entered.set()
            assert release.wait(5)
        original(path, value)

    with TestClient(app(cfg)) as client, ThreadPoolExecutor(max_workers=1) as calls:
        monkeypatch.setattr(service, "create_json", blocked)
        client.post("/v1/collections", json=request)
        wait(client, request["request_id"], {"observation_ready"})
        client.post("/v1/collections/" + request["request_id"] + "/target", json=TARGET)
        try:
            assert entered.wait(3)
            assert calls.submit(client.get, "/v1/health").result(timeout=1).status_code == 200
            assert client.get("/v1/collections/" + request["request_id"]).json()["status"] == "auditing"
            release.set()
            wait(client, request["request_id"], {"completed_audit"})
            assert client.get("/v1/health").json()["worker_state"][0]["operations"]["execute"] == 1
        finally:
            release.set()


def test_success_quota_bounds_claims_then_rolls_early_without_replaying_failed_starts(tmp_path):
    cfg = config(tmp_path)
    cfg.collection.scene_schedule = [
        CollectionSceneBudget(
            dataset="train_pilot", scene_id=s, camera="trained_ce30", attempts=4, success_goal=goal
        )
        for s, goal in [("a.glb", 2), ("b.glb", 1)]
    ]
    application = create_app(
        cfg,
        store=SimpleNamespace(scene_hashes={"train_pilot": {"a.glb": "a", "b.glb": "b"}}),
        actor_factory=FakeActor,
    )
    with TestClient(application) as client:
        requests = [dict(task(f"goal-{i}"), scene_id="a.glb") for i in range(4)]
        for request in requests[:2]:
            assert client.post("/v1/collections", json=request).status_code == 200
            wait(client, request["request_id"], {"observation_ready"})
        assert client.post("/v1/collections", json=requests[2]).status_code == 409
        client.delete("/v1/collections/" + requests[0]["request_id"])
        assert client.post("/v1/collections", json=requests[2]).status_code == 200
        wait(client, requests[2]["request_id"], {"observation_ready"})
        for request in requests[1:3]:
            client.post("/v1/collections/" + request["request_id"] + "/target", json=TARGET)
            wait(client, request["request_id"], {"completed_audit"})
        health = client.get("/v1/health").json()
        scene = health["scene_schedule"][0]
        assert (scene["claimed"], scene["terminal"], scene["successful"]) == (3, 3, 2)
        assert scene["completion_reason"] == "success_goal_reached"
        assert health["worker_state"][0]["scene_binding"][1] == "b.glb"
        assert client.post("/v1/collections", json=requests[3]).status_code == 409
        assert not (cfg.collection.root / "tasks" / requests[3]["request_id"]).exists()
        assert client.post("/v1/collections", json=requests[1]).json()["status"] == "completed_audit"
        assert client.get("/v1/health").json()["scene_schedule"][0]["successful"] == 2


def test_rolling_budget_waits_for_target_and_drains_independently_before_rebinding(tmp_path):
    cfg = config(tmp_path, workers=2)
    cfg.collection.scene_schedule = [
        CollectionSceneBudget(dataset="train_pilot", scene_id=s, camera="trained_ce30", attempts=n)
        for s, n in [("a.glb", 2), ("b.glb", 1), ("c.glb", 1), ("d.glb", 1)]
    ]
    application = create_app(
        cfg,
        store=SimpleNamespace(
            scene_hashes={"train_pilot": {s.scene_id: "hash" for s in cfg.collection.scene_schedule}}
        ),
        actor_factory=FakeActor,
    )
    with TestClient(application) as client:
        a1, a2, b, c, d = [
            dict(task(name), scene_id=scene)
            for name, scene in [
                ("a1", "a.glb"),
                ("a2", "a.glb"),
                ("b", "b.glb"),
                ("c", "c.glb"),
                ("d", "d.glb"),
            ]
        ]
        assert client.post("/v1/collections", json=c).status_code == 409
        assert not (cfg.collection.root / "tasks" / c["request_id"]).exists()
        for request in [a1, a2, b]:
            assert client.post("/v1/collections", json=request).status_code == 200
            wait(client, request["request_id"], {"observation_ready"})
        original = client.get("/v1/health").json()["worker_state"]
        assert client.post("/v1/collections", json=a1).status_code == 200
        extra = dict(task("extra"), scene_id="a.glb")
        assert client.post("/v1/collections", json=extra).status_code == 409
        assert not (cfg.collection.root / "tasks" / extra["request_id"]).exists()
        client.post(f"/v1/collections/{a1['request_id']}/target", json=TARGET)
        wait(client, a1["request_id"], {"completed_audit"})
        client.delete("/v1/collections/" + b["request_id"])
        health = client.get("/v1/health").json()
        assert [s["status"] for s in health["scene_schedule"]] == [
            "active",
            "complete",
            "active",
            "queued",
        ]
        assert health["worker_state"][0]["scene_binding"][1] == "a.glb"
        assert health["worker_state"][1]["scene_binding"][1] == "c.glb"
        assert client.post("/v1/collections", json=c).status_code == 200
        ready = wait(client, c["request_id"], {"observation_ready"})
        assert read_json(Path(ready["output"]) / "observed.json")["pid"] == original[1]["pid"]
        client.delete("/v1/collections/" + c["request_id"])
        assert client.post("/v1/collections", json=d).status_code == 200
        wait(client, d["request_id"], {"observation_ready"})
        client.delete("/v1/collections/" + d["request_id"])
        assert client.get("/v1/health").json()["scene_schedule"][0]["status"] == "active"
        client.post(f"/v1/collections/{a2['request_id']}/target", json=TARGET)
        wait(client, a2["request_id"], {"completed_audit"})
        health = client.get("/v1/health").json()
        assert all(
            s["status"] == "complete" and s["terminal"] == s["attempts"]
            for s in health["scene_schedule"]
        )
        assert health["active"] == 0
        assert [w["pid"] for w in health["worker_state"]] == [w["pid"] for w in original]
        assert health["scene_schedule"][0]["claimed"] == 2
        history = list((cfg.collection.root / "instances").glob("*/workers/1/bindings/*.json"))
        assert len(history) == 2 and read_json(history[0])["reason"] == "scene_budget_drained"


@pytest.mark.parametrize(
    "changes",
    [
        {"worker_gpus": [0, 0, 0]},
        {"worker_scenes": [{"dataset": "train_pilot", "scene_id": "a.glb", "camera": "trained_ce30"}]},
        {"scene_cache_entries": 2},
        {
            "scene_schedule": [
                {"dataset": "train_pilot", "scene_id": "a.glb", "camera": "trained_ce30", "attempts": 1}
            ]
            * 2
        },
    ],
)
def test_rolling_configuration_rejects_ambiguous_or_unbounded_bindings(tmp_path, changes):
    value = config(tmp_path).collection.model_dump()
    value["scene_schedule"] = [
        dict(dataset="train_pilot", scene_id=s, camera="trained_ce30", attempts=1)
        for s in ["a.glb", "b.glb"]
    ]
    with pytest.raises(ValueError):
        CollectionSettings.model_validate(dict(value, **changes))


def test_rolling_worker_death_never_rebinds_or_replays_its_scene(tmp_path):
    cfg = config(tmp_path)
    cfg.collection.scene_schedule = [
        CollectionSceneBudget(dataset="train_pilot", scene_id=s, camera="trained_ce30", attempts=1)
        for s in ["a.glb", "b.glb"]
    ]
    application = create_app(
        cfg,
        store=SimpleNamespace(scene_hashes={"train_pilot": {"a.glb": "a", "b.glb": "b"}}),
        actor_factory=FakeActor,
    )
    with TestClient(application) as client:
        request = dict(task("rolling-crash", "r2r_chunks", "crash"), scene_id="a.glb")
        assert client.post("/v1/collections", json=request).status_code == 200
        wait(client, request["request_id"], {"failed_or_uncertain"})
        health = client.get("/v1/health").json()
        assert health["live_workers"] == 0
        assert [s["status"] for s in health["scene_schedule"]] == ["unavailable", "queued"]
        assert health["scene_schedule"][0]["claimed"] == health["scene_schedule"][0]["terminal"] == 1
        assert client.post("/v1/collections", json=request).json()["status"] == "failed_or_uncertain"
        new = dict(task("after-rolling-crash"), scene_id="b.glb")
        assert client.post("/v1/collections", json=new).status_code == 503
        assert not (cfg.collection.root / "tasks" / new["request_id"]).exists()
        assert client.get("/v1/health").json()["worker_state"][0]["binding_generation"] == 0


def fixed_app(tmp_path):
    cfg = config(tmp_path, capacity=32, workers=2)
    cfg.collection.worker_scenes = [
        CollectionScene(dataset="train_pilot", scene_id=s, camera="trained_ce30")
        for s in ["a.glb", "b.glb"]
    ]
    application = create_app(
        cfg,
        store=SimpleNamespace(
            scene_hashes={"train_pilot": {s: "hash" for s in ["a.glb", "b.glb", "c.glb"]}}
        ),
        actor_factory=FakeActor,
    )
    return cfg, application


def test_fixed_workers_keep_scene_ownership_across_prefetch_and_reverse_target_arrival(tmp_path):
    cfg, application = fixed_app(tmp_path)
    with TestClient(application) as client:
        requests = [dict(task(str(i)), scene_id="a.glb" if i % 3 else "b.glb") for i in range(12)]
        observations = {}
        scene_pids = {}
        for request in requests:
            assert client.post("/v1/collections", json=request).status_code == 200
        for request in requests:
            ready = wait(client, request["request_id"], {"observation_ready"})
            observed = read_json(Path(ready["output"]) / "observed.json")
            assert observed["loads"] == 1
            scene_pids.setdefault(request["scene_id"], set()).add(observed["pid"])
            observations[request["request_id"]] = (ready, observed)
        assert all(len(pids) == 1 for pids in scene_pids.values())
        assert scene_pids["a.glb"] != scene_pids["b.glb"]
        for request in reversed(requests):
            cid = request["request_id"]
            assert client.post(f"/v1/collections/{cid}/target", json=TARGET).status_code == 200
        for request in requests:
            cid = request["request_id"]
            result = wait(client, cid, {"completed_audit"})
            assert (
                read_json(Path(result["output"]) / "fake-execution.json")["pid"]
                == observations[cid][1]["pid"]
            )
        health = client.get("/v1/health").json()
        assert health["fixed_scene_window"]
        assert sum(w["operations"]["observe"] for w in health["worker_state"]) == 12
        assert sum(w["operations"]["execute"] for w in health["worker_state"]) == 12
        outside = dict(task("outside"), scene_id="c.glb")
        assert client.post("/v1/collections", json=outside).status_code == 422
        assert not (cfg.collection.root / "tasks" / outside["request_id"]).exists()


def test_fixed_worker_loss_settles_its_waiting_tasks_without_migration_or_replay(tmp_path):
    _, application = fixed_app(tmp_path)
    with TestClient(application) as client:
        waiting = dict(task("waiting"), scene_id="a.glb")
        client.post("/v1/collections", json=waiting)
        wait(client, waiting["request_id"], {"observation_ready"})
        crash = dict(task("fixed-crash", "r2r_chunks", "crash"), scene_id="a.glb")
        client.post("/v1/collections", json=crash)
        wait(client, crash["request_id"], {"failed_or_uncertain"})
        result = wait(client, waiting["request_id"], {"failed_or_uncertain"})
        assert not (Path(result["output"]) / "movement-attempted.json").exists()
        assert (
            client.post("/v1/collections", json=dict(task("unserved"), scene_id="a.glb")).status_code
            == 503
        )
        healthy = dict(task("healthy"), scene_id="b.glb")
        assert client.post("/v1/collections", json=healthy).status_code == 200
        wait(client, healthy["request_id"], {"observation_ready"})
        assert client.get("/v1/health").json()["live_workers"] == 1


def test_fixed_scene_configuration_must_cover_all_workers(tmp_path):
    with pytest.raises(ValueError, match="bind every"):
        CollectionSettings(
            root=tmp_path,
            training_datasets=["train"],
            worker_gpus=[0, 1],
            worker_scenes=[dict(dataset="train", scene_id="a", camera="camera")],
        )


def test_fixed_bindings_require_registered_training_assets(tmp_path):
    cfg, _ = fixed_app(tmp_path)
    cfg.collection.worker_scenes[0].dataset = "evaluation"
    with pytest.raises(ValueError, match="registered training"):
        with TestClient(app(cfg)):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_worker_busy,expected", [(False, "cold-scene"), (True, "warm-scene")])
async def test_idle_cached_worker_has_priority_without_limiting_busy_scene_parallelism(
    cached_worker_busy, expected
):
    service = CollectionService.__new__(CollectionService)
    service.closed = False
    service.condition = asyncio.Condition()
    service.executions = deque()
    service.observations = deque(["warm-scene", "cold-scene"])
    service.active = {
        cid: {"request": dict(dataset="train_pilot", camera="camera", scene_id=cid)}
        for cid in service.observations
    }
    service._persist = AsyncMock()
    cold = SimpleNamespace(instance="cold", current=None, warm=deque(), alive=True)
    warm = SimpleNamespace(
        instance="warm", current="running" if cached_worker_busy else None,
        warm=deque([("train_pilot", "warm-scene", "camera")]),
        alive=True,
    )
    service.workers = [cold, warm]
    phase, state = await service._next(cold)
    assert phase == "observe" and state["request"]["scene_id"] == expected
    if not cached_worker_busy:
        _, state = await service._next(warm)
        assert state["request"]["scene_id"] == "warm-scene"


def test_repeated_scene_hits_preserve_affinity_to_other_cached_scenes(tmp_path):
    cfg = config(tmp_path)
    cfg.collection.scene_cache_entries = 4
    application = create_app(
        cfg,
        store=SimpleNamespace(scene_hashes={"train_pilot": {s + ".glb": "hash" for s in "abcd"}}),
        actor_factory=FakeActor,
    )
    with TestClient(application) as client:
        for i, scene in enumerate("abcdaa"):
            request = task("cache-" + str(i))
            request["scene_id"] = scene + ".glb"
            assert client.post("/v1/collections", json=request).status_code == 200
            wait(client, request["request_id"], {"observation_ready"})
        worker = application.state.collection.workers[0]
        assert [key[1] for key in worker.warm] == ["b.glb", "c.glb", "d.glb", "a.glb"]
        health = client.get("/v1/health").json()
        assert health["scene_cache_entries_per_worker"] == 4
        assert health["maximum_cached_scene_instances"] == 4


def test_reference_preparation_is_read_only_and_releases_worker(tmp_path):
    with TestClient(app(config(tmp_path, capacity=1))) as client:
        query = task("reference", kind="reference_geometry")
        query["reference_guides"] = [[0, 0, 0], [0, 0, -2]]
        assert client.post("/v1/collections", json=query).status_code == 200
        result = wait(client, query["request_id"], {"reference_complete"})
        assert result["reference_legs"] == [{"distance_m": 2.0, "initial_yaw": 0.0}]
        assert not (Path(result["output"]) / "movement-attempted.json").exists()
        assert not (Path(result["output"]) / "front.png").exists()
        assert client.get("/v1/health").json()["active"] == 0
        assert client.post("/v1/collections", json=task("next")).status_code == 200


def test_prefetch_releases_worker_and_warm_scene_survives_across_independent_batches(tmp_path):
    cfg = config(tmp_path)
    with TestClient(app(cfg)) as client:
        requests = [task(str(i)) for i in range(4)]
        for request in requests:
            assert client.post("/v1/collections", json=request).status_code == 200
        # All four observations finish without any model response, with only one worker.
        ready = [wait(client, r["request_id"], {"observation_ready"}) for r in requests]
        assert all(not (Path(r["output"]) / "movement-attempted.json").exists() for r in ready)
        rows = [read_json(Path(r["output"]) / "observed.json") for r in ready]
        assert {r["loads"] for r in rows} == {1}
        assert len({r["pid"] for r in rows}) == 1
        for request in reversed(requests):
            cid = request["request_id"]
            assert client.post(f"/v1/collections/{cid}/target", json=TARGET).status_code == 200
            receipt = wait(client, cid, {"completed_audit"})
            assert receipt["training_admission"] == "pending_review"
            original = Path(receipt["output"]) / "receipt.json"
            old_hash = file_sha256(original)
            assert (
                client.post(f"/v1/collections/{cid}/target", json=TARGET).json()["status"]
                == "completed_audit"
            )
            assert file_sha256(original) == old_hash
        later = task("next-batch")
        client.post("/v1/collections", json=later)
        result = wait(client, later["request_id"], {"observation_ready"})
        assert read_json(Path(result["output"]) / "observed.json")["loads"] == 1
        assert (
            client.delete("/v1/collections/" + later["request_id"]).json()["status"]
            == "cancelled_before_movement"
        )
        assert client.get("/v1/health").json()["active"] == 0


def test_request_and_target_are_immutable_and_admission_is_bounded(tmp_path):
    cfg = config(tmp_path, capacity=1)
    with TestClient(app(cfg)) as client:
        request = task("only")
        cid = request["request_id"]
        client.post("/v1/collections", json=request)
        wait(client, cid, {"observation_ready"})
        assert client.post("/v1/collections", json=task("overflow")).status_code == 429
        assert client.post("/v1/collections", json=dict(request, seed=5)).status_code == 409
        assert client.post("/v1/collections", json=request).json()["request_id"] == cid
        client.post(f"/v1/collections/{cid}/target", json=TARGET)
        assert (
            client.post(
                f"/v1/collections/{cid}/target", json=dict(TARGET, target_view="left")
            ).status_code
            == 409
        )
        wait(client, cid, {"completed_audit"})
        assert client.post("/v1/collections", json=task("next")).status_code == 200


def test_r2r_execution_overlaps_annotation_and_new_session_uses_same_worker(tmp_path):
    cfg = config(tmp_path)
    with TestClient(app(cfg)) as client:
        for name in ["a", "b"]:
            request = task(name, "r2r_chunks")
            response = client.post("/v1/collections", json=request)
            assert response.status_code == 200
            result = wait(client, request["request_id"], {"completed_audit"})
            assert result["public"]["instruction"] == "go"
            assert not {"goal_position", "reference_guides", "start_position"} & result["public"].keys()
            assert read_json(Path(result["output"]) / "fake-execution.json")["loads"] == 1


def test_restart_never_replays_an_unsettled_claim(tmp_path):
    cfg = config(tmp_path)
    request = task("abandoned")
    cid = request["request_id"]
    folder = cfg.collection.root / "tasks" / cid

    create_json(folder / "request.json", TaskRequest.model_validate(request).model_dump(mode="json"))
    create_json(folder / "status.json", {"status": "executing"})
    with TestClient(app(cfg)) as client:
        result = client.post("/v1/collections", json=request)
        assert result.status_code == 200
        assert result.json()["status"] == "failed_or_uncertain"
        assert not (folder / "observed.json").exists()
        assert client.post(f"/v1/collections/{cid}/target", json=TARGET).status_code == 409


def test_worker_crash_preserves_attempt_and_is_not_automatically_restarted(tmp_path):
    cfg = config(tmp_path)
    with TestClient(app(cfg)) as client:
        request = task("crash", "r2r_chunks", "crash")
        client.post("/v1/collections", json=request)
        result = wait(client, request["request_id"], {"failed_or_uncertain"})
        assert (Path(result["output"]) / "movement-attempted.json").exists()
        assert client.get("/v1/health").json()["live_workers"] == 0
        assert client.post("/v1/collections", json=task("new")).status_code == 503
        assert client.post("/v1/collections", json=request).json()["status"] == "failed_or_uncertain"


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset": "val_unseen"},
        {"scene_id": "../outside.glb"},
        {"argv": ["untrusted"]},
        {"start_rotation": [0, 0, 0, 2]},
        {"goal_position": [1, 0, 1]},
    ],
)
def test_eval_inputs_arbitrary_commands_and_invalid_geometry_are_rejected(tmp_path, changes):
    with TestClient(app(config(tmp_path))) as client:
        response = client.post("/v1/collections", json=dict(task("bad"), **changes))
        assert response.status_code == 422
        assert client.get("/v1/health").json()["active"] == 0


def test_slow_cpu_plan_releases_simulator_and_next_task_survives_budget_exhaustion(tmp_path):
    cfg = config(tmp_path)
    cfg.collection.planning_timeout_s = 1.5
    with TestClient(app(cfg)) as client:
        slow = task("slow-cpu", "r2r_chunks", "planner-slow")
        client.post("/v1/collections", json=slow)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if client.get("/v1/collections/" + slow["request_id"]).json()["status"] == "planning":
                break
            time.sleep(0.01)
        independent = task("observe-during-planning")
        client.post("/v1/collections", json=independent)
        wait(client, independent["request_id"], {"observation_ready"})
        # Observation has finished while the CPU pool is still planning the first task.
        assert client.get("/v1/collections/" + slow["request_id"]).json()["status"] == "planning"
        terminal = wait(client, slow["request_id"], {"planning_budget_exhausted"})
        assert terminal["planning_outcome"] == "budget_exhausted"
        assert not (Path(terminal["output"]) / "movement-attempted.json").exists()
        assert client.get("/v1/health").json()["live_workers"] == 1
        client.post("/v1/collections/" + independent["request_id"] + "/target", json=TARGET)
        wait(client, independent["request_id"], {"completed_audit"})
        healthy = task("after-budget", "r2r_chunks")
        client.post("/v1/collections", json=healthy)
        wait(client, healthy["request_id"], {"completed_audit"})


def test_cancel_during_cpu_plan_never_enqueues_movement(tmp_path):
    cfg = config(tmp_path)
    with TestClient(app(cfg)) as client:
        slow = task("cancel-cpu", "r2r_chunks", "planner-slow")
        client.post("/v1/collections", json=slow)
        time.sleep(0.1)
        client.delete("/v1/collections/" + slow["request_id"])
        terminal = wait(client, slow["request_id"], {"cancelled_before_movement"})
        assert not (Path(terminal["output"]) / "movement-attempted.json").exists()
        assert client.get("/v1/health").json()["live_workers"] == 1
