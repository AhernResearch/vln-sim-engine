"""Real spawned CPU workers and HTTP contracts, with deterministic fault injection."""

import hashlib
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient
from planning_fakes import FakePlanningEngine

from sim_engine.core.io import file_sha256, fingerprint
from sim_engine.planning.client import HTTPPlanningClient, wait_for_plan
from sim_engine.planning.context import PlanningContext, PlanningStopped, checkpoint
from sim_engine.planning.contracts import Blob, PlannerSettings, PlanningBudget, PlanRequest
from sim_engine.planning.service import create_app


def config(tmp_path):
    mesh = tmp_path / "only.navmesh"
    mesh.write_bytes(b"pinned CPU-only asset")
    return PlannerSettings(root=tmp_path / "planner", workers=1, capacity=2,
                           hard_timeout_grace_s=0.1,
                           assets=[dict(dataset="train", scene_id="scene.glb", path=mesh,
                                        sha256=file_sha256(mesh))])


def job(cfg, instruction="go", **budget):
    return PlanRequest(attempt_id=uuid.uuid4().hex, navmesh_sha256=cfg.assets[0].sha256,
        camera={"width": 32, "height": 32}, budget=budget or {}, task={
            "owner": uuid.uuid4().hex, "request_id": hashlib.sha256(uuid.uuid4().bytes).hexdigest(),
            "dataset": "train", "scene_id": "scene.glb", "camera": "camera", "seed": 1,
            "kind": "r2r_chunks", "instruction": instruction, "goal_position": [0, 0, -2],
            "reference_guides": [[0, 0, 0], [0, 0, -2]],
            "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1]})


def wait(client, cid):
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        state = client.get("/v1/plans/" + cid).json()
        if state["status"] not in {"queued", "planning"}:
            return state
        time.sleep(0.01)
    raise AssertionError(state)


def test_blocked_plan_claim_keeps_health_responsive_and_duplicate_claim_unique(tmp_path, monkeypatch):
    from sim_engine.planning import service

    cfg = config(tmp_path)
    value = job(cfg)
    entered, release = Event(), Event()
    original = service.create_json
    writes = []

    def blocked(path, document):
        if path.name == "input.json":
            writes.append(path)
            entered.set()
            assert release.wait(5)
        original(path, document)

    with TestClient(create_app(cfg, engine_factory=FakePlanningEngine)) as client, ThreadPoolExecutor(max_workers=3) as calls:
        monkeypatch.setattr(service, "create_json", blocked)
        first = calls.submit(client.post, "/v1/plans", json=value.model_dump(mode="json"))
        try:
            assert entered.wait(2)
            health = calls.submit(client.get, "/v1/health").result(timeout=1).json()
            assert health["evidence_io"]["pending"] >= 1
            assert client.get("/v1/plans/" + value.attempt_id).json()["status"] == "claiming"
            duplicate = calls.submit(client.post, "/v1/plans", json=value.model_dump(mode="json"))
            release.set()
            assert first.result(timeout=3).status_code == duplicate.result(timeout=3).status_code == 200
            assert wait(client, value.attempt_id)["status"] == "feasible"
            assert len(writes) == 1
        finally:
            release.set()


@pytest.mark.parametrize("instruction,expected", [
    ("planner-crash", "worker_failed"), ("planner-hang", "budget_exhausted"),
    ("planner-slow", "budget_exhausted"),
])
def test_fault_isolated_to_one_attempt_capacity_replaced_without_replay(tmp_path, instruction, expected):
    cfg = config(tmp_path)
    with TestClient(create_app(cfg, engine_factory=FakePlanningEngine)) as client:
        first = job(cfg, instruction, wall_s=0.2)
        assert client.post("/v1/plans", json=first.model_dump(mode="json")).status_code == 200
        bad = wait(client, first.attempt_id)
        assert bad["status"] == expected
        receipt = cfg.root / "attempts" / first.attempt_id / "result.json"
        before = file_sha256(receipt)
        assert client.post("/v1/plans", json=first.model_dump(mode="json")).json() == bad
        second = job(cfg)
        client.post("/v1/plans", json=second.model_dump(mode="json"))
        assert wait(client, second.attempt_id)["status"] == "feasible"
        assert file_sha256(receipt) == before
        assert not list(cfg.root.rglob("movement-attempted.json"))


def test_queue_bounds_cancellation_digest_and_restart_preserve_attempts(tmp_path):
    cfg = config(tmp_path)
    application = create_app(cfg, engine_factory=FakePlanningEngine)
    with TestClient(application) as client:
        a, b, c = job(cfg, "planner-slow", wall_s=4), job(cfg), job(cfg)
        client.post("/v1/plans", json=a.model_dump(mode="json"))
        client.post("/v1/plans", json=b.model_dump(mode="json"))
        assert client.post("/v1/plans", json=c.model_dump(mode="json")).status_code == 429
        assert not (cfg.root / "attempts" / c.attempt_id).exists()
        assert client.delete("/v1/plans/" + b.attempt_id).json()["status"] == "cancelled"
        client.delete("/v1/plans/" + a.attempt_id)
        assert wait(client, a.attempt_id)["status"] == "cancelled"
        changed = a.model_copy(update={"navmesh_sha256": "f" * 64})
        assert client.post("/v1/plans", json=changed.model_dump(mode="json")).status_code == 409
        before = client.get("/v1/plans/" + a.attempt_id).json()
        assert before["result_digest"] == fingerprint({k: v for k, v in before.items() if k != "result_digest"})
    with TestClient(create_app(cfg, engine_factory=FakePlanningEngine)) as client:
        assert client.post("/v1/plans", json=a.model_dump(mode="json")).json() == before


def test_budget_is_shared_across_candidate_scopes_and_not_swallowed_as_valueerror():
    from threading import Event
    context = PlanningContext(PlanningBudget(max_expansions=2), Event())
    with context.bind():
        checkpoint(expansion=1)
        checkpoint(expansion=1)
        with pytest.raises(PlanningStopped) as error:
            checkpoint(expansion=1)
    assert error.value.outcome == "budget_exhausted"
    assert not isinstance(error.value, ValueError)
    checkpoint(expansion=1000000)  # No ambient budget leaks into simulator/audit operations.


def test_blob_rejects_corruption_and_job_rejects_host_paths(tmp_path):
    cfg = config(tmp_path)
    blob = Blob.encode(b"test", "test.json")
    assert blob.decode() == b"test"
    blob.size += 1
    with pytest.raises(ValueError):
        blob.decode()
    value = job(cfg).model_dump(mode="json")
    value["artifacts"] = {"../../argv.json": Blob.encode(b"{}", "x.json").model_dump()}
    with pytest.raises(ValueError):
        PlanRequest.model_validate(value)


@pytest.mark.asyncio
async def test_ambiguous_post_uses_same_endpoint_get_only():
    calls = []
    attempt = "a" * 32
    async def handler(request):
        calls.append((request.method, request.url.host, request.url.path))
        if request.url.path == "/v1/health":
            return httpx.Response(200, json={"service_owner": "owned", "implementation_digest": client.implementation_digest})
        if request.method == "POST":
            raise httpx.ReadTimeout("reply lost after claim")
        return httpx.Response(200, json={"attempt_id": attempt, "status": "planning"})
    client = HTTPPlanningClient(["http://first", "http://second"])
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    value = type("Job", (), {"attempt_id": attempt, "model_dump": lambda *a, **k: {}})()
    assert (await client.submit(value))["status"] == "planning"
    assert sum(method == "POST" for method, _, _ in calls) == 1
    assert all(host == "first" for _, host, _ in calls)
    await client.close()


@pytest.mark.asyncio
async def test_wait_rejects_stale_result_before_any_execution(tmp_path):
    value = job(config(tmp_path))
    class Client:
        async def submit(self, job):
            return {"status": "feasible", "task_id": job.task.request_id, "input_digest": "0" * 64}
    with pytest.raises(RuntimeError, match="frozen input"):
        await wait_for_plan(Client(), value, lambda: False)


@pytest.mark.asyncio
async def test_remote_shutdown_cancels_only_bound_attempts_and_keeps_endpoint_evidence(tmp_path):
    cfg = config(tmp_path)
    value = job(cfg)
    calls = []
    client = HTTPPlanningClient(["http://planner"])
    async def handler(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/v1/health":
            return httpx.Response(200, json={"service_owner": "owned",
                "implementation_digest": client.implementation_digest})
        return httpx.Response(200, json={"status": "planning"})
    await client.http.aclose()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client.prepare(value, tmp_path / "collection-task")
    await client.submit(value)
    evidence = list((tmp_path / "collection-task/planning-endpoints").glob("*.json"))
    assert len(evidence) == 1
    from sim_engine.core.io import read_json
    assert read_json(evidence[0])["attempt_id"] == value.attempt_id
    assert read_json(evidence[0])["endpoint"] == "http://planner"
    await client.close()
    assert [path for method, path in calls if method == "DELETE"] == ["/v1/plans/" + value.attempt_id]
    assert not client.bindings and not client.evidence


def test_exported_navmesh_bundle_moves_without_glbs_or_episode_store(tmp_path):
    import shutil
    from types import SimpleNamespace

    from helpers import settings

    from sim_engine.config import CollectionSettings, Dataset
    from sim_engine.planning.assets import export_assets
    source = settings(tmp_path)
    source.simulation.datasets = {"train": Dataset(episodes=tmp_path / "not-present.json")}
    source.collection = CollectionSettings(root=tmp_path / "unused", training_datasets=["train"])
    mesh = source.simulation.scenes_root / "scene.navmesh"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_bytes(b"navmesh")
    export_assets(source, tmp_path / "bundle", store=SimpleNamespace(scene_hashes={"train": {"scene.glb": "hash"}}))
    shutil.move(tmp_path / "bundle", tmp_path / "relocated")
    config = PlannerSettings.read(tmp_path / "relocated/planner.json")
    assert config.assets[0].path.read_bytes() == b"navmesh"
    assert config.root == tmp_path / "relocated/jobs"
    assert not list((tmp_path / "relocated").rglob("*.glb"))
