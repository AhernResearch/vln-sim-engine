"""Identical job interface for an embedded pool and one or more remote services."""

import asyncio
import time
import uuid
from urllib.parse import urlsplit

import httpx

from sim_engine.core.io import create_json, file_sha256, fingerprint, read_json
from sim_engine.core.process import WorkerError
from sim_engine.scenes import resolve_scene

from .assets import registered_assets
from .contracts import Blob, PlannerSettings, PlanRequest
from .service import PlanningService

TERMINAL = {"feasible", "infeasible", "budget_exhausted", "cancelled", "worker_failed"}


class LocalPlanningClient:
    def __init__(self, settings, store, root, engine_factory=None):
        config = settings.collection
        assets = registered_assets(settings, store)
        options = {} if engine_factory is None else {"engine_factory": engine_factory}
        self.service = PlanningService(PlannerSettings(root=root, assets=assets,
            workers=config.planning_workers, capacity=config.planning_capacity,
            scene_cache_entries=config.planning_scene_cache_entries), **options)

    def prepare(self, job, folder):
        pass

    def release(self, attempt_id):
        pass

    async def submit(self, job):
        return await self.service.submit(job)

    async def status(self, attempt_id):
        return await self.service.status(attempt_id)

    async def cancel(self, attempt_id):
        return await self.service.cancel(attempt_id)

    async def close(self):
        await self.service.close()

    def health(self):
        return self.service.health()


class HTTPPlanningClient:
    def __init__(self, urls):
        for url in urls:
            value = urlsplit(url)
            if (value.scheme not in {"http", "https"} or not value.hostname
                    or value.username or value.password or value.query or value.fragment):
                raise ValueError("planner endpoint must be an HTTP origin without credentials")
        self.urls = [u.rstrip("/") for u in urls]
        self.http = httpx.AsyncClient(timeout=10, trust_env=False)
        self.bindings = {}
        self.evidence = {}
        self.identities = {}
        self.last_health = {}
        self.cursor = 0
        from sim_engine.core.io import implementation_sources

        self.implementation_digest = fingerprint(implementation_sources())

    def prepare(self, job, folder):
        self.evidence[job.attempt_id] = folder

    def release(self, attempt_id):
        self.bindings.pop(attempt_id, None)
        self.evidence.pop(attempt_id, None)

    async def request(self, url, method, path, value=None):
        response = await self.http.request(method, url + path, json=value)
        if response.is_error:
            raise WorkerError(response.status_code, response.text[:1000])
        return response.json()

    async def submit(self, job):
        # Only 429/503 BEFORE an attempt claim allow trying another endpoint.
        # Ambiguous responses retain the endpoint binding; never submit elsewhere.
        candidates = self.urls[self.cursor:] + self.urls[:self.cursor]
        self.cursor = (self.cursor + 1) % len(self.urls)
        for url in candidates:
            try:
                health = await self.request(url, "GET", "/v1/health")
            except (httpx.HTTPError, WorkerError):
                continue
            self.last_health[url] = health
            if health["implementation_digest"] != self.implementation_digest:
                continue
            identity = (health["service_owner"], health["implementation_digest"])
            if url in self.identities and self.identities[url] != identity:
                continue
            self.identities[url] = identity
            self.bindings[job.attempt_id] = url
            folder = self.evidence.get(job.attempt_id)
            if folder is not None:
                claim = {"endpoint": url, "service_owner": health["service_owner"],
                         "implementation_digest": health["implementation_digest"],
                         "attempt_id": job.attempt_id, "input_digest": job.digest()}
                path = folder / "planning-endpoints" / (fingerprint(claim) + ".json")
                if not path.exists():
                    create_json(path, claim)
            try:
                return await self.request(url, "POST", "/v1/plans", job.model_dump(mode="json"))
            except WorkerError as error:
                if error.status in {429, 503}:
                    self.bindings.pop(job.attempt_id)
                    continue
                raise
            except httpx.HTTPError:
                # Resolve identity using GET at the SAME service, not a second POST.
                return await self.status(job.attempt_id)
        raise WorkerError(503, "no planner endpoint admitted the read_json-only task")

    async def status(self, attempt_id):
        return await self.request(self.bindings[attempt_id], "GET", "/v1/plans/" + attempt_id)

    async def cancel(self, attempt_id):
        return await self.request(self.bindings[attempt_id], "DELETE", "/v1/plans/" + attempt_id)

    async def close(self):
        # Remote processes may serve other collectors; cancel only our active attempt IDs.
        ids = list(self.bindings)
        results = await asyncio.gather(*(self.cancel(cid) for cid in ids), return_exceptions=True)
        for cid, result in zip(ids, results):
            folder = self.evidence.get(cid)
            if isinstance(result, BaseException) and folder is not None:
                path = folder / "planning-cancel-unresolved.json"
                if not path.exists():
                    create_json(path, {"attempt_id": cid, "endpoint": self.bindings[cid],
                                 "error": repr(result), "automatic_replay": False})
            self.release(cid)
        await self.http.aclose()

    def health(self):
        return {"deployment": "remote", "endpoints": self.last_health}


def build_job(settings, request, folder, target):
    config = settings.collection
    navmesh = resolve_scene(settings, request["dataset"], request["scene_id"]).navmesh
    artifacts = {}
    if request["kind"] == "short_instructions":
        names = ["start-state.json", "camera.json", "view-sensors.json", target["target_view"] + "-depth.npz"]
        artifacts = {name: Blob.encode((folder / name).read_bytes(), name) for name in names}
    return PlanRequest(attempt_id=uuid.uuid4().hex, task=request, navmesh_sha256=file_sha256(navmesh),
                       camera=settings.cameras[request["camera"]], target=target, artifacts=artifacts,
                       profile={"version": config.planning_profile_version, "base_steps": config.base_steps},
                       budget={"wall_s": config.planning_timeout_s,
                               "max_expansions": config.planning_max_expansions,
                               "max_queries": config.planning_max_queries})


async def wait_for_plan(client, job, cancel_requested):
    began = time.monotonic()
    while True:
        try:
            state = await client.submit(job)
            break
        except WorkerError as error:
            if error.status not in {429, 503} or time.monotonic() - began > 120:
                raise
            if cancel_requested():
                state = {"attempt_id": job.attempt_id, "task_id": job.task.request_id,
                         "input_digest": job.digest(), "status": "cancelled",
                         "not_admitted": True, "artifacts": {}}
                state["result_digest"] = fingerprint(state)
                break
            # Explicit no-claim responses only; ambiguous transports never enter this loop.
            await asyncio.sleep(0.25)
    cancel_sent = False
    # Includes bounded waiting time. Never interpret transport loss as infeasibility.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + job.budget.wall_s + 120
    while state["status"] not in TERMINAL:
        if cancel_requested() and not cancel_sent:
            await client.cancel(job.attempt_id)
            cancel_sent = True
        if loop.time() > deadline:
            await client.cancel(job.attempt_id)
            raise TimeoutError("planner result unresolved; no execution or resubmission")
        await asyncio.sleep(0.05)
        state = await client.status(job.attempt_id)
    if state.get("input_digest") != job.digest() or state.get("task_id") != job.task.request_id:
        raise RuntimeError("planning result does not match frozen input")
    expected = state.get("result_digest")
    if expected != fingerprint({k: v for k, v in state.items() if k != "result_digest"}):
        raise RuntimeError("planning result file_sha256 mismatch")
    release = getattr(client, "release", None)
    if release is not None:
        release(job.attempt_id)
    return state


def install_result(folder, job, result):
    import re

    from .contracts import OUTPUT_NAMES

    blobs = {name: Blob.model_validate(value) for name, value in result.get("artifacts", {}).items()}
    if sum(blob.size for blob in blobs.values()) > 128 * 1024**2:
        raise ValueError("planning result exceeds artifact budget")
    for name in blobs:
        if name not in OUTPUT_NAMES and not re.fullmatch(r"pose-trials/[0-9]{3}\.json", name):
            raise ValueError("unsupported planning result artifact")
    for name, blob in blobs.items():
        path = folder / name
        content = blob.decode()
        # Free-task start state already exists and must never change.
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError("planner output conflicts with existing evidence")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(content)
    create_json(folder / "planning-result.json", result)
    if result["status"] != "feasible" or job.task.kind == "reference_geometry":
        return
    prepared = read_json(folder / "private-geometry.json")
    if prepared != result["prepared"]:
        raise ValueError("prepared plan differs from verified artifact")
    start = prepared["episode"]["start_position"]
    if not prepared["path"] or sum((a-b)**2 for a, b in zip(start, prepared["path"][0])) > 1e-6:
        raise ValueError("plan path does not start at its episode pose")
    if job.task.kind == "short_instructions" and start != list(job.task.start_position):
        raise ValueError("free-task planner moved the frozen start")
    create_json(folder / "execution-lease.json", {
        "identity": uuid.uuid4().hex, "task_id": job.task.request_id,
        "planning_attempt": job.attempt_id, "plan_digest": result["result_digest"],
        "prepared_sha256": file_sha256(folder / "private-geometry.json"),
        "runtime": result.get("runtime", {}), "navmesh_sha256": job.navmesh_sha256,
        "automatic_retry": False,
    })
