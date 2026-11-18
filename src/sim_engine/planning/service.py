"""Bounded CPU queue with durable attempt identities and replaceable process capacity.

Only read_json-only planning processes may be replaced. A killed job is settled once;
its input is never automatically submitted to another process.
"""

from __future__ import annotations

import asyncio
import platform
import sys
import time
import uuid
from collections import deque
from pathlib import Path

from sim_engine.core.disk import AsyncDisk
from sim_engine.core.http import service_app
from sim_engine.core.io import create_json, fingerprint, implementation_sources, read_json, replace_json
from sim_engine.core.process import SPAWN, ProcessWorker, WorkerError, WorkerTimeout
from sim_engine.navigation.native import require_native

from .contracts import PlanRequest
from .engine import PlanningEngine


def runtime_versions():
    """Native libraries are loaded only by the CPU engine, never by the HTTP parent."""
    versions = {"python": platform.python_version()}
    for name in ("habitat_sim", "numpy"):
        if name in sys.modules:
            versions[name] = getattr(sys.modules[name], "__version__", "unknown")
    return versions


def worker_failed(error):
    return {"outcome": "worker_failed", "error": error, "artifacts": {}}


class PlannerHost:
    """Child-side owner of one CPU planning engine."""

    def __init__(self, stop, settings, factory, progress):
        self.stop, self.progress = stop, progress
        self.engine = factory(settings)

    def handle(self, payload):
        value, folder = payload
        job = PlanRequest.model_validate(value)
        try:
            result = self.engine.run(job, Path(folder), self.stop, progress=self.progress)
            result["runtime"] = runtime_versions()
        except Exception as error:
            result = worker_failed(repr(error))
        return 200, result

    def close(self):
        pass  # The engine holds only process memory.


class PlanningWorker(ProcessWorker):
    """Replaceable CPU capacity; a terminated attempt is settled, never resubmitted."""

    kind = "planner"

    def __init__(self, settings, factory, folder, service_owner):
        self.settings, self.folder, self.service_owner = settings, folder, service_owner
        self.progress = SPAWN.RawArray("d", [0, 0, 0])  # queries, expansions, last checkpoint unix
        self.current = None
        self.cancel_began = None
        super().__init__(PlannerHost, settings, factory, self.progress)

    def spawn(self):
        super().spawn()
        ticks = Path(f"/proc/{self.process.pid}/stat")
        create_json(self.folder / self.instance / "ownership.json", {
            "service_owner": self.service_owner, "instance": self.instance,
            "pid": self.process.pid,
            "start_ticks": ticks.read_text().rsplit(")", 1)[1].split()[19] if ticks.exists() else None,
            "generation": self.generation, "automatic_replay": False,
        })

    def call(self, job, folder):
        self.cancel_began = None
        grace = self.settings.hard_timeout_grace_s
        try:
            _, result = self.exchange(job.attempt_id, (job.model_dump(mode="json"), str(folder)),
                                      timeout=job.budget.wall_s + grace,
                                      interrupted=self._cancel_expired)
        except WorkerTimeout:
            outcome = "budget_exhausted" if self.cancel_began is None else "cancelled"
            return {"outcome": outcome, "hard_timeout": True, "artifacts": {}}
        except WorkerError as error:
            self.terminate()
            return worker_failed(repr(error))
        if result.get("outcome") == "worker_failed":
            self.terminate()
        return result

    def _cancel_expired(self):
        if not self.stop.is_set():
            return False
        if self.cancel_began is None:
            self.cancel_began = time.monotonic()
        return time.monotonic() - self.cancel_began > self.settings.hard_timeout_grace_s


class PlanningService:
    def __init__(self, settings, engine_factory=PlanningEngine):

        require_native()
        self.settings = settings
        self.owner = uuid.uuid4().hex
        self.root = settings.root.resolve()
        self.instance_root = self.root / "instances" / self.owner
        self.assets = {(a.dataset, a.scene_id): a.sha256 for a in settings.assets}

        self.sources = implementation_sources()
        self.implementation_digest = fingerprint(self.sources)
        create_json(self.instance_root / "ownership.json", {
            "owner": self.owner, "settings": settings.model_dump(mode="json"),
            "source_sha256": self.sources, "automatic_replay": False,
        })
        self.queue = deque()
        self.work_available = asyncio.Event()
        self.active = {}
        self.disk = AsyncDisk()
        self.lock = asyncio.Lock()
        self.closed = False
        self.workers = [PlanningWorker(settings, engine_factory,
                        self.instance_root / "workers" / str(i), self.owner)
                        for i in range(settings.workers)]
        self.runners = [asyncio.create_task(self.consume(w)) for w in self.workers]

    def folder(self, cid):
        if len(cid) != 32 or any(c not in "0123456789abcdef" for c in cid):
            raise WorkerError(422, "invalid planning attempt identity")
        return self.root / "attempts" / cid

    async def status(self, cid):
        if cid in self.active:
            state = self.active[cid]
            worker = next((w for w in self.workers if w.current == cid), None)
            return dict(state["view"], progress=list(worker.progress[:]) if worker else None)
        folder = self.folder(cid)
        return await self.disk.run(self._stored_status, cid, folder)

    @staticmethod
    def _stored_status(cid, folder):
        if (folder / "result.json").exists():
            return read_json(folder / "result.json")
        if folder.exists():
            return {"attempt_id": cid, "status": "worker_failed",
                    "error": "unsettled prior lifetime; no replay", "automatic_replay": False}
        raise WorkerError(404, "unknown planning attempt")

    async def submit(self, job):
        async with self.lock:
            return await self._submit(job)

    async def _submit(self, job):
        folder = self.folder(job.attempt_id)
        if await self.disk.run(folder.exists):
            old = await self.disk.run(read_json, folder / "input.json")
            if old["input_digest"] != job.digest():
                raise WorkerError(409, "planning identity reused with different input")
            return await self.status(job.attempt_id)
        if self.closed:
            raise WorkerError(503, "planner closed")
        if len(self.active) >= self.settings.capacity:
            raise WorkerError(429, "planning capacity exhausted; no claim created")
        if self.assets.get((job.task.dataset, job.task.scene_id)) != job.navmesh_sha256:
            raise WorkerError(422, "scene is outside pinned training NavMesh registry")
        view = {"attempt_id": job.attempt_id, "task_id": job.task.request_id,
                "input_digest": job.digest(), "status": "claiming", "service_owner": self.owner,
                "implementation_digest": self.implementation_digest, "automatic_replay": False}
        self.active[job.attempt_id] = {"job": job, "view": view}
        try:
            await self.disk.run(lambda: folder.mkdir(parents=True))
            await self.disk.run(create_json, folder / "input.json", {"input_digest": job.digest(), "request": job.model_dump(mode="json")})
        except BaseException:
            self.active.pop(job.attempt_id, None)
            raise
        view["status"] = "queued"
        await self.disk.run(replace_json, folder / "status.json", view)
        self.queue.append(job.attempt_id)
        self.work_available.set()
        return view

    async def finish(self, state, result):
        cid = state["job"].attempt_id
        result = dict(result)
        outcome = result.pop("outcome")
        receipt = dict(state["view"], **result)
        receipt.update(status=outcome, navmesh_sha256=state["job"].navmesh_sha256)
        receipt["result_digest"] = fingerprint(receipt)
        await self.disk.run(create_json, self.folder(cid) / "result.json", receipt)
        self.active.pop(cid, None)
        self.queue = deque(i for i in self.queue if i != cid)
        return receipt

    async def cancel(self, cid):
        async with self.lock:
            return await self._cancel(cid)

    async def _cancel(self, cid):
        state = self.active.get(cid)
        if state is None:
            return await self.status(cid)
        state["cancel_requested"] = True
        worker = next((w for w in self.workers if w.current == cid), None)
        if worker is not None:
            worker.stop.set()
            return await self.status(cid)
        return await self.finish(state, {"outcome": "cancelled", "artifacts": {}})

    async def consume(self, worker):
        while not self.closed:
            await self.work_available.wait()
            async with self.lock:
                if self.closed:
                    return
                if not self.queue:
                    self.work_available.clear()
                    continue
                cid = self.queue.popleft()
                state = self.active.get(cid)
                if state is None:
                    continue
                job = state["job"]
                worker.current = cid
                if not worker.alive:
                    # Replace capacity for FUTURE claims only. The failed attempt stays terminal.
                    await self.disk.run(worker.spawn)
                worker.stop.clear()
                worker.progress[:] = [0, 0, 0]
                state["view"]["status"] = "planning"
                state["view"]["worker_instance"] = worker.instance
                await self.disk.run(replace_json, self.folder(cid) / "status.json", state["view"])
                scratch = self.folder(cid) / "artifacts"
                await self.disk.run(scratch.mkdir)
            try:
                result = await asyncio.to_thread(worker.call, job, scratch)
                async with self.lock:
                    if state.get("cancel_requested"):
                        result["outcome"] = "cancelled"
                    await self.finish(state, result)
            finally:
                worker.current = None

    def health(self):
        return {"status": "ok" if not self.closed else "closed", "service_owner": self.owner,
                "evidence_io": self.disk.health(),
                "capacity": self.settings.capacity, "active": len(self.active),
                "queued": len(self.queue), "workers": len(self.workers),
                "live_workers": sum(w.alive for w in self.workers),
                "implementation_digest": self.implementation_digest,
                "assets": [{"dataset": a.dataset, "scene_id": a.scene_id, "sha256": a.sha256}
                           for a in self.settings.assets],
                "worker_state": [{"instance": w.instance, "pid": w.process.pid,
                                  "attempt_id": w.current, "progress": list(w.progress[:])}
                                 for w in self.workers]}

    async def close(self):
        self.closed = True
        for worker in self.workers:
            worker.stop.set()
        await asyncio.gather(*(asyncio.to_thread(w.close) for w in self.workers))
        for runner in self.runners:
            runner.cancel()
        await asyncio.gather(*self.runners, return_exceptions=True)
        for state in list(self.active.values()):
            await self.finish(state, worker_failed("planner stopped; no replay"))
        await self.disk.close()


def create_app(settings, *, engine_factory=PlanningEngine):
    async def open_service():
        return PlanningService(settings, engine_factory)

    app = service_app("CPU NavMesh planning service", "planner", open_service)

    @app.get("/v1/health")
    async def health():
        return app.state.planner.health()

    @app.post("/v1/plans")
    async def submit(job: PlanRequest):
        return await app.state.planner.submit(job)

    @app.get("/v1/plans/{cid}")
    async def status(cid: str):
        return await app.state.planner.status(cid)

    @app.delete("/v1/plans/{cid}")
    async def cancel(cid: str):
        return await app.state.planner.cancel(cid)

    return app
