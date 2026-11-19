"""Privileged persistent collection server: bounded prefetch, warm workers, local RGB-D.

The HTTP boundary carries task descriptions and receipts, never executable code.
Workers perform whole observation/execution operations and are never auto-restarted.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import uuid
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sim_engine.collection.protocol import StatusRequest
from sim_engine.core.disk import AsyncDisk
from sim_engine.core.http import service_app
from sim_engine.core.io import create_json, implementation_sources, read_json, replace_json
from sim_engine.core.process import ProcessWorker, WorkerError
from sim_engine.data import EpisodeStore
from sim_engine.navigation.native import require_native
from sim_engine.planning.client import (
    HTTPPlanningClient,
    LocalPlanningClient,
    build_job,
    install_result,
    wait_for_plan,
)
from sim_engine.planning.tasks import RegionRequest, TaskRequest

from .audit import audit_recording
from .backend import CollectionActor, geometry_config


def collection_key(request):
    return request["dataset"], request["scene_id"], request["camera"]


class ActorHost:
    """Child-side owner of one collection actor and its warm scenes."""

    def __init__(self, stop, settings, gpu, factory):
        self.stop = stop
        self.actor = factory(settings, gpu)

    def handle(self, payload):
        phase, request, folder, target = payload
        folder = Path(folder)
        try:
            if phase == "observe":
                return 200, self.actor.observe(request, folder, self.stop)
            return 200, self.actor.execute(request, folder, target, self.stop)
        except Exception as error:
            moved = (folder / "movement-attempted.json").exists()
            rejected = isinstance(error, ValueError) and not moved
            if not rejected:
                self.actor.close()
            return (422 if rejected else 500), {"error": repr(error), "movement_attempted": moved}

    def close(self):
        self.actor.close()


class ActorWorker(ProcessWorker):
    kind = "collection"

    def __init__(self, settings, gpu, factory, owner, folder, binding=None):
        self.binding = None if binding is None else collection_key(binding.model_dump())
        self.gpu, self.folder = gpu, folder
        self.binding_generation = 0
        super().__init__(ActorHost, settings, gpu, factory)
        create_json(folder / "ownership.json", {"owner": owner, "instance": self.instance, "gpu": gpu,
                                                "scene_binding": self.binding})
        create_json(folder / "process.json",
                    {"owner": owner, "instance": self.instance, "pid": self.process.pid})
        self.timeout = settings.collection.operation_timeout_s
        self.current = None
        self.current_phase = None
        self.operation_counts = Counter()
        self.busy_s = Counter()
        self.last_cache = {}
        self.warm = deque(maxlen=settings.collection.scene_cache_entries)

    async def rebind(self, binding, disk):
        if self.current is not None:
            raise RuntimeError("cannot rebind a busy collection worker")
        generation = self.binding_generation + 1
        await disk.run(create_json, self.folder / "bindings" / f"{generation:06d}.json", {
            "instance": self.instance, "generation": generation, "previous": self.binding,
            "next": binding, "reason": "scene_budget_drained"})
        self.binding = binding
        self.binding_generation = generation
        self.warm.clear()

    async def call(self, phase, request, folder, target):
        if self.binding is not None and collection_key(request) != self.binding:
            raise WorkerError(409, "worker scene binding violation; no operation submitted")
        self.current = request["request_id"]
        self.current_phase = phase
        self.operation_counts[phase] += 1
        began = time.perf_counter()
        self.stop.clear()
        try:
            status, result = await self.submit((request["request_id"], phase),
                                               (phase, request, str(folder), target),
                                               timeout=self.timeout)
            if status != 200:
                raise WorkerError(status, json.dumps(result))
            self.last_cache = result.get("cache", self.last_cache)
            return result
        finally:
            self.busy_s[phase] += time.perf_counter() - began
            self.current = None
            self.current_phase = None


class CollectionService:
    def __init__(self, settings, store, factory=CollectionActor):
        if settings.collection is None:
            raise ValueError("collection service requires an explicit collection configuration")
        require_native()
        self.settings, self.store = settings, store
        self.owner = uuid.uuid4().hex
        self.root = settings.collection.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.instance_root = self.root / "instances" / self.owner
        code_root = Path(__file__).parent.parent

        self.source_sha256 = implementation_sources(code_root)
        create_json(
            self.instance_root / "ownership.json",
            {
                "owner": self.owner,
                "automatic_retry": False,
                "settings": {
                    k: v
                    for k, v in settings.model_dump(mode="json").items()
                    if k in {"simulation", "cameras", "collection"}
                },
                "source_sha256": self.source_sha256,
                "dataset_identity": store.scene_hashes,
            },
        )
        if any(
            name not in settings.simulation.datasets for name in settings.collection.training_datasets
        ):
            raise ValueError("collection training datasets must be registered in simulation settings")
        bindings = settings.collection.worker_scenes
        schedule = settings.collection.scene_schedule
        for binding in schedule or bindings or []:
            if (
                binding.dataset not in settings.collection.training_datasets
                or binding.camera not in settings.cameras
                or binding.scene_id not in store.scene_hashes.get(binding.dataset, {})
            ):
                raise ValueError("worker scene bindings must use registered training scenes and cameras")
        if schedule is not None:
            bindings = schedule[: len(settings.collection.worker_gpus)]
        self.workers = [
            ActorWorker(
                settings,
                gpu,
                factory,
                self.owner,
                self.instance_root / "workers" / str(i),
                None if bindings is None else bindings[i],
            )
            for i, gpu in enumerate(settings.collection.worker_gpus)
        ]
        self.planner = (
            HTTPPlanningClient(settings.collection.planning_urls)
            if settings.collection.planning_urls else
            LocalPlanningClient(settings, store, self.instance_root / "planner")
        )
        self.planning = deque()
        self.audit_queue = asyncio.Queue(maxsize=settings.collection.capacity)
        self.audit_executor = ThreadPoolExecutor(
            max_workers=settings.collection.audit_workers, thread_name_prefix="collection-audit")
        self.active = {}
        self.disk = AsyncDisk()
        self.observations, self.executions = deque(), deque()
        self.condition = asyncio.Condition()
        self.closed = False
        self.runners = [asyncio.create_task(self._consume(w)) for w in self.workers]
        self.planning_runners = [asyncio.create_task(self._consume_planning())
                                for _ in range(settings.collection.planning_workers)]
        self.audit_runners = [asyncio.create_task(self._consume_audit())
                             for _ in range(settings.collection.audit_workers)]
        self.peak_active = 0
        self.scene_schedule = []
        for i, scene in enumerate(schedule or []):
            self.scene_schedule.append(
                {
                    "key": collection_key(scene.model_dump()),
                    "attempts": scene.attempts,
                    "claimed": 0,
                    "terminal": 0,
                    "successful": 0,
                    "success_goal": scene.success_goal,
                    "status": "active" if i < len(self.workers) else "queued",
                    "worker": self.workers[i].instance if i < len(self.workers) else None,
                }
            )
        self.scene_budgets = {s["key"]: s for s in self.scene_schedule}

    async def _advance_scenes(self, finishing=None):
        if self.closed:
            return
        for scene in self.scene_schedule:
            if scene["status"] != "active":
                continue
            worker = next(w for w in self.workers if w.instance == scene["worker"])
            if not worker.alive:
                scene["status"] = "unavailable"
                continue
            reached_goal = (
                scene["success_goal"] is not None and scene["successful"] >= scene["success_goal"]
            )
            exhausted = scene["claimed"] == scene["attempts"]
            if (
                not (reached_goal or exhausted)
                or scene["terminal"] != scene["claimed"]
                or worker.current is not None
            ):
                continue
            if any(cid != finishing and collection_key(s["request"]) == scene["key"]
                   for cid, s in self.active.items()):
                continue
            following = next((s for s in self.scene_schedule if s["status"] == "queued"), None)
            if following is not None:
                await worker.rebind(following["key"], self.disk)
            scene["status"] = "complete"
            scene["completion_reason"] = (
                "success_goal_reached" if reached_goal else "attempt_budget_exhausted"
            )
            if following is not None:
                following.update(status="active", worker=worker.instance)

    def _folder(self, cid):
        if len(cid) != 64 or any(c not in "0123456789abcdef" for c in cid):
            raise WorkerError(422, "invalid collection identity")
        return self.root / "tasks" / cid

    async def status(self, cid):
        if cid in self.active:
            return self._view(self.active[cid])
        folder = self._folder(cid)
        return await self.disk.run(self._stored_status, cid, folder)

    @staticmethod
    def _stored_status(cid, folder):
        if not folder.exists():
            raise WorkerError(404, "unknown collection task")
        if (folder / "receipt.json").exists():
            return read_json(folder / "receipt.json")
        return {
            "request_id": cid,
            "status": "failed_or_uncertain",
            "output": str(folder),
            "error": "unsettled task from a prior service lifetime; never automatically resumed",
            "automatic_retry": False,
        }

    def _view(self, state):
        result = {
            "request_id": state["request"]["request_id"],
            "status": state["status"],
            "output": str(state["folder"]),
            "service_owner": self.owner,
            "automatic_retry": False,
        }
        if "public" in state:
            result["public"] = state["public"]
        return result

    async def _persist(self, state):
        await self.disk.run(replace_json, state["folder"] / "status.json", self._view(state))

    async def _finish(self, state, result):
        cid = state["request"]["request_id"]
        receipt = dict(
            self._view(state),
            **result,
            training_admission="pending_review",
            cancel_requested=state.get("cancel_requested", False),
        )
        await self.disk.run(create_json, state["folder"] / "receipt.json", receipt)
        self.observations = deque(i for i in self.observations if i != cid)
        self.executions = deque(i for i in self.executions if i != cid)
        self.planning = deque(i for i in self.planning if i != cid)
        if self.scene_schedule:
            budget = self.scene_budgets[collection_key(state["request"])]
            budget["terminal"] += 1
            budget["successful"] += int(result.get("status") == "completed_audit")
            await self._advance_scenes(finishing=cid)
        self.active.pop(cid, None)
        return receipt

    async def submit(self, request):
        value = request.model_dump(mode="json")
        cid, folder = value["request_id"], self._folder(value["request_id"])
        async with self.condition:
            if await self.disk.run(folder.exists):
                try:
                    old = await self.disk.run(read_json, folder / "request.json")
                except (OSError, ValueError) as error:
                    raise WorkerError(
                        409, "partial collection claim; manual inspection required"
                    ) from error
                if old != value:
                    raise WorkerError(409, "request_id already used with different task")
                return await self.status(cid)
            if self.closed or not any(w.alive for w in self.workers):
                raise WorkerError(503, "collection service has no live workers")
            planning_active = sum(s["status"] in {"planning", "queued_planning"}
                                  for s in self.active.values())
            if planning_active >= self.settings.collection.planning_capacity:
                raise WorkerError(429, "planning backpressure; no collection claim created")
            if len(self.active) >= self.settings.collection.capacity:
                raise WorkerError(429, "collection prefetch capacity exhausted")
            if (await self.disk.run(shutil.disk_usage, self.root)).free < self.settings.collection.minimum_free_disk_bytes:
                raise WorkerError(507, "collection disk reserve reached")
            if (await self.disk.run(os.statvfs, self.root)).f_favail < self.settings.collection.minimum_free_inodes:
                raise WorkerError(507, "collection inode reserve reached; no claim created")
            if value["dataset"] not in self.settings.collection.training_datasets:
                raise WorkerError(422, "only explicitly configured training datasets may be collected")
            velocity = self.settings.simulation.datasets[value["dataset"]].physics.velocity
            if velocity is None or abs(velocity.time_step_s - 0.25) > 1e-9:
                raise WorkerError(422, "collection requires registered 4Hz velocity physics")
            if value["camera"] not in self.settings.cameras:
                raise WorkerError(404, "unknown camera profile")
            if value["scene_id"] not in self.store.scene_hashes.get(value["dataset"], {}):
                raise WorkerError(422, "scene is not in the pinned training dataset")
            if self.scene_schedule:
                budget = self.scene_budgets.get(collection_key(value))
                if budget is None:
                    raise WorkerError(422, "scene is outside the frozen collection schedule")
                if budget["status"] != "active" or budget["claimed"] >= budget["attempts"]:
                    raise WorkerError(409, "scene is not admitting tasks; no claim created")
                if (
                    budget["success_goal"] is not None
                    and budget["successful"] + budget["claimed"] - budget["terminal"]
                    >= budget["success_goal"]
                ):
                    raise WorkerError(
                        409, "scene success goal is covered by active claims; no claim created"
                    )
            if self.settings.collection.worker_scenes is not None and not any(
                self._supports(w, value) for w in self.workers
            ):
                raise WorkerError(422, "scene/camera is outside this fixed scene window")
            if not any(w.alive and self._supports(w, value) for w in self.workers):
                raise WorkerError(503, "no live worker for this scene; no automatic replay")
            state = {"request": value, "folder": folder, "status": "claiming"}
            self.active[cid] = state
            try:
                await self.disk.run(lambda: folder.mkdir(parents=True))
                await self.disk.run(create_json, folder / "request.json", value)
                await self.disk.run(create_json,
                    folder / "lease.json",
                    {
                        "owner": value["owner"],
                        "service_owner": self.owner,
                        "request_id": cid,
                        "started_unix": time.time(),
                        "automatic_retry": False,
                    },
                )
            except BaseException:
                self.active.pop(cid, None)
                raise
            needs_plan = value["kind"] != "short_instructions"
            state["status"] = "queued_planning" if needs_plan else "queued_observation"
            if self.scene_schedule:
                self.scene_budgets[collection_key(value)]["claimed"] += 1
            await self._persist(state)
            (self.planning if needs_plan else self.observations).append(cid)
            self.peak_active = max(self.peak_active, len(self.active))
            self.condition.notify_all()
            return self._view(state)

    async def target(self, cid, target):
        async with self.condition:
            folder = self._folder(cid)
            path = folder / "target.json"
            value = target.model_dump(mode="json")
            if await self.disk.run(path.exists):
                if await self.disk.run(read_json, path) != value:
                    raise WorkerError(409, "collection target is already frozen")
                return await self.status(cid)
            state = self.active.get(cid)
            if (
                state is None
                or state["status"] != "observation_ready"
                or state["request"]["kind"] != "short_instructions"
            ):
                raise WorkerError(409, "task is not awaiting a free-region target")
            await self.disk.run(create_json, path, value)
            state.update(status="queued_planning", target=value)
            await self._persist(state)
            self.planning.append(cid)
            self.condition.notify_all()
            return self._view(state)

    async def cancel(self, cid):
        async with self.condition:
            state = self.active.get(cid)
            if state is None:
                return await self.status(cid)
            state["cancel_requested"] = True
            if state["status"] in {"planning", "auditing", "queued_audit"}:
                return dict(self._view(state), cancel_requested=True)
            running = next((w for w in self.workers if w.current == cid or (
                state["status"] in {"observing", "executing"}
                and w.instance == state.get("assigned_worker"))), None)
            if running is not None:
                running.stop.set()
                return dict(self._view(state), cancel_requested=True)
            return await self._finish(state, {"status": "cancelled_before_movement"})

    async def _next(self, worker):
        async with self.condition:
            while not self.closed:
                for pending, phase in ((self.executions, "execute"), (self.observations, "observe")):
                    eligible = [
                        cid for cid in pending if self._supports(worker, self.active[cid]["request"])
                    ]
                    if not eligible:
                        continue
                    selected = next(
                        (
                            cid
                            for cid in eligible
                            if self._key(self.active[cid]["request"]) in worker.warm
                        ),
                        None,
                    )
                    if selected is None:
                        # Let an idle worker with the scene already loaded claim
                        # this task before another worker evicts its own scene.
                        selected = next(
                            (
                                cid
                                for cid in eligible
                                if not any(
                                    other is not worker
                                    and other.current is None
                                    and other.alive
                                    and self._key(self.active[cid]["request"]) in other.warm
                                    for other in self.workers
                                )
                            ),
                            None,
                        )
                    if selected is None:
                        continue
                    pending.remove(selected)
                    state = self.active[selected]
                    state["status"] = "executing" if phase == "execute" else "observing"
                    state["assigned_worker"] = worker.instance
                    await self._persist(state)
                    # A cached worker becoming busy lets cold workers take more
                    # tasks for that scene when parallelism is useful.
                    self.condition.notify_all()
                    return phase, state
                await self.condition.wait()
            return None

    @staticmethod
    def _key(request):
        return collection_key(request)

    @staticmethod
    def _supports(worker, request):
        binding = getattr(worker, "binding", None)
        return binding is None or binding == collection_key(request)

    async def _consume(self, worker):
        while not self.closed and worker.alive:
            work = await self._next(worker)
            if work is None:
                return
            phase, state = work
            try:
                result = await worker.call(phase, state["request"], state["folder"], state.get("target"))
                key = self._key(state["request"])
                if key in worker.warm:
                    worker.warm.remove(key)
                worker.warm.append(key)
                async with self.condition:
                    if phase == "observe":
                        if state["request"]["kind"] == "reference_geometry":
                            await self._finish(state, dict(result, status="reference_complete"))
                            self.condition.notify_all()
                            continue
                        state["public"] = result["public"]
                        await self.disk.run(create_json, state["folder"] / "observation-ready.json", result)
                        if state.get("cancel_requested"):
                            await self._finish(state, {"status": "cancelled_before_movement"})
                        elif state["request"]["kind"] == "r2r_chunks":
                            state["status"] = "queued_execution"
                            self.executions.append(state["request"]["request_id"])
                        else:
                            state["status"] = "observation_ready"
                        if state["request"]["request_id"] in self.active:
                            await self._persist(state)
                    else:
                        state.update(status="queued_audit", execution_result=result)
                        await self._persist(state)
                        self.audit_queue.put_nowait(state["request"]["request_id"])
            except Exception as error:
                async with self.condition:
                    await self._finish(
                        state,
                        {
                            "status": "excluded_before_movement"
                            if isinstance(error, WorkerError) and error.status == 422
                            else "failed_or_uncertain",
                            "error": str(error),
                            "phase": phase,
                        },
                    )
            async with self.condition:
                for pending in list(self.active.values()):
                    if pending["status"] not in {"planning", "auditing", "queued_audit"} and not any(
                        w.alive and self._supports(w, pending["request"])
                        for w in self.workers
                    ):
                        await self._finish(
                            pending,
                            {
                                "status": "failed_or_uncertain",
                                "error": "no live worker for scene; evidence retained, no replay",
                            },
                        )
                await self._advance_scenes()
                self.condition.notify_all()

    async def _consume_planning(self):
        while not self.closed:
            async with self.condition:
                await self.condition.wait_for(lambda: self.closed or bool(self.planning))
                if self.closed:
                    return
                cid = self.planning.popleft()
                state = self.active.get(cid)
                if state is None:
                    continue
                state["status"] = "planning"
                await self._persist(state)
            try:
                job = await asyncio.to_thread(build_job, self.settings, state["request"],
                                              state["folder"], state.get("target"))
                state["planning_job"] = job
                self.planner.prepare(job, state["folder"])
                await self.disk.run(create_json, state["folder"] / "planning-claim.json", {
                    "attempt_id": job.attempt_id, "input_digest": job.digest(),
                    "automatic_replay": False,
                })
                result = await wait_for_plan(self.planner, job,
                                             lambda: state.get("cancel_requested", False))
                await self.disk.run(install_result, state["folder"], job, result)
                async with self.condition:
                    if state.get("cancel_requested"):
                        await self._finish(state, {"status": "cancelled_before_movement",
                                             "planning_outcome": result["status"]})
                    elif result["status"] != "feasible":
                        status = {"infeasible": "excluded_before_movement",
                                  "budget_exhausted": "planning_budget_exhausted",
                                  "cancelled": "cancelled_before_movement"}.get(
                                      result["status"], "failed_or_uncertain")
                        await self._finish(state, {"status": status, "planning_outcome": result["status"],
                                             "movement_attempted": False, "error": result.get("error")})
                    elif state["request"]["kind"] == "reference_geometry":
                        await self._finish(state, {"status": "reference_complete",
                                             "reference_legs": result["reference_legs"],
                                             "source": "navmesh_geometry_not_executed_states"})
                    elif not any(w.alive and self._supports(w, state["request"])
                                 for w in self.workers):
                        await self._finish(state, {"status": "failed_or_uncertain", "phase": "planning",
                                             "movement_attempted": False,
                                             "error": "plan retained; no live simulation worker, no replay"})
                    else:
                        r2r = state["request"]["kind"] == "r2r_chunks"
                        state["status"] = "queued_observation" if r2r else "queued_execution"
                        (self.observations if r2r else self.executions).append(cid)
                        await self._persist(state)
                    self.condition.notify_all()
            except Exception as error:
                async with self.condition:
                    if cid in self.active:
                        await self._finish(state, {"status": "failed_or_uncertain", "phase": "planning",
                                             "movement_attempted": False, "error": repr(error)})
                    self.condition.notify_all()

    async def _consume_audit(self):
        while not self.closed:
            cid = await self.audit_queue.get()
            async with self.condition:
                state = self.active.get(cid)
                if state is None:
                    continue
                state["status"] = "auditing"
                await self._persist(state)
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    self.audit_executor, audit_recording, state["request"], state["folder"],
                    state.get("target"), state["execution_result"])
            except Exception as error:
                result = {"status": "failed_or_uncertain", "phase": "audit", "error": repr(error)}
            async with self.condition:
                await self._finish(state, result)
                self.condition.notify_all()

    def health(self):
        live = sum(w.alive for w in self.workers)
        return {
            "status": "ok" if live == len(self.workers) else "degraded",
            "service_owner": self.owner,
            "evidence_io": self.disk.health(),
            "live_workers": live,
            "workers": len(self.workers),
            "capacity": min(self.settings.collection.capacity, self.settings.collection.planning_capacity),
            "scene_cache_entries_per_worker": self.settings.collection.scene_cache_entries,
            "maximum_cached_scene_instances": len(self.workers)
            * self.settings.collection.scene_cache_entries,
            "fixed_scene_window": self.settings.collection.worker_scenes is not None,
            "rolling_scene_window": bool(self.scene_schedule),
            "scene_schedule": self.scene_schedule,
            "worker_state": [
                {
                    "instance": w.instance,
                    "pid": w.process.pid,
                    "gpu": w.gpu,
                    "scene_binding": w.binding,
                    "binding_generation": w.binding_generation,
                    "current": w.current,
                    "phase": w.current_phase,
                    "operations": dict(w.operation_counts),
                    "busy_s": dict(w.busy_s),
                    "cache": w.last_cache,
                    "alive": w.alive,
                }
                for w in self.workers
            ],
            "active": len(self.active),
            "peak_active": self.peak_active,
            "queued_observations": len(self.observations),
            "queued_executions": len(self.executions),
            "queued_planning": len(self.planning),
            "queued_audits": self.audit_queue.qsize(),
            "planner": self.planner.health(),
            "root": str(self.root),
            "training_datasets": self.settings.collection.training_datasets,
            "cameras": {name: camera.model_dump() for name, camera in self.settings.cameras.items()},
            "physics": {
                name: self.settings.simulation.datasets[name].physics.model_dump()
                for name in self.settings.collection.training_datasets
            },
            "geometry_profiles": {
                kind: geometry_config(self.settings, {"seed": 0, "kind": kind})
                for kind in ("short_instructions", "r2r_chunks")
            },
            "implementation_sha256": self.source_sha256,
            "dataset_identity": self.store.scene_hashes,
        }

    async def close(self):
        async with self.condition:
            self.closed = True
            self.condition.notify_all()
        for runner in self.planning_runners:
            runner.cancel()
        await asyncio.gather(*self.planning_runners, return_exceptions=True)
        await self.planner.close()
        for worker in self.workers:
            worker.stop.set()
        await asyncio.gather(*(asyncio.to_thread(w.close) for w in self.workers))
        for runner in self.runners:
            runner.cancel()
        await asyncio.gather(*self.runners, return_exceptions=True)
        # Finish in-flight audits before sealing receipts; no background evidence writes.
        for runner in self.audit_runners:
            runner.cancel()
        await asyncio.gather(*self.audit_runners, return_exceptions=True)
        await asyncio.to_thread(self.audit_executor.shutdown, wait=True, cancel_futures=True)
        for state in list(self.active.values()):
            await self._finish(
                state,
                {"status": "failed_or_uncertain", "error": "collection service stopped; no replay"},
            )
        await self.disk.close()


def create_app(settings, *, store=None, actor_factory=CollectionActor):
    store = store or EpisodeStore(settings)

    async def open_service():
        return CollectionService(settings, store, actor_factory)

    app = service_app("Privileged training collection service", "collection", open_service)

    @app.get("/v1/health")
    async def health():
        return app.state.collection.health()

    @app.post("/v1/collections")
    async def submit(request: TaskRequest):
        return await app.state.collection.submit(request)

    @app.post("/v1/collections/status")
    async def statuses(request: StatusRequest):
        return await asyncio.gather(*(app.state.collection.status(cid) for cid in request.request_ids))

    @app.get("/v1/collections/{cid}")
    async def status(cid: str):
        return await app.state.collection.status(cid)

    @app.post("/v1/collections/{cid}/target")
    async def target(cid: str, request: RegionRequest):
        return await app.state.collection.target(cid, request)

    @app.delete("/v1/collections/{cid}")
    async def cancel(cid: str):
        return await app.state.collection.cancel(cid)

    return app
