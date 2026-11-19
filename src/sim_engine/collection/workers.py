"""Collection worker processes: one actor per GPU owning its warm scenes.

Workers perform whole observation/execution operations and are never auto-restarted.
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from pathlib import Path

from sim_engine.core.io import create_json
from sim_engine.core.process import ProcessWorker, WorkerError


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
