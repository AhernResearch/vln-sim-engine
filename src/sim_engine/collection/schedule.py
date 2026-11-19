"""Frozen rolling scene window: per-scene attempt budgets drained in configured order.

The first ``len(workers)`` scenes are bound at startup; a drained scene hands its worker to
the next queued scene. Without a configured schedule every operation is a no-op.
"""

from __future__ import annotations

from sim_engine.core.process import WorkerError

from .workers import collection_key


class SceneSchedule:
    def __init__(self, scenes, workers):
        self.entries = [
            {
                "key": collection_key(scene.model_dump()),
                "attempts": scene.attempts,
                "claimed": 0,
                "terminal": 0,
                "successful": 0,
                "success_goal": scene.success_goal,
                "status": "active" if i < len(workers) else "queued",
                "worker": workers[i].instance if i < len(workers) else None,
            }
            for i, scene in enumerate(scenes or [])
        ]
        self.budgets = {entry["key"]: entry for entry in self.entries}

    def __bool__(self):
        return bool(self.entries)

    def admit(self, request):
        """Raise unless the scene may take one more claim right now."""
        if not self:
            return
        budget = self.budgets.get(collection_key(request))
        if budget is None:
            raise WorkerError(422, "scene is outside the frozen collection schedule")
        if budget["status"] != "active" or budget["claimed"] >= budget["attempts"]:
            raise WorkerError(409, "scene is not admitting tasks; no claim created")
        if (
            budget["success_goal"] is not None
            and budget["successful"] + budget["claimed"] - budget["terminal"]
            >= budget["success_goal"]
        ):
            raise WorkerError(409, "scene success goal is covered by active claims; no claim created")

    def claim(self, request):
        if self:
            self.budgets[collection_key(request)]["claimed"] += 1

    def settle(self, request, succeeded):
        budget = self.budgets[collection_key(request)]
        budget["terminal"] += 1
        budget["successful"] += int(succeeded)

    async def advance(self, workers, active, disk, finishing=None):
        """Complete drained scenes and rebind their idle workers to the next queued scene."""
        for scene in self.entries:
            if scene["status"] != "active":
                continue
            worker = next(w for w in workers if w.instance == scene["worker"])
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
                   for cid, s in active.items()):
                continue
            following = next((s for s in self.entries if s["status"] == "queued"), None)
            if following is not None:
                await worker.rebind(following["key"], disk)
            scene["status"] = "complete"
            scene["completion_reason"] = (
                "success_goal_reached" if reached_goal else "attempt_budget_exhausted"
            )
            if following is not None:
                following.update(status="active", worker=worker.instance)
