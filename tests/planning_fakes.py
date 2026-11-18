"""Spawnable fault-injection engine: no Habitat, GPU, model or service ownership."""

import os
import time

from sim_engine.collection.storage import read, write
from sim_engine.planning.context import PlanningContext, checkpoint
from sim_engine.planning.contracts import Blob


class FakePlanningEngine:
    def __init__(self, settings):
        pass

    def run(self, job, folder, stop, *, progress=None):
        instruction = job.task.instruction
        if instruction == "planner-crash":
            os._exit(75)
        if instruction == "planner-hang":
            time.sleep(60)
        if instruction == "planner-slow":
            from sim_engine.planning.context import PlanningStopped
            context = PlanningContext(job.budget, stop, progress)
            try:
                with context.bind():
                    while True:
                        checkpoint(query=1)
                        time.sleep(0.01)
            except PlanningStopped as error:
                return {"outcome": error.outcome, "artifacts": {}}
        if job.task.kind == "reference_geometry":
            return {"outcome": "feasible", "artifacts": {},
                    "reference_legs": [{"distance_m": 2.0, "initial_yaw": 0.0}]}
        prepared = {
            "episode": {"start_position": list(job.task.start_position),
                        "start_rotation": list(job.task.start_rotation),
                        "scene_id": job.task.scene_id},
            "path": [list(job.task.start_position), [0, 0, -2]],
            "requested_goal_position": [0, 0, -2], "yaw": 0,
            "goal_adjustment_m": 0, "endpoint_review_required": False,
        }
        names = ["private-geometry.json"]
        write(folder / "private-geometry.json", prepared)
        if job.task.kind == "r2r_chunks":
            names.append("start-state.json")
            write(folder / "start-state.json", prepared)
        return {"outcome": "feasible", "prepared": prepared, "artifacts": {
            name: Blob.encode((folder / name).read_bytes(), name).model_dump(mode="json")
            for name in names}}


def fake_local_client(settings, store, root):
    from sim_engine.planning.client import LocalPlanningClient
    return LocalPlanningClient(settings, store, root, engine_factory=FakePlanningEngine)


def fake_audit(request, folder, target, result):
    assert read(folder / "fake-execution.json")
    return dict(result, status="completed_audit")
