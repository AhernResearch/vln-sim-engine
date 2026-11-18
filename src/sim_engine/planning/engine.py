"""Task-level CPU planning. Input and output are portable, content-verified artifacts."""

import math
import zipfile
from collections import OrderedDict

import numpy as np

from sim_engine.collection.limits import distance_allowed
from sim_engine.collection.region import ground_region
from sim_engine.collection.storage import digest, read, write

from .context import PlanningContext, PlanningStopped, checkpoint
from .contracts import Blob
from .geometry import NavMeshContext


class PlanningEngine:
    def __init__(self, settings):
        self.settings = settings
        self.assets = {(a.dataset, a.scene_id): a for a in settings.assets}
        self.cache = OrderedDict()

    def context(self, job):
        asset = self.assets[(job.task.dataset, job.task.scene_id)]
        if asset.sha256 != job.navmesh_sha256 or digest(asset.path) != asset.sha256:
            raise RuntimeError("NavMesh identity changed")
        config = job.profile.geometry(job.task.model_dump(mode="json"))
        key = (asset.sha256, job.task.scene_id, job.camera.model_dump_json(), job.profile.model_dump_json())
        if key in self.cache:
            backend = self.cache.pop(key)
        else:
            backend = NavMeshContext(asset.path, job.task.scene_id, job.camera, config)
            if len(self.cache) >= self.settings.scene_cache_entries:
                self.cache.popitem(last=False)
        backend.clear_task()
        backend.config = config
        self.cache[key] = backend
        return backend

    def run(self, job, folder, stop, *, progress=None):
        context = PlanningContext(job.budget, stop, progress)
        backend = None
        inputs = set(job.artifacts)
        try:
            with context.bind():
                checkpoint(phase="loading")
                for name, blob in job.artifacts.items():
                    path = folder / name
                    with path.open("xb") as stream:
                        stream.write(blob.decode())
                backend = self.context(job)
                context.geometry_queries = backend.queries
                task = job.task.model_dump(mode="json")
                if task["kind"] == "reference_geometry":
                    legs = []
                    for a, b in zip(task["reference_guides"], task["reference_guides"][1:]):
                        route = [a, b] if math.dist(a, b) < 1e-8 else backend.shortest(a, b)
                        legs.append({
                            "distance_m": sum(math.dist(x, y) for x, y in zip(route, route[1:])),
                            "initial_yaw": next((math.atan2(-(y[0]-x[0]), -(y[2]-x[2]))
                                                 for x, y in zip(route, route[1:])
                                                 if math.hypot(y[0]-x[0], y[2]-x[2]) > 1e-8), None),
                        })
                    return self.result("feasible", context, folder, inputs, reference_legs=legs)
                if task["kind"] == "r2r_chunks":
                    cid = task["request_id"]
                    q = task["start_rotation"]
                    backend.tasks[cid] = dict(task, id=cid, scene=backend.scene,
                                              source_start_yaw=2 * math.atan2(q[1], q[3]))
                    candidate = dict(id=cid, task_id=cid, scene=backend.scene,
                                     **{k: task[k] for k in ["offset", "yaw_delta", "reference_fraction"]})
                    prepared = backend.prepare(candidate, folder)
                    write(folder / "start-state.json", prepared)
                else:
                    prepared = self.free_region(backend, job, folder)
                checkpoint(phase="complete")
                return self.result("feasible", context, folder, inputs, prepared=prepared)
        except PlanningStopped as error:
            return self.result(error.outcome, context, folder, inputs)
        except ValueError as error:
            return self.result("infeasible", context, folder, inputs, error=str(error))
        finally:
            if backend is not None:
                backend.clear_task()

    def free_region(self, backend, job, folder):
        prepared = read(folder / "start-state.json")
        episode = prepared["episode"]
        if (episode["start_position"] != list(job.task.start_position)
                or episode["start_rotation"] != list(job.task.start_rotation)
                or episode["scene_id"] != job.task.scene_id):
            raise ValueError("planning pose does not match observation")
        camera, views = read(folder / "camera.json"), read(folder / "view-sensors.json")
        target = job.target.model_dump(mode="json")
        path = folder / (target["target_view"] + "-depth.npz")
        if views[target["target_view"]]["depth_sha256"] != digest(path):
            raise ValueError("observed depth identity mismatch")
        with zipfile.ZipFile(path) as archive:
            if (archive.namelist() != ["depth_m.npy"]
                    or archive.infolist()[0].file_size > job.camera.width * job.camera.height * 8 + 4096):
                raise ValueError("depth archive exceeds calibrated shape")
        with np.load(path, allow_pickle=False) as data:
            depth = data["depth_m"]
            if depth.shape != (job.camera.height, job.camera.width):
                raise ValueError("depth shape differs from camera")
            grounded, goal, checks, plan, forecast = ground_region(
                backend, target, depth, camera, views,
                np.asarray(episode["start_position"]), folder, {
                    "backend_manifest": {"config": backend.config},
                    "region_grid_size": 13, "region_plan_candidates": 12,
                    "visible_floor_policy": {
                        "enabled": True, "horizontal_limit": 0.01, "vertical_limit": 0.2,
                        "mask_scope": "full" if job.profile.version == "collection-v2" else "target_bbox",
                        "slope_degrees": 25.0, "adjustment_support_tolerance_m": 0.025,
                    },
                }, episode,
            )
        route = backend.shortest(episode["start_position"], goal)
        if not distance_allowed(sum(math.dist(a, b) for a, b in zip(route, route[1:]))):
            raise ValueError("selected goal has no nonzero connected route")
        for name, value in [("grounded-target.json", grounded), ("grounding-check.json", checks),
                            ("reference-check.json", plan), ("preflight.json", forecast),
                            ("execution-budget.json", forecast["execution_budget"])]:
            write(folder / name, value)
        episode["goals"] = [{"position": goal.tolist(), "radius": 0.08}]
        prepared.update(path=plan["path"], requested_goal_position=goal.tolist(),
                        goal_adjustment_m=0.0, endpoint_review_required=False,
                        execution_budget=forecast["execution_budget"])
        write(folder / "private-geometry.json", prepared)
        return prepared

    def result(self, outcome, context, folder, inputs, **extra):
        if context.geometry_queries is not None:
            extra["geometry_queries"] = context.geometry_queries.stats()
        artifacts = {}
        for path in sorted(folder.rglob("*")):
            if path.is_file() and str(path.relative_to(folder)) not in inputs:
                name = str(path.relative_to(folder))
                artifacts[name] = Blob.encode(path.read_bytes(), name).model_dump(mode="json")
        return dict(outcome=outcome, progress={"phase": context.phase,
                    "queries": context.queries, "expansions": context.expansions},
                    artifacts=artifacts, **extra)
