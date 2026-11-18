"""Warm scene ownership and whole-task collection operations inside a worker."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import asdict

import numpy as np
from PIL import Image

from sim_engine.core.io import create_json, file_sha256, read_json
from sim_engine.data import scene_path
from sim_engine.navigation.clearance import ClearanceProfile
from sim_engine.planning.contracts import PlanningProfile
from sim_engine.scenes import resolve_scene

from .simulation import SceneBackend


def geometry_config(settings, request):

    return PlanningProfile(version=settings.collection.planning_profile_version,
                           base_steps=settings.collection.base_steps).geometry(request)


class CollectionActor:
    """Bounded LRU, owned by one EGL process; request changes never leak task state."""

    def __init__(self, settings, gpu):
        self.settings, self.gpu = settings, gpu
        self.cache = OrderedDict()
        self.loads = self.hits = 0

    def _backend(self, request):
        dataset = self.settings.simulation.datasets[request["dataset"]]
        root = dataset.scenes_root or self.settings.simulation.scenes_root
        scene = str(resolve_scene(self.settings, request["dataset"], request["scene_id"]).render
                    if dataset.scenes else scene_path(root, request["scene_id"]))
        key = (request["dataset"], scene, request["camera"])
        if key in self.cache:
            backend = self.cache.pop(key)
            self.hits += 1
        else:
            if len(self.cache) >= self.settings.collection.scene_cache_entries:
                _, expired = self.cache.popitem(last=False)
                expired.close()
            runtime_settings = self.settings.model_dump(mode="json")
            runtime_settings.pop("collection", None)
            runtime_settings.pop("cameras", None)
            runtime_settings.pop("policy_version", None)
            runtime_settings["policy"] = {"model": "/unused/no-policy-loaded"}
            runtime_settings["simulation"]["scenes_root"] = str(root)
            runtime_settings["simulation"]["datasets"]["train_pilot"] = dataset.model_dump(mode="json")
            runtime_settings["simulation"]["datasets"]["train_pilot"]["scenes_root"] = str(root)
            runtime = {
                "scene_id": request["scene_id"],
                "settings": runtime_settings,
                "camera": self.settings.cameras[request["camera"]].model_dump(),
                "clearance_profile": asdict(ClearanceProfile()),
            }
            manifest = {
                "config": geometry_config(self.settings, request),
                "runtime": runtime,
                "tasks": [],
                "snapshot_root": "sim_engine_collection",
            }
            backend = SceneBackend({"gpu": self.gpu}, manifest, scene)
            self.loads += 1
        self.cache[key] = backend
        backend.config = geometry_config(self.settings, request)
        backend.clear_task()
        backend.tasks = {request["request_id"]: {"instruction": request["instruction"]}}
        # Route caches must never alias a task ID from another accepted request.
        backend.references.values.clear()
        return backend

    def observe(self, request, folder, stop):
        backend = self._backend(request)
        cid = request["request_id"]
        candidate = {"id": cid, "task_id": cid, "scene": backend.scene}
        if request["kind"] == "r2r_chunks":
            prepared = read_json(folder / "start-state.json")
        else:
            start = np.asarray(request["start_position"])
            if not backend.pf.is_navigable(start) or backend.guard.clearance(start) < 0.05:
                raise ValueError("initial pose lacks minimum execution clearance")
            episode = {
                "episode_id": cid,
                "_dataset": "train_pilot",
                "scene_id": request["scene_id"],
                "start_position": request["start_position"],
                "start_rotation": request["start_rotation"],
                "goals": [{"position": request["start_position"], "radius": 0.08}],
            }
            prepared = {
                "episode": episode,
                "yaw": 2 * math.atan2(episode["start_rotation"][1], episode["start_rotation"][3]),
            }
        public = backend.observe(candidate, prepared, folder, capture_depth=True)
        if request["kind"] == "short_instructions":
            public = {"images": public["images"]}
        create_json(folder / "public-observation.json", public)
        if request["kind"] == "short_instructions":
            create_json(folder / "start-state.json", prepared)
        return {"public": public, "cache": self.stats()}

    def execute(self, request, folder, target, stop):
        backend = self._backend(request)
        prepared = read_json(folder / "private-geometry.json")
        lease = read_json(folder / "execution-lease.json")

        plan = read_json(folder / "planning-result.json")
        if (lease["task_id"] != request["request_id"]
                or lease["plan_digest"] != plan["result_digest"]
                or file_sha256(folder / "private-geometry.json") != lease["prepared_sha256"]
                or file_sha256(backend.scene_asset.navmesh) != lease["navmesh_sha256"]
                or (folder / "movement-attempted.json").exists()):
            raise RuntimeError("execution lease mismatch or already consumed; no replay")
        runtime = lease["runtime"]
        if (runtime.get("habitat_sim") != backend.hs.__version__
                or runtime.get("numpy") != backend.np.__version__):
            raise RuntimeError("planner and simulator Habitat versions differ")
        if "execution_budget" in prepared:
            backend.config["max_steps"] = prepared["execution_budget"]["max_steps"]
        # Persist the intended destination before any movement, independently of the stop.
        create_json(
            folder / "selected-goal.json",
            {
                "position": prepared["requested_goal_position"],
                "goal_tolerance_m": backend.config["goal_tolerance_m"],
                "source": "planned_before_movement",
            },
        )
        # Observation prefetch may have changed the live environment. Restore, then check it.
        backend.reset_candidate(prepared["episode"])
        # Compare the same front panorama sensor used for the saved observation.
        # The primary RGB sensor can differ after yaw wrapping, even at the same pose.
        current = backend.env.observe_views(prepared["yaw"])["front"].images["rgb"]
        np.testing.assert_array_equal(current, np.asarray(Image.open(folder / "front.png")))
        if stop.is_set():
            raise InterruptedError("cancelled before movement")
        create_json(folder / "movement-attempted.json", {
            "automatic_retry": False, "execution_lease": lease["identity"],
            "plan_digest": lease["plan_digest"],
        })
        geometry = backend.rollout({"id": request["request_id"]}, prepared, folder, stop)
        create_json(folder / "geometry-result.json", {"status": "geometry_complete", "geometry": geometry})
        return {"status": "geometry_complete", "geometry": geometry, "cache": self.stats()}

    def stats(self):
        return {"scene_loads": self.loads, "cache_hits": self.hits, "cached_scenes": len(self.cache)}

    def close(self):
        for backend in self.cache.values():
            backend.close()
        self.cache.clear()
