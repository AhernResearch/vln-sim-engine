"""Native end-to-end probe of one configured runtime: environment, planning kernels, a short
collection rollout and the HTTP session API. It never calls a model or a running service."""

import io
import uuid
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from sim_engine.backends.habitat import HabitatEnvironment
from sim_engine.collection.backend import CollectionActor
from sim_engine.collection.recording import validate_recording
from sim_engine.config import CollectionSettings
from sim_engine.core.io import create_json
from sim_engine.data import EpisodeStore
from sim_engine.navigation.native import native_search
from sim_engine.navigation.queries import NavMeshQueries
from sim_engine.navigation.tracking import preview
from sim_engine.planning.contracts import PlanningProfile
from sim_engine.planning.geometry import NavMeshContext
from sim_engine.scenes import resolve_scene
from sim_engine.sim_server import create_app


class _NeverStop:
    @staticmethod
    def is_set():
        return False


def _environment(settings, dataset, episode, camera):
    """Rendering, depth, fixed-chassis panorama, warm reset and the native NavMesh kernels."""
    env = HabitatEnvironment(settings)
    try:
        rgb = env.reset(episode, settings.cameras[camera], 0)
        env.enable_depth()
        assert float(rgb.std()) > 1, "rendered RGB is blank"
        before = env.position()
        assert np.isfinite(env.capture().images["depth"]).all()
        views = env.observe_views(env.pose().yaw)
        assert len(views) == 4 and env.position() == before
        assert all(v.images["rgb"].shape == rgb.shape for v in views.values())
        env.reset(episode, settings.cameras[camera], 0)
        assert env.scene_loads == 1, "a compatible reset reloaded the scene"
        identity = env.identity()
    finally:
        env.close()
    start = episode["start_position"]
    geometry = NavMeshContext(resolve_scene(settings, dataset, episode["scene_id"]).navmesh,
                              episode["scene_id"], settings.cameras[camera],
                              PlanningProfile().geometry({"seed": 0, "kind": "short_instructions"}))
    assert geometry.pf.is_navigable(start) and geometry.shortest(start, start)
    queries = NavMeshQueries(geometry.pf)
    assert len(queries.snap_many([start])) == 1 and len(queries.clearance_many([start])) == 1
    native_search(geometry.planner, start, start, [start, start])
    return identity


def _collection(settings, dataset, episode, camera, output):
    """Observe, then execute and record a sampled short route with the shared expert."""
    settings = settings.model_copy(deep=True)
    settings.collection = CollectionSettings(root=output / "collection", training_datasets=[dataset])
    actor = CollectionActor(settings, settings.simulation.gpu)
    task = {"dataset": dataset, "scene_id": episode["scene_id"], "camera": camera,
            "request_id": "a" * 64, "instruction": "Follow the sampled NavMesh route.",
            "kind": "short_instructions", "seed": 0}
    folder = output / "collection-probe"
    folder.mkdir()
    # The collection actor registers every dataset under its frozen training name.
    probe_episode = dict(episode, _dataset="train_pilot")
    start = probe_episode["start_position"]
    prepared = {"episode": probe_episode, "yaw": 0.0}
    try:
        backend = actor._backend(task)
        backend.observe({"task_id": task["request_id"]}, prepared, folder, capture_depth=True)
        backend.pf.seed(0)
        for _ in range(100):
            goal = np.asarray(backend.pf.get_random_navigable_point()).tolist()
            try:
                path = backend.env.shortest_path(start, goal)
                distance = backend.env.distance(start, goal)
            except RuntimeError:
                continue
            if not .7 < distance < 2:
                continue
            result = preview(backend.pf, probe_episode, path, backend.expert_profile, 160)
            if result["termination"] == "expert_stop" and not result["collision_steps"]:
                break
        else:
            raise AssertionError("no supported short collection route")
        prepared.update(path=path, requested_goal_position=goal, goal_adjustment_m=0,
                        endpoint_review_required=True)
        execution = backend.rollout({"id": task["request_id"]}, prepared, folder, _NeverStop())
        assert execution["passed"], execution
        validate_recording(folder)
    finally:
        actor.close()
    return execution


def _http(settings, dataset, episode, camera):
    """A spawned EGL worker behind the real session handlers: depth, idempotent step, STOP."""

    with TestClient(create_app(settings)) as client:
        health = client.get("/v1/health").json()
        assert health["backend"] == settings.simulation.backend
        reply = client.post("/v1/sessions", json={"dataset": dataset, "camera": camera,
                                                  "episode_id": str(episode["episode_id"])})
        assert reply.status_code == 200, reply.text
        sid = reply.json()["session_id"]
        frame = f"/v1/sessions/{sid}/frames/0?sensor=depth"
        original = client.get(frame)
        assert original.status_code == 200, original.text
        depth = np.load(io.BytesIO(original.content))["depth_m"]
        assert (depth > 0).any()
        step = {"request_id": uuid.uuid4().hex, "expected_event_index": 0,
                "command": {"kind": "waypoints", "action_budget": 1,
                            "trajectory": [[i * .25, 0, 0] for i in range(1, 9)]}}
        moved = client.post(f"/v1/sessions/{sid}/step", json=step)
        assert moved.status_code == 200, moved.text
        assert client.post(f"/v1/sessions/{sid}/step", json=step).json() == moved.json()
        assert client.get(frame).content == original.content, "historical depth changed"
        stop = client.post(f"/v1/sessions/{sid}/finish", json={"reason": "policy_stop"})
        assert stop.status_code == 200, stop.text
        assert client.delete(f"/v1/sessions/{sid}").status_code == 200
        assert client.get("/v1/health").json()["active_sessions"] == 0
    return health, stop.json(), float((depth > 0).mean())


def probe_backend(settings, output: Path) -> dict:
    """Write ``receipt.json`` into a new directory; any failed check raises."""
    output.mkdir(parents=True, exist_ok=False)
    store = EpisodeStore(settings)
    dataset = next(iter(store.rows))
    episode = next(iter(store.rows[dataset].values()))
    camera = next(iter(settings.cameras))
    identity = _environment(settings, dataset, episode, camera)
    execution = _collection(settings, dataset, episode, camera, output)
    health, stop, valid_depth = _http(settings, dataset, episode, camera)
    report = {"passed": True, "runtime": identity, "health": health, "stop": stop,
              "depth_valid_fraction": valid_depth, "model_inference": False, "collection": execution}
    create_json(output / "receipt.json", report)
    return report
