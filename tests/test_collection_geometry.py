"""Simulator-side regression checks relocated from nav-train; no GPU."""

import hashlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from sim_engine.collection.region import preview_target
from sim_engine.core.io import create_json, file_sha256, read_json


def test_scene_cache_reuses_entries_and_closes_only_the_lru_on_eviction(tmp_path, monkeypatch):
    from helpers import settings

    from sim_engine.collection import backend as module
    from sim_engine.config import CollectionSettings, Dataset

    config = settings(tmp_path)
    config.simulation.datasets["train_pilot"] = Dataset(episodes=tmp_path / "train.json")
    config.collection = CollectionSettings(
        root=tmp_path / "records", training_datasets=["train_pilot"], scene_cache_entries=4
    )
    closed = []
    created = []

    def construct(worker, manifest, scene):
        instance = SimpleNamespace(
            scene=scene,
            close=lambda: closed.append(scene),
            references=SimpleNamespace(values={}),
            clear_task=lambda: None,
        )
        created.append(instance)
        return instance

    monkeypatch.setattr(module, "SceneBackend", construct)
    monkeypatch.setattr(module, "scene_path", lambda root, name: tmp_path / name)
    actor = module.CollectionActor(config, 0)

    def request(scene, cid):
        return dict(
            dataset="train_pilot",
            scene_id=scene,
            camera="trained_ce30",
            seed=1,
            kind="short_instructions",
            request_id=cid,
            instruction="",
        )

    for scene in "abcd":
        actor._backend(request(scene, scene))
    first = created[0]
    first.references.values["old-task"] = object()
    assert actor._backend(request("a", "new-task")) is first
    assert first.references.values == {}
    assert list(first.tasks) == ["new-task"]
    actor._backend(request("e", "e"))
    assert closed == [str(tmp_path / "b")]
    assert actor.stats() == {"scene_loads": 5, "cache_hits": 1, "cached_scenes": 4}
    actor.close()
    assert len(closed) == len(set(closed)) == 5


def test_production_observe_captures_rgb_depth_once_per_view_and_restores_heading(tmp_path, monkeypatch):
    from sim_engine.collection.simulation import SceneBackend

    rotations = []
    renders = []

    def observe():
        renders.append(len(renders) + 1)
        return {
            "rgb": np.full((3, 4, 3), renders[-1], dtype=np.uint8),
            "depth": np.full((3, 4), renders[-1], dtype=np.float32),
        }

    monkeypatch.setitem(
        sys.modules,
        "habitat_sim.utils.common",
        SimpleNamespace(quat_from_coeffs=lambda q: q, quat_to_coeffs=lambda q: np.asarray(q)),
    )
    backend = SceneBackend.__new__(SceneBackend)
    from sim_engine.backends.contracts import Capture

    def captured():
        obs = observe()
        return Capture(obs, [0, 0, 0], [0, 0, 0, 1],
                       {"depth": {"position": [0, 1, 0], "rotation_xyzw": [0, 0, 0, 1]}},
                       {"projections": {"rgb": np.eye(4).tolist(), "depth": np.eye(4).tolist()}})

    snapshots = {}
    def views(yaw):
        snapshots.update({key: captured() for key in ("front", "right", "back", "left")})
        return snapshots

    backend.env = SimpleNamespace(observe_views=views, capture=lambda: snapshots["front"])
    backend.camera = SimpleNamespace(model_dump=lambda: {"width": 4, "height": 3})
    backend.np = np
    backend.depth_near_far = (0.1, 100.0)
    backend.scene = "room"
    backend.loads = 1
    backend.resets = 2
    backend.tasks = {"t": {"instruction": ""}}
    backend.reset_candidate = lambda ep: None
    episode = {"start_rotation": [0, 0, 0, 1]}
    result = backend.observe(
        {"task_id": "t"}, {"episode": episode, "yaw": 0.0}, tmp_path, capture_depth=True
    )
    assert len(renders) == 4 and not rotations
    views = read_json(tmp_path / "view-sensors.json")
    for i, view in enumerate(("front", "right", "back", "left"), 1):
        with np.load(tmp_path / (view + "-depth.npz")) as data:
            assert np.all(data["depth_m"] == i)
        assert np.all(np.asarray(Image.open(tmp_path / (view + ".png"))) == i)
        assert views[view]["depth_sha256"] == file_sha256(tmp_path / (view + "-depth.npz"))
    assert set(result) == {"instruction", "images"}


def test_preview_keeps_the_same_path_scaled_budget():
    from sim_engine.collection import region

    calls = []

    def fake_preview(pf, episode, path, profile, max_steps):
        calls.append(max_steps)
        return {"termination": "expert_stop", "collision_steps": 0, "rows": []}

    previous = region.preview
    region.preview = fake_preview
    try:
        backend = SimpleNamespace(
            pf=object(), expert_profile=SimpleNamespace(dt_s=0.25, v_max_mps=0.6, omega_max_rps=1.26)
        )
        result = preview_target(
            backend,
            {"backend_manifest": {"config": {"max_steps": 160}}},
            {"start_rotation": [0, 0, 0, 1]},
            [0, 0, -30],
            [[0, 0, 0], [0, 0, -30]],
        )
        assert calls == [result["execution_budget"]["max_steps"]] and calls[0] > 160
    finally:
        region.preview = previous


@pytest.mark.parametrize("corrupt_front", [False, True])
def test_executor_restores_prefetched_start_and_persists_intended_goal_before_movement(
    tmp_path, monkeypatch, corrupt_front
):
    from sim_engine.collection.backend import CollectionActor

    goal = [0.0, 0.0, -1.0]
    prepared = {
        "episode": {
            "start_position": [0.0, 0.0, 0.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "goals": [{"position": [0.0, 0.0, 0.0], "radius": 0.08}],
        },
        "yaw": 0.0,
    }
    create_json(tmp_path / "start-state.json", prepared)
    create_json(
        tmp_path / "camera.json",
        {"sensors": {"depth": {"intrinsics_K": [[5, 0, 20], [0, 5, 20], [0, 0, 1]]}}},
    )
    create_json(
        tmp_path / "view-sensors.json",
        {"front": {"position": [0.0, 1.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}},
    )
    np.savez_compressed(tmp_path / "front-depth.npz", depth_m=np.ones((40, 40), dtype=np.float32))
    Image.new("RGB", (40, 40)).save(tmp_path / "front.png")
    restored = []

    def reset(episode):
        restored.append(episode["start_position"])

    def rollout(candidate, state, folder, stop):
        assert restored == [[0.0, 0.0, 0.0]]
        assert read_json(folder / "selected-goal.json")["position"] == goal
        assert actor_backend.config["max_steps"] == 170
        return {
            "passed": True,
            "termination": "expert_stop",
            "collision_steps": 0,
            "final_distance_m": 0.02,
        }

    actor_backend = SimpleNamespace(
        config={"goal_tolerance_m": 0.12},
        shortest=lambda a, b: [a, b],
        reset_candidate=reset,
        rollout=rollout,
        env=SimpleNamespace(
            sim=SimpleNamespace(
                get_sensor_observations=lambda: {"rgb": np.zeros((40, 40, 3), dtype=np.uint8)}
            ),
            position=lambda: [0.0, 0.0, -0.98],
        ),
        camera=SimpleNamespace(width=40, height=40),
    )
    actor = CollectionActor.__new__(CollectionActor)
    actor._backend = lambda request: actor_backend
    actor.loads = 1
    actor.hits = 1
    actor.cache = {"scene": actor_backend}

    prepared.update(path=[[0, 0, 0], goal], requested_goal_position=goal,
                    goal_adjustment_m=0, endpoint_review_required=False,
                    execution_budget={"max_steps": 170})
    create_json(tmp_path / "private-geometry.json", prepared)
    create_json(tmp_path / "planning-result.json", {"result_digest": "f" * 64})
    create_json(tmp_path / "execution-lease.json", {
        "identity": "a" * 32, "task_id": "a" * 64,
        "plan_digest": "f" * 64, "prepared_sha256": file_sha256(tmp_path / "private-geometry.json"),
        "runtime": {"habitat_sim": "test", "numpy": np.__version__},
        "navmesh_sha256": hashlib.sha256(b"test").hexdigest(),
    })
    actor_backend.hs = SimpleNamespace(__version__="test")
    actor_backend.env.capture = lambda: SimpleNamespace(images={"rgb": np.ones((40, 40, 3), dtype=np.uint8)})
    def observe_views(yaw):
        assert yaw == prepared["yaw"]
        return {"front": SimpleNamespace(images={"rgb": np.full(
            (40, 40, 3), int(corrupt_front), dtype=np.uint8)})}
    actor_backend.env.observe_views = observe_views
    actor_backend.scene_asset = SimpleNamespace(navmesh=tmp_path / "test.navmesh")
    actor_backend.np = np
    actor_backend.scene = str(tmp_path / "test.glb")
    (tmp_path / "test.navmesh").write_bytes(b"test")
    if corrupt_front:
        with pytest.raises(AssertionError):
            actor.execute({"request_id": "a" * 64, "kind": "short_instructions"}, tmp_path,
                          {"target_view": "front", "bbox": [350, 500, 650, 800]},
                          SimpleNamespace(is_set=lambda: False))
        assert not (tmp_path / "movement-attempted.json").exists()
        return
    result = actor.execute(
        {"request_id": "a" * 64, "kind": "short_instructions"},
        tmp_path,
        {"target_view": "front", "bbox": [350, 500, 650, 800]},
        SimpleNamespace(is_set=lambda: False),
    )
    assert result["status"] == "geometry_complete"
    assert read_json(tmp_path / "selected-goal.json")["source"] == "planned_before_movement"


def test_native_environment_reuses_only_compatible_scene_camera_and_physics(tmp_path, monkeypatch):
    from helpers import settings

    from sim_engine.config import Camera
    from sim_engine.habitat import HabitatEnvironment

    instances = []

    class Sim:
        def __init__(self, config):
            instances.append(self)
            self.resets = 0
            self.closed = False
            self.pathfinder = SimpleNamespace(is_loaded=True, load_nav_mesh=lambda path: True)

        def reset(self):
            self.resets += 1

        def seed(self, seed):
            self.last_seed = seed

        def initialize_agent(self, index, state):
            self.state = state

        def get_sensor_observations(self):
            return {"rgb": np.zeros((32, 32, 3), dtype=np.uint8)}

        def close(self):
            self.closed = True

    hs = SimpleNamespace(
        __version__="0.3.1",
        Simulator=Sim,
        SimulatorConfiguration=SimpleNamespace,
        CameraSensorSpec=SimpleNamespace,
        SensorType=SimpleNamespace(COLOR=0),
        agent=SimpleNamespace(
            AgentConfiguration=SimpleNamespace,
            ActionSpec=lambda *args: args,
            ActuationSpec=lambda **kw: kw,
        ),
        ActionSpec=lambda *args: args,
        ActuationSpec=lambda **kw: kw,
        Configuration=lambda *args: args,
        AgentState=SimpleNamespace,
    )
    monkeypatch.setitem(sys.modules, "habitat_sim", hs)
    monkeypatch.setitem(
        sys.modules, "habitat_sim.utils.common", SimpleNamespace(quat_from_coeffs=lambda q: q)
    )
    monkeypatch.setitem(sys.modules, "magnum", SimpleNamespace(Vector3=lambda *args: args))
    env = HabitatEnvironment(settings(tmp_path))
    camera = Camera(width=32, height=32)
    episode = {"scene_id": "room.glb", "start_position": [0, 0, 0], "start_rotation": [0, 0, 0, 1]}
    env.reset(episode, camera, 1)
    env.reset(dict(episode, start_position=[2, 0, 0]), camera, 2)
    assert len(instances) == 1 and instances[0].resets == 1 and instances[0].last_seed == 2
    assert instances[0].state.position.tolist() == [2, 0, 0]
    env.reset(episode, camera.model_copy(update={"hfov": 120}), 3)
    assert len(instances) == 2 and instances[0].closed
    env.reset(dict(episode, scene_id="other.glb"), camera, 4)
    assert len(instances) == 3 and instances[1].closed
    assert env.scene_loads == 3 and env.resets == 4
    env.close()
