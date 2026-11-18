"""Explicit native runtime probe; never calls models or existing services."""
import argparse
import io
import json
import uuid
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from sim_engine.backends import create_environment
from sim_engine.config import Settings
from sim_engine.data import EpisodeStore
from sim_engine.planning.contracts import PlanningProfile
from sim_engine.planning.geometry import NavMeshContext
from sim_engine.sim_server import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = Settings.read(args.config)
    store = EpisodeStore(settings)
    dataset = next(iter(store.rows))
    ep = next(iter(store.rows[dataset].values()))
    camera = next(iter(settings.cameras))
    env = create_environment(settings)
    try:
        rgb = env.reset(ep, settings.cameras[camera], 0)
        env.enable_depth()
        assert float(rgb.std()) > 1
        before = env.position()
        captured = env.capture()
        assert np.isfinite(captured.images['depth']).all()
        views = env.observe_views(env.pose().yaw)
        assert len(views) == 4 and env.position() == before
        assert all(v.images['rgb'].shape == rgb.shape for v in views.values())
        env.reset(ep, settings.cameras[camera], 0)
        assert env.scene_loads == 1
        identity = env.identity()
        from sim_engine.scenes import resolve_scene
        scene = resolve_scene(settings, dataset, ep['scene_id'])
        profile = PlanningProfile()
        geometry = NavMeshContext(scene.navmesh, ep['scene_id'], settings.cameras[camera],
                                  profile.geometry({'seed': 0, 'kind': 'short_instructions'}))
        assert geometry.pf.is_navigable(ep['start_position'])
        p = geometry.shortest(ep['start_position'], ep['start_position'])
        assert p
        # Exercise the required native query and A* kernels, not only PathFinder.
        from sim_engine.planning.queries import NavMeshQueries
        queries = NavMeshQueries(geometry.pf)
        assert len(queries.snap_many([ep['start_position']])) == 1
        assert len(queries.clearance_many([ep['start_position']])) == 1
        from sim_engine.planning.native import native_search
        native_search(geometry.planner, ep['start_position'], ep['start_position'],
                      [ep['start_position'], ep['start_position']])
    finally:
        env.close()
    # Exercise the collection adapter and real shared expert, with an explicit sampled goal.
    from sim_engine.collection.backend import CollectionActor
    from sim_engine.collection.tracking import preview
    from sim_engine.config import CollectionSettings
    settings.collection = CollectionSettings(root=args.output / 'collection', training_datasets=[dataset])
    actor = CollectionActor(settings, settings.simulation.gpu)
    task = {'dataset': dataset, 'scene_id': ep['scene_id'], 'camera': camera,
            'request_id': 'a' * 64, 'instruction': 'Follow the sampled NavMesh route.',
            'kind': 'short_instructions', 'seed': 0}
    backend = actor._backend(task)
    collection_folder = args.output / 'collection-probe'
    collection_folder.mkdir()
    collection_ep = dict(ep, _dataset='train_pilot')
    start = collection_ep['start_position']
    prepared = {'episode': collection_ep, 'yaw': 0.0}
    try:
        backend.observe({'task_id': task['request_id']}, prepared, collection_folder, capture_depth=True)
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
            result = preview(backend.pf, collection_ep, path, backend.expert_profile, 160)
            if result['termination'] == 'expert_stop' and not result['collision_steps']:
                break
        else:
            raise AssertionError('no supported short collection route')
        prepared.update(path=path, requested_goal_position=goal, goal_adjustment_m=0,
                        endpoint_review_required=True)
        execution = backend.rollout({'id': task['request_id']}, prepared, collection_folder,
                                    type('Stop', (), {'is_set': lambda self: False})())
        assert execution['passed'], execution
        from sim_engine.collection.recording import validate_recording
        validate_recording(collection_folder)
    finally:
        actor.close()
    # Separate spawned EGL worker through actual sim-engine HTTP handlers.
    with TestClient(create_app(settings)) as client:
        health = client.get('/v1/health').json()
        assert health['backend'] == settings.simulation.backend
        reply = client.post('/v1/sessions', json={'dataset': dataset,
                            'episode_id': str(ep['episode_id']), 'camera': camera})
        assert reply.status_code == 200, reply.text
        sid = reply.json()['session_id']
        original = client.get(f'/v1/sessions/{sid}/frames/0?sensor=depth')
        assert original.status_code == 200, original.text
        depth = np.load(io.BytesIO(original.content))['depth_m']
        assert (depth > 0).any()
        step = {'request_id': uuid.uuid4().hex, 'expected_event_index': 0,
                'trajectory': [[i * .25, 0, 0] for i in range(1, 9)], 'action_budget': 1}
        moved = client.post(f'/v1/sessions/{sid}/step', json=step)
        assert moved.status_code == 200, moved.text
        assert client.post(f'/v1/sessions/{sid}/step', json=step).json() == moved.json()
        assert client.get(f'/v1/sessions/{sid}/frames/0?sensor=depth').content == original.content
        stop = client.post(f'/v1/sessions/{sid}/finish', json={'reason': 'policy_stop'})
        assert stop.status_code == 200, stop.text
        assert client.delete(f'/v1/sessions/{sid}').status_code == 200
        assert client.get('/v1/health').json()['active_sessions'] == 0
    report = {'passed': True, 'runtime': identity, 'health': health, 'stop': stop.json(),
              'depth_valid_fraction': float((depth > 0).mean()), 'model_inference': False, 'collection': execution}
    (args.output / 'receipt.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
