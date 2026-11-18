import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import TestEnvironment, episode, request, settings

from sim_engine.backends.contracts import Capture
from sim_engine.backends.habitat import GaussianEnvironment
from sim_engine.config import Dataset
from sim_engine.data import EpisodeStore
from sim_engine.pool import SessionPool
from sim_engine.scenes import resolve_scene
from sim_engine.session import Session
from sim_engine.sim_server import create_app
from sim_engine.tasks import metrics


class RGBDEnvironment(TestEnvironment):
    def capture(self):
        return Capture({'rgb': self.image(), 'depth': np.full(
            (self.camera.height, self.camera.width), self.x + 1, dtype=np.float32)},
            self.position(), [0, 0, 0, 1], {'depth': {'position': self.position()}},
            {'depth_units': 'meters', 'depth_convention': 'axial_minus_camera_z'})


def test_intermediate_frames_and_states_are_causal(tmp_path):
    cfg = settings(tmp_path)
    env = RGBDEnvironment(cfg)
    session = Session.open('s', env, episode(), 'trained_ce30', cfg.cameras['trained_ce30'], 0, cfg)
    session.step(request())
    assert len(session.frames) >= 3
    for index, snapshot in enumerate(session.sensor_frames):
        assert snapshot.position == session.positions[index]
        assert snapshot.images['depth'][0, 0] == pytest.approx(1 + index * .25)


def test_http_depth_is_frozen_and_unknown_sensor_keeps_session(tmp_path):
    cfg = settings(tmp_path)
    store = SimpleNamespace(metadata={}, scene_hashes={}, get=lambda dataset, key: episode(key))
    app = create_app(cfg, store=store,
                     pool_factory=lambda cfg: SessionPool(cfg, factory=RGBDEnvironment))
    with TestClient(app) as client:
        sid = client.post('/v1/sessions', json={'dataset': 'test', 'episode_id': 'a'}).json()['session_id']
        url = f'/v1/sessions/{sid}/frames/0?sensor=depth'
        original = client.get(url)
        np.testing.assert_array_equal(np.load(io.BytesIO(original.content))['depth_m'], 1)
        assert client.post(f'/v1/sessions/{sid}/step', json=request()).status_code == 200
        assert client.get(url).content == original.content
        assert client.get(url.replace('sensor=depth', 'sensor=unknown')).status_code == 404
        assert client.get(f'/v1/sessions/{sid}/metrics').status_code == 200
        assert client.delete(f'/v1/sessions/{sid}').status_code == 200


def test_development_metrics_do_not_require_goals(tmp_path):
    result = metrics(TestEnvironment(settings(tmp_path)), {'_task': 'development'}, [], stopped=True)
    assert result['evaluation_available'] is False
    assert result['success'] is result['distance_to_goal'] is None


def test_gs_rejects_vanilla_or_cpu_runtime():
    for runtime in (SimpleNamespace(__version__='0.3.1', cuda_enabled=True),
                    SimpleNamespace(__version__='0.3.3', cuda_enabled=False),
                    SimpleNamespace(__version__='0.3.3', cuda_enabled=True,
                                    SimulatorConfiguration=SimpleNamespace)):
        with pytest.raises(RuntimeError):
            GaussianEnvironment.validate_runtime(runtime)


def test_goal_reward_validation(tmp_path):
    cfg = settings(tmp_path)
    (tmp_path / 'room.glb').write_bytes(b'scene')
    (tmp_path / 'room.navmesh').write_bytes(b'navmesh')
    ep = dict(episode(), scene_id='room.glb')
    data = tmp_path / 'episodes.json'
    data.write_text(json.dumps({'episodes': [ep]}))
    cfg.simulation.datasets['dev'] = Dataset(episodes=data, task='development')
    with pytest.raises(ValueError, match='scoring goals'):
        EpisodeStore(cfg)
    ep.pop('goals')
    data.write_text(json.dumps({'episodes': [ep]}))
    assert EpisodeStore(cfg).get('dev', 'a')['_task'] == 'development'
    cfg.simulation.reward_mode = 'progress_v1'
    with pytest.raises(ValueError, match='goal rewards'):
        EpisodeStore(cfg)
    cfg.simulation.backend = 'habitat_gs'
    with pytest.raises(ValueError, match='explicit frozen'):
        resolve_scene(cfg, 'dev', 'room.glb')


def test_depth_evidence_mismatch_cannot_be_cached_as_valid(tmp_path, monkeypatch):
    import hashlib
    import sys

    binary = tmp_path / 'bindings.so'
    binary.write_bytes(b'pinned binding')
    monkeypatch.setitem(sys.modules, 'habitat_sim._ext', SimpleNamespace(
        habitat_sim_bindings=SimpleNamespace(__file__=str(binary))))
    env = GaussianEnvironment.__new__(GaussianEnvironment)
    env.hs = SimpleNamespace(__version__='0.3.3')
    env.settings = settings(tmp_path)
    env.settings.simulation.backend = 'habitat_gs'
    report = tmp_path / 'validation.json'
    env.settings.simulation.depth_validation = report
    value = {'passed': True, 'bindings_sha256': 'wrong', 'depth_units': 'meters',
             'depth_convention': 'axial_minus_camera_z'}
    report.write_text(json.dumps(value))
    for _ in range(2):
        with pytest.raises(ValueError, match='does not match'):
            env.identity()
    value['bindings_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
    report.write_text(json.dumps(value))
    assert env.identity()['depth_convention'] == 'axial_minus_camera_z'
