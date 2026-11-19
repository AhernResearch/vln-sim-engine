import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from helpers import TestEnvironment, episode, request, settings

from sim_engine.backends.contracts import Capture
from sim_engine.backends.depth import depth_contract
from sim_engine.backends.runtime import HABITAT, HABITAT_GS
from sim_engine.config import Dataset, Simulation
from sim_engine.data import EpisodeStore
from sim_engine.pool import SessionPool
from sim_engine.scenes import resolve_scene
from sim_engine.session import Session
from sim_engine.sim_server import create_app
from sim_engine.tasks import DevelopmentTask


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
    result = DevelopmentTask().metrics(TestEnvironment(settings(tmp_path)), {}, [], stopped=True)
    assert result['evaluation_available'] is False
    assert result['success'] is result['distance_to_goal'] is None


def test_gs_rejects_vanilla_or_cpu_runtime():
    for runtime in (SimpleNamespace(__version__='0.3.1', cuda_enabled=True),
                    SimpleNamespace(__version__='0.3.3', cuda_enabled=False),
                    SimpleNamespace(__version__='0.3.3', cuda_enabled=True,
                                    SimulatorConfiguration=SimpleNamespace)):
        with pytest.raises(RuntimeError):
            HABITAT_GS.validate(runtime)
    with pytest.raises(RuntimeError):
        HABITAT.validate(SimpleNamespace(__version__='0.3.3'))


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
    assert '_task' not in EpisodeStore(cfg).get('dev', 'a')
    cfg.simulation.reward_mode = 'progress_v1'
    with pytest.raises(ValueError, match='goal rewards'):
        EpisodeStore(cfg)
    cfg.simulation.backend = 'habitat_gs'
    with pytest.raises(ValueError, match='explicit frozen'):
        resolve_scene(cfg, 'dev', 'room.glb')


def test_depth_is_metric_only_for_mesh_or_a_matching_passed_report(tmp_path):
    assert depth_contract(HABITAT, None, 'any')['depth_convention'] == 'axial_minus_camera_z'
    assert depth_contract(HABITAT_GS, None, 'binding')['depth_convention'] == 'unvalidated'
    report = tmp_path / 'validation.json'
    value = {'passed': True, 'bindings_sha256': 'other', 'depth_units': 'meters',
             'depth_convention': 'axial_minus_camera_z'}
    for change in ({}, {'passed': False, 'bindings_sha256': 'binding'},
                   {'bindings_sha256': 'binding', 'depth_units': 'millimeters'}):
        report.write_text(json.dumps(dict(value, **change)))
        with pytest.raises(ValueError, match='does not match'):
            depth_contract(HABITAT_GS, report, 'binding')
    report.write_text(json.dumps(dict(value, bindings_sha256='binding')))
    contract = depth_contract(HABITAT_GS, report, 'binding')
    assert contract['depth_convention'] == 'axial_minus_camera_z'
    assert len(contract['depth_validation_sha256']) == 64


def test_runtime_and_scene_source_must_agree_at_load(tmp_path):
    mesh = settings(tmp_path).simulation.model_dump()
    mesh['datasets'] = {'test': {'episodes': tmp_path / 'episodes.json'}}
    packages = dict(mesh, datasets={'test': dict(mesh['datasets']['test'],
                                                 scenes={'m_floor': tmp_path / 'scene.json'})})
    for value, backend in ((mesh, 'habitat_gs'), (packages, 'habitat')):
        with pytest.raises(ValueError, match='does not match runtime'):
            Simulation.model_validate(dict(value, backend=backend))
    assert Simulation.model_validate(dict(packages, backend='habitat_gs')).backend == 'habitat_gs'
    with pytest.raises(ValueError, match='depth_validation'):
        Simulation.model_validate(dict(mesh, depth_validation=tmp_path / 'report.json'))
