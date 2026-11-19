import gzip
import json
import math
from types import SimpleNamespace

from helpers import episode, settings

from sim_engine.config import Dataset, Physics
from sim_engine.core.io import file_sha256
from sim_engine.data import EpisodeStore
from sim_engine.metrics import navigation_metrics
from sim_engine.tasks import NavigationTask


def test_dataset_specific_geometry_and_physics_are_kept_separate(tmp_path):
    config = settings(tmp_path)
    annotation = tmp_path / "episodes.json.gz"
    with gzip.open(annotation, "wt") as stream:
        json.dump({"episodes": [episode()]}, stream)
    for name, height in [("r2r", 1.5), ("rxr", 0.88)]:
        root = tmp_path / name
        geometry = root / "mp3d/one/one.glb"
        geometry.parent.mkdir(parents=True)
        geometry.write_bytes(b"same-scene")
        geometry.with_suffix(".navmesh").write_bytes(name.encode())
        config.simulation.datasets[name] = Dataset(
            episodes=annotation, scenes_root=root, physics=Physics(agent_height_m=height)
        )
    store = EpisodeStore(config)
    asset = "mp3d/one/one.navmesh"
    assert store.scene_hashes["r2r"][asset] != store.scene_hashes["rxr"][asset]
    assert store.scene_hashes["rxr"][asset] == file_sha256(tmp_path / "rxr" / asset)
    assert store.get("r2r", "a")["_dataset"] == "r2r"
    assert store.get("rxr", "a")["_dataset"] == "rxr"
    assert store.metadata["rxr"]["physics"]["agent_height_m"] == 0.88
    assert store.metadata['r2r']['ndtw_reference'] == 'unavailable'
    assert '_metric_reference_path' not in store.get('r2r', 'a')


def test_ndtw_uses_explicit_locations_instead_of_sparse_route_waypoints(tmp_path):
    sparse = [[0., 0., 0.], [2., 0., 0.]]
    dense = [[i*.25, 0., 0.] for i in range(9)]
    episode_value = {'reference_path': sparse, 'goals': [{'position': dense[-1]}]}
    fake = SimpleNamespace(distance=math.dist)
    result = NavigationTask().metrics(fake, episode_value, dense, stopped=True)
    assert result['success'] == 1 and 'ndtw' not in result
    episode_value['_metric_reference_path'] = dense
    result = NavigationTask().metrics(fake, episode_value, dense, stopped=True)
    assert result['ndtw'] == 1
    assert navigation_metrics(dense, sparse, dense[-1], math.dist)['ndtw'] < 1
