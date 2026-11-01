import gzip
import json

from test_sessions import episode, settings

from sim_engine.config import Dataset, Physics
from sim_engine.data import EpisodeStore, sha256


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
    assert store.scene_hashes["rxr"][asset] == sha256(tmp_path / "rxr" / asset)
    assert store.get("r2r", "a")["_dataset"] == "r2r"
    assert store.get("rxr", "a")["_dataset"] == "rxr"
    assert store.metadata["rxr"]["physics"]["agent_height_m"] == 0.88
