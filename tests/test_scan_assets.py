import io
import json
import math
import struct
import zipfile
from pathlib import Path

import pytest
from helpers import settings

from sim_engine.config import CollectionSettings, Dataset
from sim_engine.core.io import create_json, file_sha256
from sim_engine.data import EpisodeStore
from sim_engine.planning.assets import registered_assets
from sim_engine.planning.client import build_job
from sim_engine.scenes import resolve_scene
from sim_engine.scenes.package import HABITAT_GS_COMMIT, asset_path, load_scan_package, verify_manifest
from sim_engine.scenes.prepare import prepare, transform_ply

FIELDS = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
          "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]


def ply(values=None):
    header = ("ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
              + "".join("property float " + field + "\n" for field in FIELDS)
              + "end_header\n")
    values = values or [1, 2, 3, 0, 0, 1, .2, .3, .4, 2, -3, -4, -5, 1, 0, 0, 0]
    return header.encode() + struct.pack("<17f", *values)


@pytest.fixture
def prepared(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("scan/point_cloud/iteration_100/point_cloud_3.ply", ply())
        bundle.writestr("scan/mesh/mesh.ply", ply())
        bundle.writestr("scan/render/assets/poses.json", json.dumps({"poses": [{"T": [1, 2, 3]}]}))
        bundle.writestr("../../escape", "must never extract")
        bundle.writestr("scan/semantic_goals.json", '{"goal": [99, 99, 99]}')
    return prepare(archive, tmp_path / "scene")


def add_navigation(manifest):
    folder = manifest.parent / "navigation"
    folder.mkdir()
    (folder / "scene.navmesh").write_bytes(b"navmesh fixture")
    create_json(folder / "episodes.json", {"purpose": "development", "episodes": [
        {"episode_id": "0", "scene_id": "m_floor", "start_position": [0, 0, 0],
         "start_rotation": [0, 0, 0, 1], "instruction": {"instruction_text": "Observe, then stop."}}]})
    create_json(folder / "receipt.json", {"schema": "open_nav_scan_navigation_v1",
        "source_manifest_sha256": file_sha256(manifest),
        "assets": {key: {"path": name, "sha256": file_sha256(folder / name)} for key, name in
                   (("navmesh", "scene.navmesh"), ("episodes", "episodes.json"))}})
    return folder


def test_coordinate_conversion_rotates_covariance_and_preserves_other_properties(tmp_path):
    output = tmp_path / "scene.gs.ply"
    transform_ply(io.BytesIO(ply()), output, gaussian=True)
    raw = output.read_bytes().split(b"end_header\n", 1)[1]
    values = struct.unpack("<17f", raw)
    assert values[:3] == pytest.approx([1, 3, -2])
    assert values[3:6] == pytest.approx([0, 1, 0])
    assert values[6:13] == pytest.approx([.2, .3, .4, 2, -3, -4, -5])
    assert values[13:] == pytest.approx([math.sqrt(.5), -math.sqrt(.5), 0, 0])


def test_truncated_and_nonfinite_gaussians_fail_without_a_complete_manifest(tmp_path):
    for name, data in (("truncated", ply()[:-1]), ("nan", ply([math.nan] + [1] * 16))):
        with pytest.raises(ValueError):
            transform_ply(io.BytesIO(data), tmp_path / name, gaussian=True)


def test_package_is_frozen_and_annotations_are_excluded(prepared):
    manifest = verify_manifest(prepared)
    assert manifest["upstream"]["commit"] == HABITAT_GS_COMMIT
    assert manifest["evaluation_available"] is False
    assert not (prepared.parent.parent / "escape").exists()
    assert not list(prepared.parent.rglob("*goals*"))
    before = prepared.read_bytes()
    with pytest.raises(FileExistsError):
        prepare(prepared.parent.parent / "source.zip", prepared.parent)
    assert prepared.read_bytes() == before
    mesh = asset_path(prepared.parent, manifest["assets"]["mesh"]["path"])
    mesh.write_bytes(mesh.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(prepared)


def test_navigation_receipt_rejects_changed_navmesh_and_goal_annotations(prepared):
    folder = add_navigation(prepared)
    assert load_scan_package(prepared)[3][0]["start_position"] == [0, 0, 0]
    (folder / "scene.navmesh").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_scan_package(prepared)


def test_even_hash_consistent_scan_episodes_reject_evaluation_data(prepared):
    folder = add_navigation(prepared)
    episode_path = folder / "episodes.json"
    value = json.loads(episode_path.read_text())
    value["episodes"][0]["goals"] = [{"position": [9, 9, 9]}]
    episode_path.write_text(json.dumps(value))
    receipt_path = folder / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["assets"]["episodes"]["sha256"] = file_sha256(episode_path)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="cannot contain evaluation annotations"):
        load_scan_package(prepared)


def test_scene_paths_cannot_escape_package(tmp_path):
    for path in ("../outside", "/etc/passwd"):
        with pytest.raises(ValueError):
            asset_path(tmp_path, path)



def test_scene_registry_resolves_manifest_and_exports_navmesh(prepared, tmp_path):

    navigation = add_navigation(prepared)
    cfg = settings(tmp_path)
    cfg.simulation.backend = 'habitat_gs'
    cfg.simulation.datasets['scan'] = Dataset(
        episodes=navigation / 'episodes.json', task='development', scenes={'m_floor': prepared})
    cfg.collection = CollectionSettings(root=tmp_path / 'records', training_datasets=['scan'])
    store = EpisodeStore(cfg)
    scene = resolve_scene(cfg, 'scan', 'm_floor')
    assert scene.navmesh == navigation / 'scene.navmesh'
    assert store.scene_hashes['scan']['m_floor']['manifest'] == file_sha256(prepared)
    assets = registered_assets(cfg, store)
    assert len(assets) == 1 and assets[0].path == scene.navmesh
    cfg.simulation.backend = 'habitat'
    with pytest.raises(ValueError, match='runtime'):
        resolve_scene(cfg, 'scan', 'm_floor')


def test_planning_job_uses_manifest_navmesh_instead_of_scene_id_filename(prepared, tmp_path):

    navigation = add_navigation(prepared)
    cfg = settings(tmp_path)
    cfg.simulation.backend = 'habitat_gs'
    cfg.simulation.datasets['scans'] = Dataset(
        episodes=navigation / 'episodes.json', task='development', scenes={'m_floor': prepared})
    cfg.collection = CollectionSettings(root=tmp_path / 'records', training_datasets=['scans'])
    folder = tmp_path / 'task'
    folder.mkdir()
    for name in ['start-state.json', 'camera.json', 'view-sensors.json']:
        (folder / name).write_text('{}')
    (folder / 'front-depth.npz').write_bytes(b'opaque frozen depth fixture')
    job = build_job(cfg, {'request_id': 'a' * 64, 'owner': 'b' * 32,
        'kind': 'short_instructions', 'dataset': 'scans', 'scene_id': 'm_floor',
        'camera': 'trained_ce30', 'seed': 0, 'start_position': [0, 0, 0],
        'start_rotation': [0, 0, 0, 1]}, folder,
        {'target_view': 'front', 'bbox': [400, 600, 600, 800]})
    assert job.navmesh_sha256 == file_sha256(navigation / 'scene.navmesh')
    assert job.task.dataset == 'scans'
    assert set(job.artifacts) == {'start-state.json', 'camera.json', 'view-sensors.json',
                                  'front-depth.npz'}


def test_pinned_commit_matches_deployment_lock():
    lock = Path(__file__).resolve().parents[1] / 'deployment/habitat-gs/habitat-gs.lock.json'
    assert json.loads(lock.read_text())['commit'] == HABITAT_GS_COMMIT
