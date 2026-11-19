"""Immutable scan packages: verified manifest, baked NavMesh and development episodes.

A package is loaded only by a runtime whose scene source is ``scan_package``.
"""

import json
import math
import re
from pathlib import Path

from sim_engine.core.io import file_sha256

HABITAT_GS_COMMIT = "cb01b2fdf850ce3cab28ad00461a68ac7719f23f"
SCAN_TO_HABITAT = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]


def vector(value, size):
    return (isinstance(value, (list, tuple)) and len(value) == size
            and all(type(v) in (int, float) and math.isfinite(v) for v in value))


def asset_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Asset must remain inside the scene package: " + str(relative))
    return path


def verify_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != "open_nav_scan_v1"
            or manifest.get("purpose") != "development"
            or manifest.get("backend") != "habitat_gs"):
        raise ValueError("Expected a Habitat-GS development scene manifest")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("upstream", {}).get("commit", "")):
        raise ValueError("Habitat-GS source must be pinned to a full commit")
    if manifest["upstream"]["commit"] != HABITAT_GS_COMMIT:
        raise ValueError("Unsupported Habitat-GS commit")
    if (manifest.get("units") != "metre" or manifest.get("scan_to_habitat") != SCAN_TO_HABITAT
            or not {"render", "mesh", "scan_poses", "stage", "geometry_stage"} <= set(manifest["assets"])):
        raise ValueError("Scene package has an unsupported coordinate or asset contract")
    for asset in manifest["assets"].values():
        source = asset_path(path.parent, asset["path"])
        if file_sha256(source) != asset["sha256"]:
            raise ValueError("Scene asset hash mismatch: " + str(source))
    return manifest


def load_scan_package(path):
    path = Path(path).resolve()
    manifest = verify_manifest(path)
    navigation = path.parent / "navigation"
    receipt = json.loads((navigation / "receipt.json").read_text())
    if (receipt.get("schema") != "open_nav_scan_navigation_v1"
            or receipt.get("source_manifest_sha256") != file_sha256(path)
            or set(receipt.get("assets", {})) != {"navmesh", "episodes"}):
        raise ValueError("Navigation receipt does not match the prepared scene")
    for item in receipt["assets"].values():
        if file_sha256(asset_path(navigation, item["path"])) != item["sha256"]:
            raise ValueError("Navigation asset hash mismatch: " + item["path"])
    episodes_path = asset_path(navigation, receipt["assets"]["episodes"]["path"])
    data = json.loads(episodes_path.read_text())
    if data.get("purpose") != "development" or not data.get("episodes"):
        raise ValueError("Scan episodes must explicitly select development")
    seen = set()
    for ep in data["episodes"]:
        if set(ep) != {"episode_id", "scene_id", "start_position", "start_rotation", "instruction"}:
            raise ValueError("Development scan episodes cannot contain evaluation annotations")
        if not isinstance(ep["episode_id"], str) or ep["episode_id"] in seen:
            raise ValueError("Invalid or duplicate scan episode ID")
        seen.add(ep["episode_id"])
        if ep["scene_id"] != manifest["scene_id"]:
            raise ValueError("Scan episode references a different scene")
        if not vector(ep["start_position"], 3) or not vector(ep["start_rotation"], 4):
            raise ValueError("Invalid scan start pose")
        if abs(sum(v*v for v in ep["start_rotation"]) - 1) > 1e-4:
            raise ValueError("Scan start rotation must be a unit quaternion (xyzw)")
        if (set(ep["instruction"]) != {"instruction_text"}
                or not isinstance(ep["instruction"]["instruction_text"], str)
                or not ep["instruction"]["instruction_text"].strip()):
            raise ValueError("Scan episode needs a navigation instruction")
    return path, manifest, receipt, data["episodes"]


def check_scan_package(path):
    path, manifest, receipt, episodes = load_scan_package(path)
    return {"dataset": "scan", "split": "development", "backend": "habitat_gs",
            "episode_count": len(episodes), "evaluation_available": False,
            "scene_manifest_sha256": file_sha256(path), "habitat_gs_commit": manifest["upstream"]["commit"],
            "geometry_review": "required", "navmesh_sha256": receipt["assets"]["navmesh"]["sha256"]}
