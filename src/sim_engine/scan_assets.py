"""Prepare immutable, metre-scale scan assets without importing Habitat or CUDA.

Only explicitly selected ZIP members are read. Source archives and evaluation
annotations are never modified or copied into the navigation episode file.
"""

import argparse
import json
import math
import re
import shutil
import struct
import zipfile
from pathlib import Path

from sim_engine.core.io import create_json, file_sha256


def vector(value, size):
    return (isinstance(value, (list, tuple)) and len(value) == size
            and all(type(v) in (int, float) and math.isfinite(v) for v in value))


HABITAT_GS_COMMIT = "cb01b2fdf850ce3cab28ad00461a68ac7719f23f"
SCAN_TO_HABITAT = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
PLY_TYPES = {"float": "f", "float32": "f", "double": "d", "float64": "d",
             "uchar": "B", "uint8": "B", "char": "b", "int8": "b",
             "ushort": "H", "uint16": "H", "short": "h", "int16": "h",
             "uint": "I", "uint32": "I", "int": "i", "int32": "i"}


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


def ply_header(stream):
    lines, fields, vertex_count, element = [], [], None, None
    for _ in range(1024):
        line = stream.readline(4096)
        if not line or not line.endswith(b"\n"):
            raise ValueError("Invalid or oversized PLY header")
        text = line.decode("ascii").strip()
        lines.append(line)
        parts = text.split()
        if parts[:2] == ["element", "vertex"]:
            vertex_count, element = int(parts[2]), "vertex"
        elif parts[:1] == ["element"]:
            if vertex_count is None:
                raise ValueError("PLY vertices must be the first element")
            element = parts[1]
        elif parts[:1] == ["property"] and element == "vertex":
            if len(parts) != 3 or parts[1] not in PLY_TYPES:
                raise ValueError("Unsupported PLY vertex property: " + text)
            fields.append((parts[2], PLY_TYPES[parts[1]]))
        if text == "end_header":
            break
    else:
        raise ValueError("Oversized PLY header")
    if lines[0].strip() != b"ply" or b"format binary_little_endian 1.0\n" not in lines:
        raise ValueError("Expected binary little-endian PLY")
    if vertex_count is None or vertex_count <= 0 or vertex_count > 20_000_000:
        raise ValueError("Invalid PLY vertex count")
    names = [name for name, _ in fields]
    if len(names) != len(set(names)) or not {"x", "y", "z"} <= set(names):
        raise ValueError("PLY requires unique x/y/z properties")
    return lines, fields, vertex_count


def transform_ply(source, destination, gaussian=False):
    """Rotate Z-up to Y-up, including Gaussian covariances, without rescaling.

    The rotation is Rx(-90 degrees): (x,y,z) -> (x,z,-y). SH degree zero
    is supported; rotating higher SH bands requires an additional conversion.
    """
    lines, fields, count = ply_header(source)
    names = [name for name, _ in fields]
    indices = {name: i for i, name in enumerate(names)}
    if gaussian:
        required = {"rot_0", "rot_1", "rot_2", "rot_3", "opacity",
                    "scale_0", "scale_1", "scale_2", "f_dc_0", "f_dc_1", "f_dc_2"}
        if not required <= set(names) or any(name.startswith("f_rest_") for name in names):
            raise ValueError("This importer requires a degree-zero Gaussian PLY")
    record = struct.Struct("<" + "".join(kind for _, kind in fields))
    with Path(destination).open("xb") as output:
        for line in lines[:-1]:
            # Exported coordinate metadata would become stale after rotation.
            if not re.match(rb"comment (min|max|offset|shift|scale)[xyz] ", line):
                output.write(line)
        output.write(b"comment open_nav_scan_v1 rotation x,z,-y; units unchanged\nend_header\n")
        for _ in range(count):
            raw = source.read(record.size)
            if len(raw) != record.size:
                raise ValueError("Truncated PLY vertex payload")
            values = list(record.unpack(raw))
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Non-finite PLY vertex property")
            for triple in (("x", "y", "z"), ("nx", "ny", "nz")):
                if set(triple) <= set(names):
                    _, iy, iz = [indices[name] for name in triple]
                    values[iy], values[iz] = values[iz], -values[iy]
            if gaussian:
                w, x, y, z = [values[indices["rot_" + str(i)]] for i in range(4)]
                norm = math.sqrt(w*w + x*x + y*y + z*z)
                if norm < 1e-8:
                    raise ValueError("Zero Gaussian quaternion")
                c = math.sqrt(0.5) / norm
                for i, v in enumerate((c*(w+x), c*(x-w), c*(y+z), c*(z-y))):
                    values[indices["rot_" + str(i)]] = v
            output.write(record.pack(*values))
        shutil.copyfileobj(source, output)
    return count


def prepare(archive, output, lod=3):
    """Import the XGRIDS M-floor scan; never extract arbitrary archive paths."""
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if lod not in range(5):
        raise ValueError("LOD must be between zero (full) and four")
    suffix = "/point_cloud/iteration_100/point_cloud{}.ply".format("" if lod == 0 else "_" + str(lod))
    with zipfile.ZipFile(archive) as bundle:
        members = [i for i in bundle.infolist() if not i.filename.startswith("__MACOSX/")]

        def member(ending):
            found = [i for i in members if i.filename.endswith(ending)]
            if len(found) != 1 or found[0].file_size > 1_000_000_000:
                raise ValueError("Missing, ambiguous or oversized scan asset: " + ending)
            return found[0]

        selected = {"render": member(suffix), "mesh": member("/mesh/mesh.ply"),
                    "scan_poses": member("/render/assets/poses.json")}
        output.mkdir(parents=True, exist_ok=False)
        (output / "stages").mkdir()
        assets = {}
        for key, ending in (("render", "gs.ply"), ("mesh", "mesh.ply")):
            path = output / "stages" / ("m_floor." + ending)
            with bundle.open(selected[key]) as source:
                count = transform_ply(source, path, gaussian=(key == "render"))
            assets[key] = {"path": str(path.relative_to(output)), "sha256": file_sha256(path),
                           "source_member": selected[key].filename, "vertices": count}
        poses = json.loads(bundle.read(selected["scan_poses"]))["poses"]
        positions = []
        for pose in poses:
            x, y, z = pose["T"]
            if not all(math.isfinite(v) for v in (x, y, z)):
                raise ValueError("Non-finite scan position")
            positions.append([x, z, -y])
        path = output / "scan_positions.json"
        create_json(path, {"positions": positions, "role": "coverage_diagnostic_only"})
        assets["scan_poses"] = {"path": path.name, "sha256": file_sha256(path),
                                "source_member": selected["scan_poses"].filename}
        for key, asset in (("stage", "render"), ("geometry_stage", "mesh")):
            path = output / (key + ".stage_config.json")
            create_json(path, {"render_asset": assets[asset]["path"], "up": [0, 1, 0],
                              "front": [0, 0, -1], "norm_quaternion": False})
            assets[key] = {"path": path.name, "sha256": file_sha256(path)}
    manifest = {"schema": "open_nav_scan_v1", "backend": "habitat_gs", "purpose": "development",
                "scene_id": "m_floor", "lod": lod, "units": "metre",
                "scan_to_habitat": SCAN_TO_HABITAT,
                "upstream": {"repository": "https://github.com/zju3dv/habitat-gs",
                             "commit": HABITAT_GS_COMMIT},
                "source": {"archive": archive.name, "sha256": file_sha256(archive)},
                "assets": assets, "geometry_review": "required", "evaluation_available": False}
    create_json(output / "scene.json", manifest)
    return output / "scene.json"


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new, immutable scene directory")
    parser.add_argument("--lod", type=int, choices=range(5), default=3)
    args = parser.parse_args()
    print(prepare(args.archive, args.output, args.lod))


if __name__ == "__main__":
    main()
