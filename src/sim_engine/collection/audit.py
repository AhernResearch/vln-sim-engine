"""Recording and original-box audits run after the simulator releases its worker."""

import json

from sim_engine.core.io import create_json, read_json
from sim_engine.navigation.visual_region import inside_bbox, project_world

from .recording import validate_recording


def audit_recording(request, folder, target, result):
    geometry = result["geometry"]
    validation = validate_recording(folder) if geometry["passed"] else None
    if validation:
        create_json(folder / "recording-check.json", validation)
    region = None
    if target:
        camera, views = read_json(folder / "camera.json"), read_json(folder / "view-sensors.json")
        last = None
        with (folder / "world_states.jsonl").open() as stream:
            for line in stream:
                last = json.loads(line)
        if last is None:
            raise ValueError("recording has no terminal frame")
        width, height = camera["resolution_wh"]
        uv = project_world(last["world_position"], camera["sensors"]["depth"]["intrinsics_K"],
                           views[target["target_view"]], width, height)
        region = {"projected_uv": uv, "bbox": target["bbox"],
                  "inside_original_bbox": inside_bbox(uv, target["bbox"])}
        create_json(folder / "region-terminal-check.json", region)
    passed = bool(geometry["passed"] and validation and (region is None or region["inside_original_bbox"]))
    return dict(result, status="completed_audit" if passed else "failed_or_uncertain",
                grounding=read_json(folder / "grounded-target.json") if target else None,
                region_terminal=region, recording_complete=bool(validation))
