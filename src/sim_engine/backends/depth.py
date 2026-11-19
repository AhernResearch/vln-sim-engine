"""Whether captured depth may be used as metric geometry.

Mesh depth is metric by construction. Gaussian depth is admitted as metric only with a
passed validation report for the exact loaded binding; otherwise it stays unvalidated
and visible-region planning rejects it.
"""

from __future__ import annotations

import json
from pathlib import Path

from sim_engine.core.io import file_sha256

AXIAL = "axial_minus_camera_z"
UNVALIDATED = "unvalidated"


def depth_contract(runtime, validation: Path | None, bindings_sha256: str) -> dict:
    if runtime.depth_source == "mesh":
        return {"depth_convention": AXIAL}
    if validation is None:
        return {"depth_convention": UNVALIDATED}
    report = json.loads(Path(validation).read_text())
    if (report.get("bindings_sha256") != bindings_sha256 or report.get("passed") is not True
            or report.get("depth_convention") != AXIAL or report.get("depth_units") != "meters"):
        raise ValueError("depth validation does not match this runtime")
    return {"depth_convention": AXIAL, "depth_validation_sha256": file_sha256(validation)}
