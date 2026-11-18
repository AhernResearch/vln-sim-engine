# Habitat and Gaussian scene backends

Simulation, collection and planning use one sim-engine implementation. Run ordinary
Habitat 0.3.1 and the pinned CUDA Habitat-GS 0.3.3 in separate environments and
services. The Python `habitat_sim` distributions must not share an environment.
Retain the GS wheel’s Pillow 10.4 dependency when installing sim-engine; the common
package supports Pillow 10.4 through 12.x.
`deployment/habitat-gs.lock.json` pins the upstream source and CUDA compatibility
patches. Runtime identity includes the actual native binding hash. An installed
package's version alone does not establish source provenance; retain the build
receipt alongside the lock and deployment source snapshot.

## Scene preparation and serving

The standalone importer and baking code derive from open-nav-harness-tools;
source attribution and Apache license are packaged under `sim_engine/licenses`.
Only degree-zero binary little-endian Gaussian PLY is currently supported.

```bash
python -m sim_engine prepare-scan --archive M-floor-exam.zip --output /scenes/m-floor-v1 --lod 3
# Run baking with the pinned GS interpreter; creates navigation/ only once.
python -m sim_engine bake-scan --manifest /scenes/m-floor-v1/scene.json
export GS_SCENE_ROOT=/scenes/m-floor-v1
export POLICY_MODEL=/models/checkpoint
python -m sim_engine serve-sim --config examples/server.gs.json --port 18820
```

Existing frozen M-floor packages are supported without modification. Scene maps
bind logical scene IDs to manifests. Asset and navigation receipts are verified at
admission and initial worker load. Packages are immutable throughout deployment;
prepare a new directory when changing assets. Agent radius and height must match
the baked NavMesh settings. Legacy Habitat configs and RGB endpoints remain valid.

Baking samples recorded scan positions, converts them to float32 for the native
binding, and chooses the supported start with the greatest NavMesh boundary
clearance (queried up to 2 m). The navigation receipt records that clearance and
the selection rule. This avoids automatically starting at the first marginal
boundary point; it does not certify all routes or scan geometry.
`bake-scan --agent-radius-m R --agent-height-m H` sets the baked agent geometry
(defaults 0.18 m and 1.5 m); the serving dataset `physics` must use the same values.

`simulation.sensors` selects `rgb` (default), `rgbd`, or `panorama`.
`GET /v1/sessions/{id}/frames/{event}?sensor=depth` returns NPZ containing float32
`depth_m`; `sensor=front_rgb`, `right_rgb`, `back_rgb`, `left_rgb` and corresponding
`*_depth` require the panorama rig. Metadata is in `X-Sensor-Metadata`, encoding in
`X-Encoding`, and the payload hash in `X-Content-SHA256`. Captured sensor poses
are privileged telemetry; do not add them to policy generation requests. Missing
sensors/events return 404. Historical data is stored at execution time, never
reconstructed from the current simulator state.

Development episodes contain a start and instruction, with no fabricated scoring
goals. Metrics expose `evaluation_available=false` and null success/SPL/nDTW/goal
distance. Progress rewards and goal-less expert requests are rejected. A policy
service uses the same protocol and frame fetching as ordinary Habitat; point its
`policy.sim_url` at the GS service and choose a camera matching the policy.

## Collection and planning

Collection uses the same backend factory, state snapshots and RGB-D interface as
sessions. Four-direction captures keep the chassis fixed. Export planner assets
with `export-planner-assets`; scene resolution selects the manifest's frozen
NavMesh rather than deriving a filename from `scene_id`. The planner still uses
CPU PathFinder and the existing native kernels, without loading the GS renderer.
Run planner and collection with matching runtime versions and NavMesh identity.
The start-pool generator also resolves manifest assets.

`examples/collection.gs.json` provides a single-worker collection configuration;
set `GS_COLLECTION_ROOT` to a new collection output root.

Collection uses the existing `collection` configuration and explicit
`training_datasets` allowlist; adding a GS dataset to this operational allowlist
does not grant training admission. Raw collection receipts remain
`training_admission=pending_review`. Review asset geometry, depth accuracy and
endpoint semantics before any downstream admission.

GS depth is labeled `gaussian_unvalidated`. Before using it to select metric
visible-region targets, validate its units and axial projection convention:

```bash
python deployment/validate_gs_depth.py --output /evidence/depth-contract-v1 --gpu 2
```

Set `simulation.depth_validation` to the generated `receipt.json`. The service
requires a passed report matching the loaded binding SHA256. The two synthetic
planes test axial depth at 2 m and 4 m with a maximum error below 0.08 m across
central and off-axis pixels. This establishes the conversion convention, not scan
geometry accuracy. Missing evidence permits raw capture and NavMesh-only planning;
visible-region planning rejects unvalidated depth.

## Verification

Run unit/HTTP regressions with `.venv/bin/python -m pytest -q` and
`.venv/bin/ruff check src tests deployment`. Native verification is explicit:

```bash
python deployment/probe_backend.py --config examples/server.gs.json --output /evidence/gs-probe-v1
```

The probe uses a new output directory, renders actual GS RGB-D, checks fixed-state
panoramas and cached reset, loads CPU/native NavMesh queries, records and executes
a short sampled expert route, validates recording alignment, and exercises the
HTTP API through a spawned worker (including deduplication, historical depth and
STOP). It does not invoke a model or change running services. The model-client
loop remains the existing `rollout` command and must be validated separately
against the selected policy deployment.
