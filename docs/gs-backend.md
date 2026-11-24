# Habitat and Gaussian scene backends

Simulation, collection and planning use one sim-engine implementation. Run ordinary
Habitat 0.3.1 and the pinned CUDA Habitat-GS 0.3.3 in separate environments and
services. The Python `habitat_sim` distributions must not share an environment.
Retain the GS wheel’s Pillow 10.4 dependency when installing sim-engine; the common
package supports Pillow 10.4 through 12.x.
`deployment/habitat-gs/habitat-gs.lock.json` pins the upstream source and CUDA compatibility
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

## Building the runtime

Everything for the build lives in `deployment/habitat-gs/` and is not part of the sim-engine
package, so ordinary installs need no GS dependency: the lock, the three CUDA compatibility
patches, `build_habitat_gs.sh` and `compile_habitat_gs.sh`.

```bash
HABITAT_GS_PYTHON=/usr/bin/python3.11 \
  deployment/habitat-gs/build_habitat_gs.sh /opt/habitat-gs/DEPLOY OWNER_ID
```

`DEPLOY` is a new, owned directory (`DEPLOY/owner` contains `OWNER_ID`) holding the offline inputs:
`build-inputs`, `upstream-source`, `source-resources`, `glm-headers`, `imath-source` and
`extra-wheels` archives plus `wheels/`. The script creates `DEPLOY/venv` (Python 3.11, numpy 2.2.6,
Pillow 10.4.0), applies the patches idempotently to `DEPLOY/upstream`, and compiles with
`CUDA_ARCHITECTURES` (default 80) and `BUILD_JOBS` (default 6). It writes a compile log and
`runtime-freeze.txt`; keep them with the lock. It touches nothing outside `DEPLOY`. Scene
preparation and baking use sim-engine's own `prepare-scan` and `bake-scan`; the old harness
`prepare_scan_task.py` and `record_habitat_gs.py` depended on the removed `navgpt` package and
were not ported.

## Serving environment

The GS venv has no fastapi, uvicorn or httpx. `deployment/habitat-gs/serve-requirements.txt` pins
the serving stack to the versions sim-engine is tested with. Either route keeps `habitat_sim`
untouched:

```bash
# A. Into a GS venv you own
uv pip install --python DEPLOY/venv/bin/python --no-deps -r deployment/habitat-gs/serve-requirements.txt
# B. Into a deps/ directory (a shared GS venv stays unmodified)
uv pip install --target DEPLOY/deps --python-platform x86_64-manylinux_2_28 --python-version 3.11 \
  --no-deps -r deployment/habitat-gs/serve-requirements.txt
PYTHONPATH=DEPLOY/deps:SIM_ENGINE/src DEPLOY/venv/bin/python -m sim_engine serve-sim --config ...
```

For route B the sim-engine checkout must include the compiled `sim_engine/navigation/_astar.abi3.so`
(the stable-ABI kernel works under Python 3.11). Ordinary Habitat 0.3.1 environments are unchanged.

## Session API on GS

Sessions use the same HTTP API as ordinary Habitat, including `observe`, `point_goal`, `retrace`
(by `leg_id` or `event_index`) and `velocity`. `point_goal` grounds on GS depth as it is
(`gaussian_unvalidated`); no validation gate or extra reply field is added for it. `validate-depth`
remains required only for visible-region target selection.

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
python -m sim_engine validate-depth --output /evidence/depth-contract-v1 --gpu 2
```

Set `simulation.depth_validation` to the generated `receipt.json`. The service
requires a passed report matching the loaded binding SHA256. The two synthetic
planes test axial depth at 2 m and 4 m with a maximum error below 0.08 m across
central and off-axis pixels. This establishes the conversion convention, not scan
geometry accuracy. Missing evidence permits raw capture and NavMesh-only planning;
visible-region planning rejects unvalidated depth.

## Verification

Run unit/HTTP regressions with `.venv/bin/python -m pytest -q` and
`.venv/bin/ruff check src tests`. Native verification is explicit:

```bash
python -m sim_engine probe-backend --config examples/server.gs.json --output /evidence/gs-probe-v1
```

The probe uses a new output directory, renders actual GS RGB-D, checks fixed-state
panoramas and cached reset, loads CPU/native NavMesh queries, records and executes
a short sampled expert route, validates recording alignment, and exercises the
HTTP API through a spawned worker (including deduplication, historical depth and
STOP). It does not invoke a model or change running services. The model-client
loop remains the existing `rollout` command and must be validated separately
against the selected policy deployment.

## Session acceptance (model-free)

`probe-backend` now also runs the session flow after its depth, deduplication and STOP checks.
The same flow runs against a running simulator with

```bash
python -m sim_engine probe-session --sim-url URL --dataset D --camera C --episode-id E --output NEW_DIR
```

It runs observe, `point_goal`, observe, `retrace` to an event, one `velocity` step and `finish`
(`forced_stop`), and fails if any reset, observe or step reply contains a privileged key (goals,
reference, poses; the list is in the receipt). `point_goal` tries a short list of floor pixels; a
rejected pixel executes nothing, and motion is never retried. The `finish` and `metrics` replies
are scoring endpoints and are not scanned. The flow's logic is unit-tested with the in-process test
environment in `tests/test_legs.py`.

Result of one run on the A100 (GPU 4, own server on port 18931, new output directory,
serving deps in a private `deps/`; the shared GS venv was not modified): scene package
`m-floor-open-tasks-v2` (frozen, unmodified), camera 480x270, hfov 120, height 0.88 m, level,
episode `m-floor-reception-001`, runtime `habitat_gs` 0.3.3, bindings SHA256
`03f2ae21...b9030`, receipt SHA256 `19ddd896...0e03`.

| Step | Result |
|---|---|
| point_goal (pixel 500,800 of the front view) | `reached`, events 0 to 11, 1.653 m, 0 collisions |
| retrace to event 5 | `reached`, events 11 to 26, 0.884 m, 0 collisions |
| velocity 0.2 m/s | 1 action, event 27 |
| finish / cleanup | `forced_stop`; 0 active sessions afterwards |
| privileged-field scan | 6 replies, none found |

Receipt: [`evidence/gs-session-acceptance-20261124/receipt.json`](evidence/gs-session-acceptance-20261124/receipt.json).
This shows the session API runs on GS and nothing privileged leaks in those replies. It is one
episode, one run and no model; it is not navigation success or a benchmark score.
