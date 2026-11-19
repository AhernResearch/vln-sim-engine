# VLN Sim Engine

Stateful Habitat simulation, asynchronous vLLM policy serving, and privileged expert-data
collection for visual navigation. One implementation serves two jobs:

- **Evaluation**: run a navigation policy against Habitat episodes (`serve-sim` +
  `serve-policy` + `rollout`).
- **Data production**: execute planned expert trajectories and record RGB-D evidence
  (`serve-collection` + optional `serve-planner`). The client that submits tasks, calls the
  annotation model and builds releases is [nav-data](https://github.com/AhernResearch/nav-data);
  training lives in nav-train.

The service boundary follows [GRAVA Sim Engine](https://github.com/AhernResearch/grava-sim-engine)
(reference commit `06dd03fd104501a8737e8e17888fc13adccc1eab`). Package layout and layering
rules are in [architecture](docs/architecture.md); the HTTP contract is in [API](docs/api.md).

| Command | Default port | Needs GPU | Needs Habitat | Purpose |
|---|---|---|---|---|
| `serve-sim` | 18810 | yes (EGL) | yes | Sessions: reset/step, immutable RGB frames, privileged metrics |
| `serve-policy` | 18811 | yes | no | vLLM `AsyncLLM` policy over frame references |
| `rollout` | – | no | no | Concurrent evaluation client; atomic per-episode outcomes |
| `serve-collection` | 18812 | yes (EGL) | yes | Expert collection: observe, plan, execute, record, audit |
| `serve-planner` | 18814 | no | yes (CPU PathFinder) | Standalone CPU planning pool for collection |

Asset and acceptance commands: `prepare-navmeshes`, `audit-navmesh`, `export-planner-assets`,
`build-start-pool`, `prepare-scan`, `bake-scan`, `validate-depth`, `probe-backend`
(`python -m sim_engine <command> --help`).

## 1. Installation

Python 3.10. Use **separate environments**; their native stacks conflict.

| Environment | Contents | Used by |
|---|---|---|
| dev (`.venv`) | `.[dev]` (ruff, pytest) | tests, `rollout` |
| habitat | `.[habitat]` + habitat-sim 0.3.1 headless with Bullet, `numpy<2` | `serve-sim`, `serve-collection`, `serve-planner`, asset commands |
| habitat-gs | `.[habitat]` + pinned CUDA Habitat-GS 0.3.3 ([lock](deployment/habitat-gs.lock.json)) | Gaussian-scan runtime ([GS backend](docs/gs-backend.md)) |
| policy | `.[policy]` + GPU-compatible PyTorch / vLLM | `serve-policy` |

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
/path/to/habitat/python -m pip install -e '.[habitat]'
/path/to/policy/python  -m pip install -e '.[policy]'
```

The validated mesh build is habitat-sim v0.3.1, commit
[`3d6d67d6deae4ab2472cc84df7a3cef1503f606d`](https://github.com/facebookresearch/habitat-sim/tree/3d6d67d6deae4ab2472cc84df7a3cef1503f606d),
built with `setup.py bdist_wheel --headless --with-bullet`. Replay parity with the earlier
0.2.4 build is in [`docs/evidence/habitat-031-velocity-acceptance.json`](docs/evidence/habitat-031-velocity-acceptance.json);
rendered pixels differ slightly, so 0.3.1 scores are a new evaluation identity. The two
`habitat_sim` distributions (0.3.1 and GS 0.3.3) must never share an environment.

`pip install` compiles the native planning kernel `sim_engine.navigation._astar` from
`src/sim_engine/navigation/*.cpp`. Planning and collection refuse to start without it, so
re-run the editable install after pulling C++ changes.

## 2. Configuration

Every service reads one JSON config with `simulation`, `policy`, `cameras` and (for
collection) `collection` sections. `${VAR}` references are expanded from the environment.

| Example | Use |
|---|---|
| [`examples/server.json`](examples/server.json) | R2R/RxR evaluation; needs `VLN_ASSET_ROOT`, `VLN_MODEL`, `VLN_RXR_SCENES_ROOT` |
| [`examples/server.gs.json`](examples/server.gs.json) | GS evaluation; needs `GS_SCENE_ROOT`, `POLICY_MODEL` |
| [`examples/collection.train.v1.json`](examples/collection.train.v1.json) | Mesh training collection (16 workers on GPUs 0/2/3/4) |
| [`examples/collection.gs.json`](examples/collection.gs.json) | Single-worker GS collection; needs `GS_COLLECTION_ROOT` too |

Each dataset selects its own scene root and physics. The example keeps the historical R2R
1.5 m height / 0.1 m radius / 15° turns / sliding configuration, and gives RxR 0.88 m height /
0.18 m radius / 30° turns / no sliding (from its
[task configuration](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/config/rxr_vlnce_english_task.yaml)).
Both keep the trained policy's 1.25 m camera and selectable pitch; these are diagnostic camera
protocols, so their scores are not official leaderboard scores.

**For each deployment, copy the example and freeze it, with a new output root.** Evaluation
identity includes checkpoint, processor, prompt/implementation hashes, assets, camera and
concurrency. A camera or engine change requires a new output directory.

## 3. Asset preparation (Habitat environment)

### Mesh scenes (MP3D / R2R / RxR)

The supplied R2R navmeshes use a taller agent, so RxR needs its own derived set. Derived
navmeshes record geometry, source/output hashes and all Recast settings. Original assets
remain available for historical reproduction.

```bash
export VLN_ASSET_ROOT=/path/to/assets
export VLN_MODEL=/path/to/checkpoint
export VLN_RXR_SCENES_ROOT=/path/to/derived/rxr_scene_datasets
/path/to/habitat/python -m sim_engine prepare-navmeshes \
  --config examples/server.json --dataset rxr_val_unseen_en \
  --source-root "$VLN_ASSET_ROOT/scene_datasets" --output "$VLN_RXR_SCENES_ROOT"
/path/to/habitat/python -m sim_engine audit-navmesh \
  --config examples/server.json --dataset rxr_val_unseen_en --output outputs/rxr-connectivity.json
```

### Gaussian scans (GS environment, CUDA)

```bash
python -m sim_engine prepare-scan --archive M-floor-exam.zip --output /scenes/m-floor-v1 --lod 3
python -m sim_engine bake-scan --manifest /scenes/m-floor-v1/scene.json \
  --agent-radius-m 0.18 --agent-height-m 1.5      # must match the dataset `physics`
python -m sim_engine validate-depth --output /evidence/depth-contract-v1 --gpu 2
python -m sim_engine probe-backend --config examples/server.gs.json --output /evidence/gs-probe-v1
```

Scan packages are immutable; prepare a new directory to change assets. Set
`simulation.depth_validation` to the produced `receipt.json`; without a passing receipt, GS
depth stays `gaussian_unvalidated` and visible-region planning rejects it. Details are in
[GS backend](docs/gs-backend.md).

## 4. Evaluation: simulator + policy + rollout

```bash
/path/to/habitat/python -m sim_engine serve-sim    --config examples/server.json          # GPU, Habitat
/path/to/policy/python  -m sim_engine serve-policy --config examples/server.json --gpu 2  # GPU, vLLM
curl --fail http://127.0.0.1:18810/v1/health
curl --fail http://127.0.0.1:18811/v1/health
.venv/bin/python -m sim_engine rollout \
  --sim-url http://127.0.0.1:18810 --policy-url http://127.0.0.1:18811 \
  --dataset r2r_val_unseen --episodes 10 1078 --concurrency 2 \
  --camera trained_ce30 --output outputs/r2r-pitch30
```

- The simulator uses the physical GPU index from the config. `serve-policy` sets
  `CUDA_VISIBLE_DEVICES` from `--gpu` before importing vLLM; vLLM memory utilization is a
  fraction of the whole GPU, so leave room for the simulator and other services.
- `rollout` writes each completed episode atomically and resumes only with the same model,
  assets, camera and concurrency. The output directory is the evaluation identity.
- `--max-plans N` is a **diagnostic-only** end-to-end check, never a benchmark score. Without
  it, the configured 500-action budget applies.
- STOP: the legacy `trajectory_v1` profile has no STOP supervision, and reaching the budget is
  never a successful stop. For STOP-trained checkpoints set `policy.prompt_profile` to
  `navigation_v2`. The model must then return `{"stop": false, "trajectory": [eight poses]}`
  or `{"stop": true, "trajectory": []}`. Only an explicit valid STOP ends an episode, and
  success still requires the stop to be within the goal radius.
- GS evaluation uses the same commands with `examples/server.gs.json`. GS development episodes
  have no scoring goals, so their metrics report `evaluation_available=false`.

## 5. Data production: collection service

`serve-collection` owns the persistent Habitat processes, scene cache, four-view RGB-D
observation, target grounding, planning, expert execution, raw recording and audit. It loads
**no model**: model calls, instruction interpretation, PointNav labels and release building
belong to the client ([nav-data](https://github.com/AhernResearch/nav-data)).

```bash
# Copy and freeze examples/collection.train.v1.json first; set a NEW collection.root.
/path/to/habitat/python -m sim_engine serve-collection --config collection.json --port 18812
curl --fail http://127.0.0.1:18812/v1/health
```

The essential `collection` keys are:

| Key | Meaning |
|---|---|
| `root` | Shared output root; the client must see the **same absolute path** |
| `training_datasets` | Explicit allowlist of datasets that may be collected |
| `worker_gpus` | One entry per worker process (e.g. four per GPU) |
| `capacity` | Global in-flight task limit (backpressure; 429 when exhausted) |
| `scene_cache_entries` | Scene backends cached per worker (LRU) |
| `operation_timeout_s`, `minimum_free_disk_bytes` | Per-operation timeout and disk reserve |
| `planning_profile_version` | Must be `collection-v5` (older profiles are rejected) |
| `planning_workers`, `planning_capacity`, `planning_urls` | Embedded or remote CPU planning (below) |

Operational rules:

- The privileged API binds to loopback; run it on a trusted collection host. The nav-data
  client runs on the same host or on one that shares the filesystem.
- Task kinds: `short_instructions` (the model draws a box on four views), `r2r_chunks`
  (assigned local instruction, private guides and goal) and `reference_geometry` (NavMesh
  measurement only, with no rendering or movement).
- Claims, frozen targets and terminal receipts are create-only. Identical resubmissions return
  the existing claim; changed ones conflict. Timeouts, worker loss and restarts leave the task
  uncertain. **Workers are never restarted automatically and movement is never replayed.**
- All collected data is `training_admission=pending_review`. Collection is not training
  admission.
- Disk and inodes both run out: an earlier production run stopped on inode exhaustion with
  472 GiB free. Monitor `df -i` alongside `minimum_free_disk_bytes`.

Endpoints, worker scheduling and the deployment measurements are in
[collection service](docs/collection-service.md). Fixed scene bindings are in
[fixed-scene collection](docs/fixed-scene-collection.md), and box-pixel sampling in
[visible-region sampling](docs/visible-region-mask-sampling.md).

### Planning: embedded or standalone

By default, `serve-collection` starts an embedded CPU planning pool, so no extra service is
needed. To move planning to other machines, export a relocatable asset bundle and point
collection at it:

```bash
/path/to/habitat/python -m sim_engine export-planner-assets --config collection.json --output /new/planner-bundle
/path/to/habitat/python -m sim_engine serve-planner --config /new/planner-bundle/planner.json --port 18814
```

```json
{"collection": {"planning_urls": ["http://planner-host:18814"], "planning_workers": 8, "planning_capacity": 64}}
```

Planner hosts need the same sim-engine commit and Habitat/NumPy versions, but no GPU, model
or scene GLB. The protocol, budgets and failure semantics (`feasible`, `infeasible`,
`budget_exhausted`, `cancelled`, `worker_failed`) are in [planning service](docs/planning-service.md).

### Start pools

`build-start-pool --config C --dataset D --camera CAM --owner O --output DIR` samples
train-only start poses with NavMesh clearance (`--successes`, `--minimum-per-scene`). It uses
the CPU PathFinder only, with no simulator or movement. The pool is an input for preparing
free-instruction cases; the frozen case list itself is a nav-data artifact.

## 6. Contract

- A model request carries instruction + camera profile + causal observations only. Goals,
  global poses, reference paths and rewards stay on the simulator side, and so do captured
  sensor poses. Infrastructure failure is a failed operation. Invalid model output is
  recorded as `invalid_output`.
- Steps carry `request_id` and `expected_event_index`. A retry returns the cached receipt; a
  changed request with the same ID or a stale index returns 409. Sessions hold their worker
  until closed. Worker failures invalidate their sessions and are never silently retried.
- The eight `[x, y, yaw]` poses share a current-body origin (x forward, y left, yaw CCW;
  metres/radians). They are spatial progress samples. The controller executes at most four
  primitive actions per plan.
- Public policy serving uses fixed weights. Hot weight sync, PPO/GRPO and a validated reward
  are future work. `progress_v1` reward is experimental; benchmark mode defaults to `none`.

## 7. Validation

```bash
.venv/bin/ruff check src tests
.venv/bin/python -m pytest -q
```

Unit tests cover session isolation, idempotency, stale requests, causal history, strict
schemas, exclusion of privileged inputs, collection queue/claim/cancel/no-replay and layering.
They do not establish navigation quality or throughput: native Habitat and vLLM acceptance
must be measured on real assets and GPUs. Recorded acceptance:
[2026-11-01 native evaluation](docs/validation-20261101.md),
[GS validation](docs/validation-gs-20261116.md),
[prompt-v2 parity](docs/evidence/prompt-v2-parity.json),
[STOP contract](docs/evidence/native-stop-contract-v2.json). Keep `docs/evidence/` and
`docs/validation-*.md` in sync when protocol behavior changes.
