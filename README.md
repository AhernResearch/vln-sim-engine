# VLN Sim Engine

Independent stateful Habitat simulation and asynchronous vLLM inference services for
visual navigation evaluation and future reinforcement-learning rollouts.

The service boundary follows [GRAVA Sim Engine](https://github.com/AhernResearch/grava-sim-engine)
(reference commit `06dd03fd104501a8737e8e17888fc13adccc1eab`): simulation, scoring,
clients and model training have separate dependencies. VLN adds persistent episode
sessions because each observation depends on executed actions.

## Components

- `serve-sim`: one Habitat/EGL owner per worker **process**, with independent episode
  state, collision handling, immutable RGB frames and privileged metrics.
- `serve-policy`: vLLM `AsyncLLM`, dynamic engine scheduling, raw RGB frame references,
  history-image caching, bounded admission, generated tokens/log-probabilities and
  fixed policy-version labels.
- `rollout`: multiple episodes run concurrently; each episode retains causal order.
  Completed episodes are written atomically and can be resumed with the same
  model, assets, camera and concurrency configuration.

The model request contract contains instruction, camera profile and causal
observations only. Scene goals, global positions, reference paths and rewards stay
on the simulator/rollout side. An infrastructure failure remains a failed operation.
Invalid model output is explicitly recorded as `invalid_output`.

## Installation

Python 3.10. Simulator and policy use separate environments.
The validated native Habitat build is commit
[`f179b584bcd713c5a2a998132211e2cae881d6d1`](https://github.com/facebookresearch/habitat-sim/tree/f179b584bcd713c5a2a998132211e2cae881d6d1),
built headless with Bullet for Python 3.10.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
# Policy environment: install the pinned core and GPU-compatible PyTorch stack.
/path/to/policy/python -m pip install -e '.[policy]'
# Simulator environment additionally needs pinned habitat-sim 0.2.4, headless,
# with Bullet. It does not require Torch, Swift or vLLM.
/path/to/habitat/python -m pip install -e '.[habitat]'
```

Set asset/checkpoint locations and use the same configuration for both services:

```bash
export VLN_ASSET_ROOT=/path/to/assets
export VLN_MODEL=/path/to/epoch1-model
export VLN_RXR_SCENES_ROOT=/path/to/derived/rxr_scene_datasets
# The supplied R2R navmeshes use a taller agent. Build a separate RxR set.
/path/to/habitat/python -m sim_engine prepare-navmeshes \
  --config examples/server.json --dataset rxr_val_unseen_en \
  --source-root "$VLN_ASSET_ROOT/scene_datasets" --output "$VLN_RXR_SCENES_ROOT"
/path/to/habitat/python -m sim_engine audit-navmesh \
  --config examples/server.json --dataset rxr_val_unseen_en --output outputs/rxr-connectivity.json
/path/to/habitat/python -m sim_engine serve-sim --config examples/server.json
/path/to/policy/python -m sim_engine serve-policy --config examples/server.json --gpu 2
```

The simulator uses its physical GPU index from the configuration. The policy CLI
sets `CUDA_VISIBLE_DEVICES` before importing the inference engine. vLLM memory
utilization is a fraction of total GPU capacity, so it must leave room for the
simulator and any existing services.

Datasets select their own scene root and physics. The example keeps the historical
R2R 1.5m height / 0.1m radius / 15° turns / sliding configuration, and gives RxR
0.88m height / 0.18m radius / 30° turns / no sliding, following its
[task configuration](https://github.com/jacobkrantz/VLN-CE/blob/master/habitat_extensions/config/rxr_vlnce_english_task.yaml).
Both retain the trained policy's 1.25m camera and selectable pitch; these are
diagnostic camera protocols, so their scores are not official leaderboard scores.
Derived navmeshes record geometry, source/output hashes and all numeric Recast
settings. Original assets remain available for historical reproduction.

```bash
curl --fail http://127.0.0.1:18810/v1/health
curl --fail http://127.0.0.1:18811/v1/health
.venv/bin/python -m sim_engine rollout \
  --sim-url http://127.0.0.1:18810 --policy-url http://127.0.0.1:18811 \
  --dataset r2r_val_unseen --episodes 10 1078 --concurrency 2 \
  --camera trained_ce30 --output outputs/r2r-pitch30
```

`--max-plans 2` is a short **diagnostic-only** end-to-end check; it must not be
reported as a complete benchmark score. Without it, the configured 500-action
budget applies. The present q8 policy has no STOP supervision; reaching the budget
never counts as a successful stop. `policy_stop` is available only for a future
policy that explicitly emits a stop decision.

## Protocol

See [the API contract](docs/api.md). Both services expose `/docs` and `/openapi.json`.
See [service boundaries](docs/architecture.md) for ownership and RL integration.

Simulator step requests include `request_id` and `expected_event_index`. A retry
returns the cached result; a changed request with the same ID or a stale event
index returns 409. Sessions reserve a worker until explicitly closed. Worker
failures invalidate their sessions; partial simulation is never silently retried.

The eight `[x,y,yaw]` poses use a common current-body origin: x forward, y left,
yaw counterclockwise, metres/radians. They are **spatial progress samples**, not
eight seconds or eight actuator commands. The controller executes at most four
primitive actions and returns an observation for each executed action.

Public policy serving uses fixed checkpoint weights. Hot weight synchronization,
PPO/GRPO training and a validated long-horizon reward are subsequent work. The
optional `progress_v1` reward is experimental; benchmark mode defaults to `none`.
Tokens, behavior log-probabilities, actual execution prefixes and policy-version
labels provide a rollout interface for that future training work.

## Validation

```bash
.venv/bin/ruff check src tests
.venv/bin/python -m pytest -q
```

Tests exercise session isolation, idempotent execution, stale requests, causal
history, strict numerical schemas and exclusion of privileged policy inputs.
Native Habitat and vLLM acceptance must additionally be measured on real assets
and a real GPU; unit tests do not establish navigation quality or throughput.

The [2026-11-01 native acceptance](docs/validation-20261101.md) records both
benchmarks, live vLLM requests, input compatibility, and dataset-specific navmeshes.
