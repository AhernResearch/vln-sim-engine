# Fixed scene collection workers

Collection settings optionally accept `worker_scenes`, with exactly one binding
per `worker_gpus` entry. This freezes each worker to a registered training
dataset/scene/camera key for the lifetime of the service:

```json
{
  "worker_gpus": [0, 2],
  "worker_scenes": [
    {"dataset": "train_pilot", "scene_id": "mp3d/17DRP5sb8fy/17DRP5sb8fy.glb", "camera": "visible_88cm_hfov120"},
    {"dataset": "train_pilot", "scene_id": "mp3d/2n8kARJN3HM/2n8kARJN3HM.glb", "camera": "visible_88cm_hfov120"}
  ]
}
```

These are fields inside `collection`, alongside its explicit output root and
capacity. Bindings must use configured training datasets, pinned registered
scenes and known cameras. A task outside this scene window is rejected before
claim creation. A worker never steals work from another binding, including while
waiting for a model target; it can observe further tasks for its own scene.
Execution remains prioritized over new observations for that worker.

The IPC boundary independently checks the binding before submitting an operation.
Both observations and execution therefore use the same fixed worker. Waiting
targets are still file-backed and do not reserve the worker itself. If a bound
worker dies, its active/waiting tasks become failed or uncertain; new work for
that binding is unavailable. Other scene workers continue. Nothing migrates or
replays automatically. A per-task exception can still close a backend and cause
a subsequent independent task to rebuild the **same** scene; that is distinct
from dispatch-driven scene switching and must be counted in measurements.

Health adds `fixed_scene_window` and per-worker identities, bindings, current
phase, operation counters, accumulated busy seconds and last cache statistics.
Ownership files also retain the binding. Omitting both `worker_scenes` and
`scene_schedule` preserves the existing cache-affinity scheduler.

nav-train remains responsible for quotas, training-pose selection, model calls,
labels and result accounting. A bounded client must interleave or fairly admit
tasks across the active scenes: placing all tasks for the first scene at the
front of a 128-entry prefetch queue would leave other pinned workers idle.
The fixed16 experiment uses round-robin scene order and distinct generated task
owners for warmup and measurement. It does not change default mainline settings.

Unit tests cover multiple prefetched tasks and out-of-order target arrival with
unchanged worker/scene ownership, rejection outside the window, invalid bindings,
and failure of one pinned worker without replay or migration of its tasks.

The 2026-11-14 train-only experiment completed 1,024 attempts across 16 distinct
fixed scenes in 1,118.01 seconds: 750 audited completions, 261 exclusions before
movement and 13 failures/uncertain attempts. No dispatch-driven scene switching
occurred; two invalid-depth recording exceptions caused the next independent
tasks to rebuild their original scene (18 total loads including 16 initial loads).
All workers survived. Static batch drain exposed long planning/preview tails;
rolling replacement is a next experiment, not a measured capability here.
The owned experimental service on 18813 was stopped after draining; the existing
18812 mainline was not controlled. Full results and ownership records are in
[measurement evidence](evidence/fixed16-scenes-20261114.analysis.json).
Implementation validation: 109 sim-engine tests and Ruff passed.

## Rolling scene window

Use `collection.scene_schedule` instead of `worker_scenes` to freeze an ordered
list of unique dataset/scene/camera bindings and their **attempt** budgets:

```json
{
  "worker_gpus": [0, 2],
  "scene_cache_entries": 1,
  "scene_schedule": [
    {"dataset": "train_pilot", "scene_id": "mp3d/17DRP5sb8fy/17DRP5sb8fy.glb", "camera": "visible_88cm_hfov120", "attempts": 100},
    {"dataset": "train_pilot", "scene_id": "mp3d/2n8kARJN3HM/2n8kARJN3HM.glb", "camera": "visible_88cm_hfov120", "attempts": 100},
    {"dataset": "train_pilot", "scene_id": "mp3d/7y3sRwLe3Va/7y3sRwLe3Va.glb", "camera": "visible_88cm_hfov120", "attempts": 100}
  ]
}
```

There must initially be at least one scheduled scene per worker (up to 32).
Only active bindings accept new claims. An accepted claim consumes one attempt;
its success, exclusion, cancellation or uncertain failure consumes one terminal
result. Re-reading an existing claim never consumes another attempt. A queued,
finished or exhausted scene is rejected before creating a claim.

When one scene has reached its complete attempt budget, every claim is terminal,
and its worker is idle, that worker binds to the next queued scene. Other scenes
do not wait for a batch barrier. Observation-ready tasks waiting for a model still
count as active claims and prevent premature replacement. The existing actor's
one-entry cache evicts the old scene when the next scene is first used; the worker
process is retained. Each binding transition gets a new immutable file under
`instances/<owner>/workers/<index>/bindings/`; initial ownership is preserved.

Health exposes `rolling_scene_window`, `scene_schedule` (key, budget, claimed,
terminal, status and worker identity), and each worker's binding generation.
Worker death makes its scene unavailable; no task migrates or replays. The frozen
schedule is scoped to this service lifetime, without automatic restart/resume.

The nav-train shared HTTP client admits only currently active scene keys, rotates
fairly between them and bounds pending tasks per scene (`per_scene_pending`,
default 8). Its full candidate set must match the pristine frozen schedule quotas;
one controller owns that schedule. Both R2R and free-instruction adapters use the
same admission code. nav-train retains input selection and labeling; it owns no
simulator processes or scene replacement operations.

The sequential 16-versus-32 experiment uses all 61 registered train scenes,
16 attempts each (976 per phase), identical recorded positions/random headings,
64 model slots and a global prefetch limit of 128. Cold loads and final draining
are included. This is a short per-scene budget experiment, not a claim of steady
100k production throughput. Default mainline service settings are unchanged.

Both configurations have now completed: 16 workers took 718.61s (694 audited
completions), 32 workers took 550.91s (687 audited completions), each from 976
attempts. All 976 four-view image sets matched across phases. Complete-output
throughput increased from 3,477/h to 4,489/h (29.1%). The 32-worker phase retained
at least 10.41 GiB free on each collection GPU and 679.1 GiB available host memory
at sampled resource points. Both phases loaded 61 scenes plus one same-scene
rebuild following an invalid-depth recording; quota-driven replacements were
45 and 29 respectively. All completed PointNav labels passed independent checks.
Both owned experimental services were stopped; the existing mainline was retained.
[Full measurement and validation](evidence/rolling-workers-20261114.analysis.json).

## Production success quotas and start pools

Scheduled scenes can additionally set `success_goal <= attempts`. The service
counts audited completions separately from terminal attempts, and admits no more
in-flight claims than the remaining success goal. Rejections and cancellations
consume attempts without consuming successes. On reaching the goal, it drains
existing claims and binds the same worker to the next scene even if unused
attempt reserve remains. Exhausting attempts first is recorded as a shortfall,
never an automatic retry or budget expansion. Legacy attempt-only schedules keep
their behavior. nav-train marks unclaimed reserve candidates `not_needed`.

`sim-engine build-start-pool --config C --dataset D --camera K --owner O --output DIR`
prepares train-only start pools using
CPU PathFinder without creating a Simulator. It validates registered scenes and
freezes NavMesh hashes, seeds and samples. Allocation uses a per-scene minimum
plus eligible NavMesh area; sampling filters tiny islands and inadequate
clearance, with pose-cell/yaw-bin deduplication. The client only reads this pool.

The 100k-success production owner is `d7cf8d673fed49e68fa53b8ef469f1b2`, deployed
under `/data/nav_storage/sim-engine/deployments/free-production-d7cf8d673fed`.
Its 32-worker service on 18813 uses 61 scenes, a 100,000 total success goal and
300,000 finite reserve starts. Original 18812 service ownership is retained.
Process launch/shutdown belongs to its generated owner; no automatic restart.
[Frozen production plan](evidence/free-instruction-production-100k-20261114.plan.json).
Validation: 118 sim-engine tests, including bounded success admission and start
pool allocation/filtering, passed with Ruff.
