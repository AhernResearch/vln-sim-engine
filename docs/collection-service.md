# Training collection service

The current deployment includes bounded sampling from safe pixels inside the
original model box. See [visible-floor candidate sampling](visible-region-mask-sampling.md)
for the current service identity and sparse-support regression evidence. Older
deployment sections below retain their historical identities and measurements.

For scene-grouped production experiments, the service also supports optional
[fixed scene bindings](fixed-scene-collection.md). The default configuration
remains unchanged; bindings are explicitly frozen into a separate deployment.

`sim-engine serve-collection` owns the persistent Habitat processes, scene cache,
four-view RGB-D observation, target grounding, planning, expert rollout and raw
recording. Both nav-train pipelines submit whole tasks to this service. Model
requests, instruction interpretation, PointNav labels and training admission stay
in nav-train. No model or policy is loaded by the collection service.

```bash
python -m sim_engine serve-collection \
  --config examples/collection.train.v1.json --port 18812
```

The example registers only R2R train scenes, an 88 cm level 480×270 HFOV120 camera,
4 Hz velocity physics and 16 workers, four per GPU on 0/2/3/4, with one cached
scene per worker. Use a frozen copy with a
new output root for each deployment. It requires the configured training assets
and Habitat environment. The policy field is an unused settings placeholder.
The privileged API binds to loopback; it belongs on a trusted collection host.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/v1/health` | Service owner, capacity, source hashes, asset hashes and profiles |
| POST | `/v1/collections` | Claim a task and enqueue observation |
| POST | `/v1/collections/status` | Batch status for up to 256 task IDs |
| GET | `/v1/collections/{id}` | Status or immutable terminal receipt |
| POST | `/v1/collections/{id}/target` | Freeze a free-task view and bounding box |
| DELETE | `/v1/collections/{id}` | Cancel queued work or signal its owned worker |

Requests contain a 64-character hexadecimal request ID, 32-character owner,
registered dataset/camera/scene, seed and start pose (world xyz, xyzw rotation).
`short_instructions` rejects assigned instructions, goals and reference paths.
`r2r_chunks` requires a local instruction, private sparse guides and goal, with
bounded start perturbation. `reference_geometry` measures ordered NavMesh legs
for train chunk preparation without rendering or movement. Scenes must be in the
pinned, explicitly allowed training dataset; arbitrary asset paths and executable
arguments are not accepted.

Each worker is spawned once. A bounded LRU retains scene backends, and same-scene
resets retain Habitat when camera, physics and GPU configuration match. A single
render captures RGB and depth for each of the four views. Observation completion
releases the worker: a free task waiting for a model occupies a capacity slot and
files, but no worker. Workers prefer queued executions and warm scenes. An idle
worker with the scene cached takes priority over a cold worker; when cached
workers are busy, other workers can load that scene for additional concurrency. R2R
execution starts independently of its instruction annotation. Before execution,
the saved start pose is restored and its front RGB must exactly match the saved
observation; prefetching cannot silently execute from another task's pose.

Free-task geometry keeps the current visible-floor box policy: choose and refine
inside the original box, require visible support on the current floor, plan and
preview, then execute once. There is no 8 m distance cap. The step budget scales
with path length and turning time. The selected world goal is written to
`selected-goal.json` before movement; the actual stop does not replace it.

Raw RGB-D, states, commands, camera metadata and receipts are written on the
service host. The client needs the same absolute shared filesystem paths and
links these immutable artifacts into its own run. HTTP carries task descriptions,
observation handles, target boxes and receipts, without per-frame round trips.
Only the public four images (plus the assigned instruction for R2R) may reach a
model. Private geometry and acceptance metrics are separate server artifacts.

Service and worker identities are generated at startup and persisted alongside
source/configuration provenance. Repeated identical submissions read the existing
claim; changed requests or targets conflict. Final receipts are create-only.
Timeout, process loss and shutdown preserve leases and attempted-movement evidence.
Workers are never automatically restarted, and movement is never replayed. An
incomplete claim from a previous service lifetime remains uncertain. Clients that
disappear while waiting for a target leave a capacity slot until explicit
cancellation; there is no automatic expiry or recovery.

Capacity is global to this server. Model slots remain client-side; concurrent
nav-train runs must divide their shared model endpoint budget. The example's
capacity and worker count are configuration defaults, not a measured throughput
recommendation. All collected data remains `pending_review` / `audit_only`.

The migrated collection implementation preserves its source file hashes in
`sim_engine/collection/provenance.json` and inherited notices in `licenses/`.
It does not import nav-train or the neighboring harness. See
`tests/test_collection_server.py` for spawned-process queue, claim, cancellation,
crash and no-replay checks, and `tests/test_collection_geometry.py` for rendering,
start restoration and Habitat cache invalidation checks.

## Validated deployment, 2026-11-14

A100 native integration completed two free tasks and one R2R task through the
existing model endpoints: all `completed_audit`, zero collisions, complete RGB-D
recordings, final distances 3.08/1.65/0.57 cm. Both free-task stops remained in their
original boxes and retained PointNav labels. One worker loaded the scene once
across both pipelines and a read-only reference query. The free batch took 14.14 s
and the R2R task 4.68 s; this is functional evidence, without a paired throughput
comparison. The initial R2R development manifest had the free-task step setting
and was rejected before admission; a corrected manifest used a new owner/run.

The initial validated implementation was deployed on A100 loopback port 18812 with four
workers on GPUs 0/2/3/4, capacity128, and the example's `train-v1` shared root.
Service owner: `d31e84004b294d1899037309e9399738`; frozen deployment:
`/data/nav_storage/sim-engine/deployments/collection-service-9e46a1f4f130`.
The separately owned development service was stopped after validating its owner,
PID, process start ticks and configuration path. Full production was not started.

Tests: 89 sim-engine tests passed; nav-train 1140 passed/12 skipped; harness 311
passed plus4 subtests. Relevant lint/type checks and the frontend build passed.
Evidence: [native integration](evidence/collection-service-20261114.native.json),
[service identity](evidence/collection-service-20261114.health.json),
[validation and deployment](evidence/collection-service-20261114.validation.json).

## Four cached scenes per worker

The training example now sets `scene_cache_entries=4`. Four workers can retain up
to 16 scene/camera instances, loaded on demand. These may include duplicate scenes
across workers. The fifth distinct entry in a worker evicts and closes its least
recently used backend. Repeated hits keep the other cached scenes in the scheduler's
affinity list. Health reports `scene_cache_entries_per_worker` and
`maximum_cached_scene_instances`.

An observation-only A100 probe loaded four training scenes in one process, then
revisited all four. Process RSS reached 4.31 GiB; GPU0 memory increased by 4.11 GiB.
All four views matched their original RGB pixels on every revisit, with four total
scene loads and four hits. First observations took 4.38–8.86 s; cached observations
took 0.258–0.326 s. This measures observation rendering and file writes, not complete
trajectory throughput. The machine had 736 GiB available host memory and roughly
22–25 GB free on the four simulator GPUs before the probe. Scene sizes vary; these
measurements cover the four current mainline source scenes, not every train scene.
The probe issued no movement or model requests and released its resources on exit.

Previous four-cache deployment (superseded by the 16-worker configuration below):
`/data/nav_storage/sim-engine/deployments/collection-cache4-dcc923dc5e55`.
The idle previous service was stopped only after owner/PID/start-time/configuration
checks. The shared root and port remain `train-v1` and18812, so prepared nav-train
clients keep working. Service owner: `414a7a3553794001862db82a94cb7a3f`.
Tests: 91 passed, Ruff passed. Evidence:
[native measurements](evidence/collection-cache4-20261114.native.json),
[baseline](evidence/collection-cache4-20261114.baseline.json),
[current service health](evidence/collection-cache4-20261114.health.json).

## Current deployment: 16 workers, one cached scene each

The training example and live service now use 16 independent workers, four on each
of GPUs 0/2/3/4. Each retains one scene/camera backend. Maximum resident instances
remain16, while up to16 simulation operations can run concurrently. Scenes are
loaded on demand and are not permanently assigned to workers. The collection CLI
sets OpenBLAS/OMP/MKL/NumExpr thread limits to1 before importing NumPy, preventing
a separate CPU-sized numerical thread pool in every process. Each loaded worker
had two OS threads during the native test.

Native validation submitted two batches of16 observation-only tasks, covering the
four current mainline source scenes and 16 initial headings. All16 workers rendered
concurrently. Loaded worker RSS totaled18.63 GiB; added GPU memory on 0/2/3/4 was
2.01/3.47/4.81/4.17 GiB, with17.52–22.33 GiB still free per participating GPU. The
host has112 logical CPUs and about736 GiB available memory at the initial check.
These are sampled loaded-state readings, not worst-case bounds across all scenes.

The first observation batch took25.67 s including scene loads. The repeated batch
took0.529 s, all16 jobs hit a cached scene, and all four RGB views matched their
original pixels for every job. An initial16-worker check exposed cold workers
stealing tasks from idle cached workers; the scheduler now defers those claims.
The earlier diagnostic and fixed measurements are preserved separately. This
tests rendering, recording and dispatch, not full trajectory/model throughput.
All 32 final observation tasks were explicitly cancelled before movement; there
were no model requests or training exports.

Current frozen deployment:
`/data/nav_storage/sim-engine/deployments/collection-workers16-1a0dddb328aa`.
Service owner: `ddf3f404c6b64875822dbcfbfd110e67`. Port18812, shared root `train-v1`,
queue capacity128 and nav-train client configuration are unchanged. Both the
separate profiling service and idle previous production service were stopped
after ownership/PID/start-time/configuration checks. Tests:93 passed, Ruff passed;
the four nav-train client tests passed. Evidence:
[native observation batches](evidence/collection-workers16-20261114.native.json),
[validation summary](evidence/collection-workers16-20261114.validation.json),
[service health](evidence/collection-workers16-20261114.health.json),
[earlier affinity diagnostic](evidence/collection-workers16-20261114.before-affinity.json).

## Concurrent end-to-end validation

The live 16-worker service also completed a concurrent nav-train test: 16 free tasks
and 16 R2R candidates, with 16 model slots per client and 32 combined. R2R completed
16/16; free tasks completed 13/16 with three exclusions before movement. Total wall
time was 31.53 s. All 32 model calls completed, with zero collisions, crashes or
timeouts. The 29 executed trajectories retained 684 validated frames; 13 free tasks
retained 252 PointNav frame labels.

Sampled simulator-operation concurrency peaked at 16 and model request intervals
at 18. Actual expert rollout intervals peaked at 6, with 13.35 s overlapping model
requests. This verifies cross-pipeline overlap, not saturated 16-rollout throughput
or a speedup over a serial baseline. The sample uses four cached training scenes.
Two excluded boxes had no safe floor pixels; one had only six and the 13×13 grid
missed them. Whether those pixels could support a feasible plan is untested; the
mainline selection policy was unchanged. All three were rejected before movement.

The service remained healthy with 16 live workers and no active tasks afterwards.
Raw run:
`/data/nav_storage/nav-train/collections/concurrent-e2e-20261114-9357fe50243b`.
Evidence: [per-task audit](evidence/collection-concurrent-e2e-20261114.audit.json),
[controller result](evidence/collection-concurrent-e2e-20261114.controller.json),
[sampled monitoring](evidence/collection-concurrent-e2e-20261114.monitor.jsonl).

## 独立规划主线（2026-11-14）

采集服务现已把自由框选点、R2R预处理及参考几何移入独立CPU规划池，录制校验
使用独立审计队列。默认全部在同机运行，也可通过 `planning_urls` 接入远程或混合
服务。部署、预算、执行租约和故障语义见[规划服务说明](planning-service.md)。
旧冻结部署不会随工作区源码自动改变，旧生产不自动续跑。

## HTTP循环与证据I/O隔离（2026-11-15）

在A100采集中观察到HTTP服务主线程等待 `wait_transaction_locked`、
`jbd2_log_wait_commit`、`rq_qos_wait` 和文件页读取；同步领取、回执及内嵌规划池
的证据写入会使磁盘等待阻塞整个HTTP事件循环。旧版故障注入中，卡住领取写盘
会同时让健康检查超时，复现了这条故障传播路径；这不能证明历史每次超时都源于同一原因。

运行期证据读写现交给独立的 `AsyncDisk`：每服务4个线程、最多16个已提交操作，
队列满时异步等待。采集与内嵌规划服务都使用该机制，健康检查只读内存。
`evidence_io` 返回待处理数量、最老操作及等待时间、完成数和最长耗时，便于区别
磁盘积压与进程退出。构造阶段的身份和worker启动记录仍在HTTP就绪前冻结。

领取、冻结目标、状态变更、终态回执和场景交接保持串行所有权；证据落盘后才派发
下一阶段。领取落盘中返回 `claiming`，不把半写入目录误报为旧生命周期失败。
终态回执与场景切换完成前不向客户端发布终态成功。取消协程时等待已提交的I/O完成，
避免后台迟到写入越过关闭封存；没有请求重发、运动重试或自动续跑。

完整回归196项及Ruff通过。旧版在两项阻塞领取的健康检查测试中超时，修复版通过；
A100独立运行四项故障测试通过，覆盖采集领取、终态回执、规划领取及重复取消期间
的写入顺序，未调用模型或仿真GPU。

部署为 `/data/nav_storage/sim-engine/deployments/free-v5-io-6ca6ed773bdb`。
通过独立测试后对旧生产进行所有权/PID启动时间/argv核验并发送一次受控中断，
保留全部已完成与未决运动；新生产仍须重新核对逐场景剩余配额和来源哈希。
磁盘永久不可用仍会触发停止，线程隔离不等于存储故障自动恢复。

[内核等待采样](evidence/collection-io-isolation-20261115/disk-journal-blocking-diagnosis.json)、
[本地验证](evidence/collection-io-isolation-20261115/validation.json)、
[A100故障注入验证](evidence/collection-io-isolation-20261115/remote-validation.json)。

修复版已恢复实际生产。启动后连续30次、每两秒一次的独立健康读取全部成功，
最长115.6ms；主线程采样为运行、epoll或futex等待，未在这些采样点出现日志提交等待。
这是启动窗口观测，不是全量运行稳定性保证。首批5条、105帧通过完整录制、冻结目标、
逐帧PointNav、规划摘要及执行租约核验，均无碰撞且停止点在原框内。
启动验证快照确认32仿真、32规划存活，累计46914条；新增证据I/O最长耗时约0.84秒。

[启动验证](evidence/collection-io-isolation-20261115/startup-validation.json)、
[健康读取采样](evidence/collection-io-isolation-20261115/startup-health-samples.json)、
[首批录制和PointNav核验](evidence/collection-io-isolation-20261115/launch-audit.json)、
[继承核验摘要](evidence/collection-io-isolation-20261115/reconciliation-summary.json)。

后续13:01北京时间核查发现，生产在09:44因inode耗尽停止。数据盘尚有约472GiB
可用字节，但45774848个inode全部用尽，控制器状态、异常摘要和服务回执均无法新建。
最后客户端进度报告累计92167条，终态逐条核对尚未完成，未续跑。
此前1642次守护监控没有健康读取超时；可用字节保留线漏检inode，需要增加独立
的inode保留线。原 `status.json` 的 `running` 已过期，不能据此判断任务仍在运行。

[停止检查证据](evidence/collection-io-isolation-20261115/stopped-inodes-20261115.json)。
