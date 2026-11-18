# 独立 CPU 规划与异步采集

采集主线的全局寻路、框内落点筛选、净空检查、平滑和只读控制器预演，
由独立 CPU 规划池执行。等待规划期间，仿真 worker 可以继续处理其他任务的
观测和实际执行。两条主线共用同一规划协议和实现；nav-train 不管理仿真或规划进程。

```mermaid
flowchart LR
  C[CollectionService 调度与持久化] --> O[SimulationWorker 观测]
  O --> M[nav-train 模型画框]
  M --> P[PlanningClient]
  C --> P
  P --> L[本机 PlanningService]
  P --> R[远程 PlanningService]
  L --> E[PlanningEngine / NavMeshContext]
  R --> E
  E --> X[冻结计划与 ExecutionLease]
  X --> S[SimulationWorker 执行与录制]
  S --> A[独立 Audit 队列]
  A --> F[不可变回执 / pending_review]
```

自由指令：观测 → 模型画框 → 规划 → 执行 → 审计。
R2R chunk：规划与预演 → 观测 → 执行 → 审计；训练侧模型标注可与执行重叠。
`reference_geometry` 直接走 CPU 规划服务，不占用仿真操作。
模型公开输入仍只有四图，以及 R2R 的指令；私有目标、路径和验收指标不进入模型请求。

## 工程边界

| 对象 / 文件 | 职责 |
| --- | --- |
| `planning/contracts.py` | 版本化 PlanRequest、PlanningProfile、PlanningBudget、校验 Blob、独立 PlannerSettings |
| `planning/tasks.py` | 采集与规划共用的 TaskRequest / RegionRequest 输入契约 |
| `navigation/` | 纯几何：净空 A*、走廊、引导、可见区域、路径跟踪、NavMesh 查询缓存与 C++ 内核；不持有仿真器、不写文件 |
| `planning/geometry.py` 的 NavMeshContext | 只加载 PathFinder 与 NavMesh；不需要 GLB、EpisodeStore 或 Simulator |
| `navigation/budget.py` 的 PlanningContext | 每个任务共享总墙钟、查询与 A* 展开预算，及时取消，任务结束释放查询缓存 |
| `planning/engine.py` 的 PlanningEngine | 自由框、R2R 和参考测量；复用现有选点、净空与跟踪算法 |
| `planning/service.py` 的 PlanningService / PlanningWorker | 有界在途任务、CPU spawn 进程、独立尝试、硬超时、状态与回执 |
| `planning/client.py` | 本机/HTTP 两种客户端；部署位置不改变业务输入输出；验证和安装产物 |
| `planning/assets.py` | 从显式训练注册表导出独立 NavMesh 资产包，按内容去重 |
| `collection/server.py` 的 CollectionService | 场景窗口、任务阶段、反压、取消、一次性执行准入与审计调度 |
| `collection/backend.py` 的 CollectionActor | 观测、恢复状态、执行已冻结路径与实际录制；不进行候选搜索或全局规划 |
| `collection/audit.py` | 仿真 worker 释放后检查录制与原框终点；独立有界线程池 |

4Hz 路径跟踪和单步碰撞/净空保护仍紧贴实际仿真状态；它们不重新选择目标或执行全局 A*。
原有离线 SceneBackend 通过共享 NavMeshContext 保持兼容，服务主线不再调用它的规划方法。
迁移来源的 license、NOTICE 和 provenance 统一保存在 `sim_engine/licenses/`。
依赖方向为 `core ← navigation ← planning ← collection`，由 `tests/test_layering.py` 强制。

## 本机、远程与混合部署

默认不配置 `collection.planning_urls`：采集服务在同机启动独立 CPU 进程池。
可先全部部署在 A100，无需额外监听端口。默认规划进程4、审计线程4，均可配置。

```json
{
  "planning_workers": 24,
  "planning_capacity": 64,
  "planning_scene_cache_entries": 4,
  "planning_timeout_s": 120,
  "planning_max_expansions": 80000,
  "planning_max_queries": 2000000,
  "audit_workers": 4
}
```

上例是可配置的 A100 并发起点，不是已验证的生产吞吐配置。
`capacity` 控制采集在途任务；`planning_capacity` 对新领取任务施加反压。
已领取任务的模型目标先持久化，再等待规划，不因下游拥塞丢弃模型产物。
采集客户端使用服务公开的有效容量，且保留逐场景 pending 限制。

独立部署：先在拥有训练资产注册表的机器导出 CPU 资产包：

```bash
sim-engine export-planner-assets --config collection.json --output /new/owned/planner-bundle
```

把该目录和同一冻结版本的 sim-engine 分发到规划机器。`planner.json` 中的资产和
jobs 路径相对于配置文件解析，整包可移动。修改进程数等配置后，启动：

```bash
sim-engine serve-planner --config /new/owned/planner-bundle/planner.json --host 127.0.0.1 --port 18814
```

规划机器需要相同 Habitat / NumPy 运行版本和 CPU 内存，不需要 GPU、模型或场景 GLB。
远程部署配置可达的监听地址，再把采集配置设为：

```json
{
  "planning_urls": ["http://planner-host:18814"],
  "planning_workers": 8,
  "planning_capacity": 64
}
```

配置 URL 后，采集端不再启动内嵌规划池；`planning_workers` 控制并发调度槽数，
远端真正的 CPU 进程数由各自 `planner.json` 决定。
混合部署就是在 A100 和 Ubuntu 各启动独立规划服务，把两个 URL 都放入列表。
客户端轮转 endpoint，仅在确定没有领取任务的429/503情况下改投；网络结果不明确时
固定查询原 endpoint 的同一尝试，不重新提交。
Ubuntu 的依赖安装、实际资源预算与跨机吞吐尚未验证；本次原生验证是在 A100 上
通过独立 HTTP 服务完成，不据此宣称跨机提速。

## 协议、预算与故障语义

- `POST /v1/plans`：完整任务级 RPC；不是逐 NavMesh 查询的远程调用。
- `GET /v1/plans/{attempt_id}`：排队/计算进度或终态；`DELETE` 请求取消。
- `GET /v1/health`：服务身份、注册资产、并发容量、队列、worker PID 与查询/展开进度。
- 输入使用 dataset/scene ID + NavMesh SHA256。自由框任务只传选中视角的深度、
  相机标定、sensor pose、冻结起点和框；不传 RGB 或整条录制。Blob 有类型、大小、内容哈希。
- 结果保留 input digest、task/attempt ID、源码 digest、运行库版本、计划、预演和校验产物。
  HTTP 调度端要求源码版本一致；执行前再次核验 NavMesh、Habitat/NumPy 版本及冻结计划。
- `feasible`、`infeasible`、`budget_exhausted`、`cancelled`、`worker_failed` 分别记录。
  计算总预算贯穿所有候选及其 A*/平滑/预演；不被每次候选重置，不当作“几何不可达”。
  采集终态增加 `planning_budget_exhausted`，nav-train 保留原规划 outcome。
- 当前默认120秒、8万展开、200万查询是初始预算，仍需大规模校准保留率。
  这限制计算成本，没有恢复8米轨迹长度上限；实际执行预算仍由路径推导。
- 协作取消进入内部循环；原生调用不返回时，父进程在墙钟预算+grace后终止该 CPU 进程。
  取消标志与进度使用无锁共享内存，避免强制终止留下锁。失败尝试先封存，
  下一次新任务可替换 CPU 容量；原任务不会重放。
- 规划服务重启后，历史未完成尝试报告不确定失败，不续算或覆盖旧回执。
  网络/服务错误独立收尾，不触发仿真运动。确定未领取的容量拒绝可有界等待。
- 安装计划后创建 `execution-lease.json`。运动前冻结 `selected-goal.json`，验证观测恢复，
  create-only 写入 `movement-attempted.json` 消耗凭据。失联或执行不确定不自动重试。
- 审计读取实际保存的终帧，而非依赖已被下一个任务改变的 live Simulator 状态。
- 场景重绑定等待规划、执行和审计全部终态。单个仿真 worker 丢失时，健康场景可继续，
  该场景未领取的备用候选保留为未开始；历史生产外部 supervisor 不自动更新或重启。
- nav-train 中断只为已提交候选封存结果，不为几十万个未领取候选创建目录。

## 验证

新任务默认使用 `collection.planning_profile_version: collection-v5`，共用 C++ 受约束
A*、任务级精确几何查询缓存、批量查询接口、原生边检查与顺序 fairing。
框内地面检查、有向边证书和有界走廊矩阵计算继续保留。采集配置与规划请求共用
`CURRENT_PLANNING_PROFILE`，C++ 扩展为构建必需项，服务启动检查内核。
`collection-v2` / `collection-v3` / `collection-v4` 的显式请求继续保留。
ROI 掩码需结合 `computed_mask` 读取，框外未计算区域不代表障碍。
v5 内核分工：`navigation/queries.py::NavMeshQueries` 绑定一个不可变 NavMesh，持有任务级
snap / clearance / island 精确缓存（float32 坐标字节为键，各 65,536 项 LRU，统计随回执保存为
`geometry_queries`，任务结束清理）；`_astar.cpp` 沿用原 A* 顺序并直接使用该缓存；
`_path.cpp` 做顺序扫掠边检查与原位 fairing。相对 v4，10 组配对规划 CPU 总时间
54.40 → 34.47 秒；未测量生产吞吐。各轮优化的配对测量保留在
`evidence/planning-{optimization-round1,native-astar,mainline-review,query-path}-2026111*/`。
以下是拆池时的历史验证。

CPU 故障注入覆盖：进程退出、硬超时、预算共享、运行/排队取消、容量拒绝、
相同尝试的幂等查询、重启不重放、错误哈希/路径、丢失提交响应与过期结果拒绝。
采集测试证明：同一场景的慢 CPU 规划期间，仿真 worker 仍可完成另一个任务观测；
预算耗尽后 worker 保持存活，后续正常执行可继续。

A100 原生证据和完整结果见 [独立规划联调](evidence/planning-split-20261114.native.json)。
测试使用新 owner、新输入，两个仿真 worker、两个 CPU worker、独立 HTTP 服务、
只有61个物理 NavMesh 的资产目录。最终8条固定框测试中5条通过录制/终点审计、
3条移动前剔除；通过的记录零碰撞、终点均在原框内，最大目标距离约0.075m。
前两轮各有1条无效深度记录保留为不确定证据，没有重放。
1条 R2R 通过，参考几何查询完成且不增加仿真操作。
原900秒长尾输入以新的只读规划尝试在约2.1秒预算耗尽返回，不构造 Simulator。

测试未调用模型、未做语义审核，因此不代表生成质量或吞吐评估。
测试服务和子进程已关闭，原18812服务身份及16个worker保持；十万条旧生产没有重启。

最终部署快照的64个源文件与当前实现逐文件哈希一致；原长尾任务的框内候选结果
与旧记录完全相同。sim-engine 130项、nav-train 1147项（12项跳过）、harness 311项
及4个subtests通过；相应Ruff、mypy和前端构建通过。
[汇总验证与进程关闭证据](evidence/planning-split-20261114.validation.json)。
