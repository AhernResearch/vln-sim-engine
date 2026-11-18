# 规划优化第一轮：框内计算、精确缓存与矩阵批处理

第一轮实现位于 sim-engine，两条采集主线共用。10 组真实冻结输入的配对 CPU
规划总耗时从 268.771 秒降至 100.088 秒，约快 2.69 倍。128 条输入的原框内
安全掩码、NavMesh 点、净空值和完整候选列表逐项一致。独立 HTTP 规划服务与
原生 Simulator 联调通过。该小规模、特意包含慢例的对照不代表生产吞吐提升倍数。

此前瓶颈证据见 [规划分析](planning-optimization-analysis-20261114.md)。

## 实现与兼容

- `collection/safe_image.py`：保留全图深度投影与法线计算，只对目标框及一像素
  halo 执行昂贵的逐点 NavMesh 检查。128 条输入的地面待检查点从 3,319,677
  降到 567,665，保留约 17.1%。
- `collection/region.py`：产物保持全图形状，ROI 产物额外保存 `computed_mask`；
  JSON 的 `schema_version: 2`、`computed_pixel_bounds_xyxy` 和
  `outside_computed_bounds: unknown_not_obstacle` 明确区分未计算区域。
  框外的 false/NaN 不能解释为障碍；需要全图时使用旧版本配置。
- `collection/query_cache.py`、`clearance.py`：为完整有向边检查缓存成功和失败。
  键包含完整净空配置与原始 float64 端点，不做量化、不交换端点。
  LRU 默认上限 65,536 项，任务结束清理；缓存命中仍检查取消/墙钟预算，
  并计入原检查消耗的逻辑查询数。抛异常或中断的检查不写入缓存。
- `collection/corridor.py`：每条参考走廊只计算一次线段起点、方向与分母，
  保留精确点查询缓存。平滑阶段将多个点到参考线段的距离矩阵化；
  每块最多 32,768 个点段对，不构建无限制的 N×M 张量。
  每批最多提前计算 32 个查询点，逻辑计数仍在首个失败处短路。

`collection.planning_profile_version` 新任务默认 `collection-v3`。
`collection-v2` 保留完整掩码和旧计算路径；请求与配置显式记录版本。
HTTP 服务之间仍要求相同源码 digest，选择 v2 不意味着可以混用不同源码节点。
目标候选顺序、A* 扩展顺序、净空门槛、路径验证和控制器均保持原逻辑；
可见范围内仍无 8 米路径长度上限。

## 配对 CPU 对照

使用相同输入、资产、运行库与预算，旧版取自当前生产冻结部署；新版取自
独立冻结目录。两版交替先后、每次仅运行一个 CPU 子进程；宿主机同时有生产负载，
未调用模型、创建 Simulator 或发出运动命令。

| 输入 | 原版秒 | 新版秒 | 原版/新版 | 两版结果 |
| --- | ---: | ---: | ---: | --- |
| 普通成功 | 2.207 | 0.381 | 5.80 | feasible |
| 慢成功 | 46.865 | 15.394 | 3.04 | feasible |
| 慢剔除 | 50.091 | 18.145 | 2.76 | infeasible |
| 查询/展开预算长尾 | 92.281 | 27.001 | 3.42 | budget_exhausted |
| s01-p00034 | 2.883 | 1.988 | 1.45 | feasible |
| s02-p00111 | 5.095 | 0.818 | 6.23 | feasible |
| s04-p00116 | 3.061 | 0.594 | 5.16 | feasible |
| s05-p00082 | 2.899 | 1.182 | 2.45 | feasible |
| 可见目标、约 30 米测地路径 | 62.950 | 34.433 | 1.83 | budget_exhausted |
| R2R 预检 | 0.438 | 0.152 | 2.87 | feasible |
| 合计 | 268.771 | 100.088 | 2.69 | |

非预算耗尽例的完整 prepared 计划、候选与选点报告完全一致；自由框的框内掩码、
点与净空哈希一致。预算耗尽例校验框内几何和候选，并保留实际预算终态；
更快耗尽计算预算不能当作成功规划。ROI 省略框外查询，因此全任务查询计数允许下降，
边缓存自身仍按原逻辑成本计数。

[配对结果](evidence/planning-optimization-round1-20261115/summary.json)、
[输入列表](evidence/planning-optimization-round1-20261115/cases.json)、
[128 条 ROI 对照](evidence/planning-optimization-round1-20261115/roi-corpus.json)。

## 回归与原生联调

sim-engine 全量 145 项通过，包含 15 项优化回归：有向失败缓存、配置隔离、
取消与预算、LRU 上限、浮点边界和退化线段、批次短路、边界框与亚像素框、
v2/v3 隔离、带障碍的完整搜索和平滑与原版结果一致。
sim-engine Ruff 通过；nav-train 1147 项通过、12 项跳过；harness 311 项及
4 个 subtests 通过，其 Ruff、mypy、navgpt 未定义名检查与前端构建通过。
没有更改用户界面流程。

原生联调使用新 owner、新采样起点、2 个 CPU worker、2 个 Simulator worker
和独立 HTTP 端口，规划节点仅加载 NavMesh 资产。8 条固定框自由任务中 6 条
完成录制与审计、2 条移动前剔除；1 条 R2R 完成审计；参考几何查询完成且不增加
仿真操作。超短墙钟预算探针以 budget_exhausted 返回，无仿真操作。
未调用模型、未做语义质量评审，没有自动重试。

新版配对基准及原生联调冻结的 66 个源文件分别与工作区哈希一致。
临时服务与 4 个 worker 均已退出。当前生产服务身份保持，32 个 Simulator
worker、24 个 CPU worker 存活，仍运行原冻结版本；本轮没有热更新生产。

[原生结果](evidence/planning-optimization-round1-20261115/native-validation.json)、
[源码清单](evidence/planning-optimization-round1-20261115/source-manifest.json)、
[验证汇总](evidence/planning-optimization-round1-20261115/validation.json)、
[退出及生产健康](evidence/planning-optimization-round1-20261115/shutdown-and-production.json)。

完整远端证据分别保存在：

- `/data/nav_storage/sim-engine/diagnostics/planning-opt-20261114-6e6451b7d63a`
- `/data/nav_storage/sim-engine/deployments/planning-opt-native-3a653d4acb55`

## 下一轮：减少重复搜索，再扩大批处理

优先验证同一起点、多个邻近候选目标共享可通行图和搜索结果。当前不同候选会
反复经过相同拐角，精确缓存已经降低单次成本，共享搜索有机会进一步减少展开。
复用必须保持 NavMesh、楼层、净空配置一致；不同目标的参考走廊约束不能直接
互相替代。候选验收仍按原排名进行，R2R 仍按有序引导点验收。

矩阵化继续适合独立数值计算：点到线段、候选到支持区域距离、批量粗筛。
先观察本轮之后的热点，再决定扩展哪项。A* 堆的动态展开与拓扑依赖不适合
直接改成整批矩阵；NavMesh 的 snap/try_step 目前也是标量接口，NumPy
广播不会自动消除这部分原生调用开销。若需要批量原生接口，应放在几何适配层，
保持规划对象与 Simulator 解耦，不为此向 nav-train 增加仿真实现。

算法层可另立版本验证粗路径引导的局部细化与失败拐角修补。它们可能改变搜索顺序、
路径和保留率，需同时比较普通、不可达、长可见路径和 R2R，不能仅比较成功例速度。
上线前还需独立测量端到端吞吐、长尾、保留率和 CPU/模型/审计的负载转移。
