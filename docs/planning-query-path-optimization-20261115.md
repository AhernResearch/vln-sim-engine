# v5 主线：任务级查询缓存与原生路径检查

默认主线更新为 `collection-v5`。自由指令和 R2R 共用任务级精确几何查询缓存、
有界批量接口、C++ 扫掠边检查与顺序 fairing。现有 freespace 算法、显示和模型
输入没有改动；控制器、目标选择、原框约束及录制合同保持原逻辑。

相对上一版 C++ v4，10 组配对规划在临时内存盘保存中间产物时，墙钟总耗时
54.67 → 34.89 秒，约快 1.57 倍，CPU 总时间 54.40 → 34.47 秒。
同一组输入直接在生产存储写产物时，CPU 总时间 52.38 → 36.17 秒，约快
1.45 倍；墙钟 54.89 → 48.68 秒，约快 1.13 倍。宿主机有生产负载，等待与
调度影响墙钟。本轮没有测量新版本生产吞吐。

当前正在运行的生产任务继续使用原冻结部署。本轮没有热更新、停止或重放它。

随后完成的真实模型、录制、审计及 PointNav 端到端对照见
[端到端耗时报告](../../nav-train/docs/planning-e2e-performance-20261115.md)。
其基线为正在生产的优化前 v2，区别于本文 v4 对 v5 的规划内核对照。

## 实现边界

| 对象 / 文件 | 职责 |
| --- | --- |
| `planning/queries.py::NavMeshQueries` | 绑定一个不可变 NavMesh，持有任务级缓存，提供 `snap_many / clearance_many / island_many` |
| `planning/_geometry.h` | 共享稳定 ABI 基元、float32 字节键、LRU 与缓存类型 |
| `planning/_queries.cpp` | 原生缓存与有序批量循环；命中时直接返回，未命中调用现有 NavMesh 绑定 |
| `planning/_astar.cpp` | 沿用原 A* 顺序，直接使用共享查询缓存，减少 Python 回调 |
| `planning/_path.cpp` | 顺序扫掠边检查、原位 fairing，不改变更新依赖 |
| `planning/native.py` | Python/C++ 适配，保留采样数和数值边界判定 |
| `planning/geometry.py::NavMeshContext` | 多个净空规划器共用缓存；任务前后统一清理 |

缓存键使用实际传入 NavMesh 的 float32 三坐标字节，不做额外空间量化，保留
正负零区别；clearance 键还包含搜索半径的 double 字节。每个缓存绑定同一个
PathFinder 对象，不跨场景、任务或可变 NavMesh 使用。
默认 snap、clearance、island 各有最多 65,536 个 LRU 项，命中/未命中/淘汰计数
随计划回执保存为 `geometry_queries`，用于后续定位。

非有限 snap 返回和负 island 等静态几何失败可以缓存；异常、无效 clearance、
取消和预算耗尽不作为几何失败证书保存。批量接口先校验输入，分块默认 256 点，
原生单批上限 4096；有效批次按输入顺序处理，首个运行错误或取消向上传播。
底层仍使用现有标量 NavMesh 绑定，不新增逐点远程 RPC。

clearance 每个逻辑点查询仍计数一次，含缓存命中；snap/island 保留原调用方的
预算计数语义并检查取消。完整有向边证书仍由现有 `edge_memo` 按原成本计数。
结果回执先复制统计，再清理缓存，后续任务不能读到上一任务的证书。

## 原生路径核与等价性

扫掠边检查继续检查每个原采样点的 snap、水平/高度偏差、净空和无滑动 step。
采样数仍由原 NumPy norm 与 ceil 公式计算，原生核按原算术顺序生成点。
距离恰好接近 1e-4 / 1e-3 阈值时，回调原 NumPy 判定，覆盖浮点边界。
编译继续关闭 fast-math 和 FMA 合并。

fairing 保留逐点原位更新：后一个点使用已更新的前一个点。端点与有序引导点
继续固定；独立线段距离的有界矩阵计算和可见性 shortcut 保持已有实现。
没有通过一次全矩阵更新改成另一种平滑算法，没有调整控制器或净空门槛。

v5 显式启用 `query_optimizations` 与 `native_path_optimizations`；v4 仍是原有
C++ A*、旧路径后处理，v2/v3 显式请求也保留。默认值仍统一来自
`CURRENT_PLANNING_PROFILE`。扩展要求 `KERNEL_API=2`，旧二进制在服务启动时
被拒绝，必须重新构建；没有自动回退或移动重试。

构建包含三个 `.cpp` 和共享 `.h`，头文件加入运行实现哈希与 wheel。原生联调
和配对基准的二进制哈希相同；联调的 73 个源码/资源文件与工作区一致。
配对基准冻结后只将默认常量从 v4 改为 v5，基准中的新版任务本就显式指定 v5。

## 配对测量

两轮使用相同 10 组冻结输入、资产与预算，逐例交替先后，每次只运行一个 CPU
子进程，不创建 Simulator、调用模型或执行运动。基准使用上一轮 v4 冻结源码
作为基线，新版单独冻结。内存盘轮只将规划中间产物放入 `/dev/shm`；计时后复制
至持久证据目录并清除临时目录。输入和运行库仍来自原有位置。

| 样例 | v4 秒，内存盘 | v5 秒，内存盘 | 两版结果 |
| --- | ---: | ---: | --- |
| 普通成功 | 0.502 | 0.525 | feasible |
| 慢成功 | 7.264 | 3.969 | feasible |
| 慢剔除 | 5.554 | 2.485 | infeasible |
| 计算预算长尾 | 14.725 | 8.355 | budget_exhausted |
| s01-p00034 | 1.212 | 0.805 | feasible |
| s02-p00111 | 0.672 | 0.647 | feasible |
| s04-p00116 | 0.522 | 0.486 | feasible |
| s05-p00082 | 1.121 | 0.792 | feasible |
| 可见目标、约 30 米测地路径 | 22.971 | 16.561 | budget_exhausted |
| R2R 预检 | 0.131 | 0.262 | feasible |

两轮各 10 组对照均通过。完成规划的完整 prepared、候选和选点报告相同，框内
掩码、NavMesh 点和净空哈希相同。所有样例终态、逻辑查询数和展开数相同。
预算耗尽例仍为预算耗尽。短例存在等待波动：该 R2R 样例 CPU 为
0.116 → 0.099 秒，墙钟没有同样下降。

慢成功样例的全阶段 snap 缓存命中 176,899 次，未命中 20,544 次，命中率约
89.6%，无淘汰；这里包含 A*、边检查和平滑，不是此前只对 A* 输入统计的 99.1%。

[普通存储对照](evidence/planning-query-path-20261115/paired-summary.json)、
[内存盘对照](evidence/planning-query-path-20261115/tmpfs-summary.json)、
[CPU/墙钟/查询统计](evidence/planning-query-path-20261115/tmpfs-timings.json)、
[构建信息](evidence/planning-query-path-20261115/build.json)。

## 回归、原生联调与运行状态

sim-engine 192 项通过，新增查询/路径回归 33 项，覆盖 float32 精确键、正负零、
可修改返回值隔离、跨 NavMesh 拒绝、半径隔离、负 island、几何失败与运行错误、
LRU、任务清理、预算、批次中途取消、采样整数边界、阈值 ULP、有序/重复引导点、
完整路径与查询计数对照。Ruff 通过。
nav-train 1147 项通过、12 项跳过；harness 311 项及 4 个 subtests 通过，其 Ruff、
mypy、navgpt 未定义名检查与前端构建通过。

新的 mandatory API-2 wheel 在独立目录构建、解包、加载成功，包含全部 C++
源码与头文件，默认版本为 v5；使编译器失败时没有生成 wheel。
默认 v5 原生联调使用新 owner、新采样起点、两个独立 HTTP 服务、2 个规划 worker
和 2 个 Simulator worker。8 条自由任务中 6 条完成录制与审计、2 条移动前剔除；
R2R 审计通过；参考几何查询与预算探针没有增加仿真操作。没有模型语义审核。

临时服务、四个 worker 和内存盘暂存目录均已退出/清除；原生产身份、32 个
Simulator worker、24 个 CPU worker 保持。没有新增 freespace 展示、二维替代
目标算法或模型输入变更。现有安全掩码继续保留。

[验证汇总](evidence/planning-query-path-20261115/validation.json)、
[默认配置原生联调](evidence/planning-query-path-20261115/native-validation.json)、
[打包验证](evidence/planning-query-path-20261115/package-validation.json)、
[进程与生产健康](evidence/planning-query-path-20261115/shutdown-and-production.json)。

## 后续观察

新慢例剖析中，snap 适配回调从 150,880 次降到全阶段 20,544 次；预算/取消检查
约 62 万次，step 适配约 3.5 万次，文件 fsync 也出现可测等待。
剖析累计时间相互嵌套且含工具开销，不能直接相加。
下一轮可评估预算检查的原生适配、step 数据往返，以及共享候选搜索图来减少
重复展开；控制器拐角策略需另立版本评估。freespace 工作按用户要求暂缓。

[优化后热点](evidence/planning-query-path-20261115/post-profile.json)。完整远端证据位于
`/data/nav_storage/sim-engine/diagnostics/planning-query-path-0ef5079977d7`，默认配置
原生联调位于 `/data/nav_storage/sim-engine/deployments/planning-query-mainline-79220a039e83`。
