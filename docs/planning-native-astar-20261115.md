# C++ A* 选型与第二轮规划优化

本报告记录 C++ 后端尚为可选项时的验证。后续已将 v4 设为默认，并将原生构建
改为必需项，见 [主线切换与全链路复查](planning-mainline-and-pipeline-review-20261115.md)。

本轮在 sim-engine 增加可选的 C++ 搜索后端 `collection-v4`。同一冻结源码、
相同输入与计算预算下，10 组配对规划的墙钟总耗时为 105.31 → 62.29 秒，
约快 1.69 倍；CPU 总时间为 93.17 → 52.97 秒，约快 1.76 倍。
慢成功样例约快 2.03 倍、慢剔除约快 2.30 倍。短任务存在波动，没有稳定的普遍增益。
当前默认仍为 `collection-v3`，生产任务保持原冻结版本。

## C++ 库调查与选择

| 候选 | 官方接口/源码 | 与当前任务的关系 |
| --- | --- | --- |
| Boost.Graph | [A* 文档](https://www.boost.org/latest/libs/graph/doc/html/graph/algorithms/shortest_paths/astar_search.html)，[许可证](https://www.boost.org/LICENSE_1_0.txt) | 支持隐式图及 visitor，是以后通用图搜索框架的优先候选；需要适配动态邻居、预算及同分排序 |
| MicroPather | [项目](https://github.com/leethomason/MicroPather)，[头文件](https://raw.githubusercontent.com/leethomason/MicroPather/master/micropather.h) | 小型通用 A*；使用浮点成本，节点与排序语义需要重新验证 |
| stlastar | [项目](https://github.com/justinhj/astar-algorithm-cpp)，[头文件](https://raw.githubusercontent.com/justinhj/astar-algorithm-cpp/master/stlastar.h) | 模板状态接口可适配自定义图；头文件的 g/h/f 为 float，与当前 double 成本不同 |
| Recast/Detour | [项目](https://github.com/recastnavigation/recastnavigation)，[Habitat PathFinder 实现](https://github.com/facebookresearch/habitat-sim/blob/main/src/esp/nav/PathFinder.cpp) | Habitat 的基础 NavMesh 查询已经使用它；额外净空与参考走廊约束仍需上层实现 |

本轮采用约 225 行自有 C++ 搜索内核，沿用现有搜索语义，便于将语言开销与算法变化
分别评估。没有复制或引入以上第三方库源码；现有许可证与 provenance 保持。
实测当前 Python 的 `heapq.heappush.__module__` 为 `_heapq`：堆本身已经是 C 实现，
收益来自下沉邻居构造、点距离、走廊约束、状态表与搜索循环。

## 实现边界

- `planning/_astar.cpp`：隐式三维搜索状态 `(ix, iz, floor_bin)`，保留八邻居顺序、
  double 成本、递增 serial 同分排序、Python round 的 ties-to-even 规则、
  过期队列项与展开计数规则、接近终点后的安全边检查。
- `planning/native.py`：独立适配层。NavMesh snap 和既有 swept-edge 检查通过
  回调执行；搜索循环与每次走廊查询仍检查共享预算和取消。
  原生异常向上传播，没有自动重试或搜索回退。
- 参考路径已满足净空时继续使用原快速路径；后续平滑、完整安全验证、候选顺序和
  控制器保持现有实现。保留原框内选点与可见范围内无 8 米路径上限的规则。
- `planning/implementation.py`：采集与规划端统一冻结 `.py`、`.cpp` 和原生二进制
  哈希，HTTP 校验覆盖编译产物。同源码不同二进制也会被识别为不同实现。
- nav-train 没有增加仿真或 C++ 实现；自由指令与 R2R 共用该后端。

配置为 `collection.planning_profile_version: collection-v4` 才会启用原生搜索。
`collection-v2`、`collection-v3` 继续使用 Python 搜索，默认仍为 v3。
显式选择 v4 时，如果扩展未编译或不能加载，会报错，不静默改用 Python。

`setup.py` 将扩展作为可选构建项，目标为 CPython 3.10 的 Limited API，
不依赖 NumPy C API 或 Habitat C++ 头文件。其兼容范围遵循
[CPython Stable ABI 说明](https://docs.python.org/3/c-api/stable.html)。
本轮验证了 Linux 的 `cp310-abi3-linux_x86_64` wheel 构建、源码与二进制入包，
以及包内二进制在 Python 3.10 上导入和运行。目标机仍需要兼容的系统 C++ 运行库，
规划服务也仍需要原有 Habitat/NavMesh 运行环境。

新部署可用 `uv build --wheel` 构建；使用 C++17 编译器和 Python 开发头文件。
可选构建失败时基础包仍可安装，启用 v4 前须在目标环境执行以下加载检查：

```bash
python -c 'from sim_engine.planning.native import require_native; print(require_native().__file__)'
```

采集与规划服务应分发相同冻结包及二进制，不修改正在运行的部署。

## 配对结果与限制

两版使用同一源码快照，分别指定 v3/v4；逐例顺序交替，每次一个 CPU 子进程。
宿主机有生产负载，部分阶段与独立原生联调重叠。对照不创建 Simulator、不调用模型、
不执行运动。每次为新只读规划尝试，不覆盖旧生产产物。

| 输入 | Python v3 秒 | C++ v4 秒 | 两版结果 |
| --- | ---: | ---: | --- |
| 普通成功 | 0.431 | 0.395 | feasible |
| 慢成功 | 14.554 | 7.167 | feasible |
| 慢剔除 | 18.011 | 7.833 | infeasible |
| 计算预算长尾 | 30.163 | 13.523 | budget_exhausted |
| s01-p00034 | 5.134 | 1.556 | feasible |
| s02-p00111 | 1.007 | 1.818 | feasible |
| s04-p00116 | 0.582 | 0.803 | feasible |
| s05-p00082 | 1.483 | 2.651 | feasible |
| 可见目标、约 30 米测地路径 | 30.808 | 26.380 | budget_exhausted |
| R2R 预检 | 3.141 | 0.166 | feasible |

所有样例的查询数、展开数和终态一致；非预算耗尽样例的完整计划、候选与选点报告
一致，框内几何哈希一致。预算耗尽样例仍是预算耗尽，不能算作成功规划。
本轮观察到的一致性不等于对全部浮点边界或全部场景的证明。

普通成功、s02-p00111、s04-p00116 和该 R2R 样例没有进入 A*。R2R 的 CPU
时间仅 0.160 → 0.134 秒，其墙钟大幅下降主要是等待波动，不能当作原生算法增益。
短例的两版墙钟与 CPU 时间均完整保留。总体结果属于特意包含长尾的诊断集，
没有按生产分布加权，也未测量新版本端到端吞吐。

[配对报告](evidence/planning-native-astar-20261115/summary.json)、
[CPU/墙钟与计数](evidence/planning-native-astar-20261115/timings-and-progress.json)、
[构建信息](evidence/planning-native-astar-20261115/build.json)。

## 验证与下一处热点

sim-engine 158 项通过，其中原生 A* 回归 13 项，包含绕障、正负楼层舍入、
有序与重复引导点、路径/展开/查询对照、共享预算、取消、回调异常和二进制身份。
Ruff 通过；nav-train 1147 项通过、12 项跳过；harness 311 项及 4 个 subtests
通过，其 Ruff、mypy、navgpt 未定义名检查与前端构建通过。

独立 HTTP 原生联调使用 2 个 CPU worker 和 2 个 Simulator worker。
8 条新采样自由任务中 6 条完成录制与审计、2 条移动前剔除；1 条 R2R 审计通过；
参考几何查询没有增加仿真操作。没有模型调用、语义审核或自动重试。
测试服务及四个 worker 已退出；生产服务身份未变，32 个 Simulator worker 和
24 个 CPU worker 存活。

69 个运行源码/资源文件与本地工作区一致，目标二进制哈希另行记录；
配对对照和原生联调使用相同源码与二进制快照。构建生成的 egg-info 属于包元数据，
不计入运行实现文件匹配。

[原生联调](evidence/planning-native-astar-20261115/native-validation.json)、
[验证汇总](evidence/planning-native-astar-20261115/validation.json)、
[退出与生产健康](evidence/planning-native-astar-20261115/shutdown-and-production.json)。

慢成功样例的下一轮剖析中，原生搜索函数自身约 0.31 秒；snap 适配调用累计约
1.51 秒，未命中的边检查累计约 3.09 秒。这些是含剖析开销的数据，累计时间可能
嵌套，不能相加当成阶段占比。下一步优先考虑共享候选的 snap/局部图查询、
批量 NavMesh 适配，以及边采样和平滑的数值核，继续减少标量往返。
更换另一个 A* 堆实现的空间已小于这些部分。

[Python 热点](evidence/planning-native-astar-20261115/python-hotspots.json)、
[原生后端热点](evidence/planning-native-astar-20261115/native-hotspots.json)。

完整远端配对证据位于
`/data/nav_storage/sim-engine/diagnostics/planning-native-search-26d0b461a2e7/paired-06de65ce1c39`；
原生联调位于 `/data/nav_storage/sim-engine/deployments/planning-native-astar-204a53a3cc82`。
