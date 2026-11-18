# 智元、智园1、智园2：GS 接入验证

2026-11-16 在 A100 GPU 2 的独立 Habitat-GS 0.3.3 环境中验证，三个场景均通过
sim-engine 原生仿真、RGB-D、NavMesh、专家短路径采集及 HTTP 会话测试。
原始目录中选取的九个输入文件在测试后重新校验 SHA256，均未改变。

| 输入目录 | scene_id | 测试 LOD | 高斯数量 | NavMesh 面积 | 专家执行 | 碰撞 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `/home/holo/智元-output` | `zhiyuan` | 3 | 1,123,730 | 201.33 m² | 9 步 | 0 |
| `/home/holo/智园1-output` | `zhiyuan1` | 4 | 681,577 | 565.33 m² | 13 步 | 0 |
| `/home/holo/智园2-output` | `zhiyuan2` | 3 | 1,169,951 | 224.93 m² | 14 步 | 0 |

智园1 没有 `point_cloud_3.ply`，显式选择了现有的 LOD 4。此结论适用于表中的
资产版本；没有验证完整分辨率或其他 LOD。NavMesh 面积是烘焙结果，不代表实测建筑面积。

## 验证内容与结果

- 使用现有导入器，将选定的 GS PLY、`mesh/mesh.ply`、`render/assets/poses.json`
  组成最小 ZIP 输入；沿用米制及 `(x,z,-y)` 坐标转换。写入各自独立的 scene_id
  后冻结 manifest，再烘焙导航资产。没有把原始目录重命名或覆盖。
- NavMesh 使用代理半径 0.18 m、高度 1.5 m；扫描位姿支持的起点候选分别为
  920、1,545、1,169 个。最终起点额外 NavMesh 净空分别为 1.411、1.746、1.739 m。
- 480×270、水平视场角 120°、相机高度 0.88 m、俯仰角 0°；RGB 非空，
  深度有限，首帧正深度覆盖率分别为 99.866%、100%、100%。四方向观察保持机体位置不变。
- 相同场景缓存复位、原生 NavMesh 查询与 A* 内核探针通过；附近采样的实际短路径
  经专家预演、执行、RGB-D 记录及记录对齐校验通过。
- 三次执行均正常 `expert_stop`，最终目标误差分别为 0.0023、0.0737、0.0691 m。
- HTTP 测试使用真实 spawned worker：创建会话、获取帧、执行动作、请求去重、
  历史深度不变、结束会话、释放资源均通过。
- 本地回归 **212 passed**；Ruff 和 `git diff --check` 通过。

本次没有模型推理；HTTP 的结束请求是测试脚本发出的，不能视为模型产生的 STOP。
GS 深度仍为 `gaussian_unvalidated`，正深度覆盖率不等于几何精度，采集数据仍待审核。
此测试证明所选资产能够接入并执行短路径，不是全场景导航成功率评测。

## 测试发现并修正的问题

1. 智园1 的扫描坐标含整数 `0`，直接传 Python 混合数值列表会被 Magnum 的
   `snap_point` 绑定拒绝。`scan_bake.select_start` 现统一传入 `float32` 数组。
2. 原烘焙器取第一个可用起点，智元和智园2 的该起点靠近 NavMesh 边界，
   未满足规划器默认额外净空要求。现从扫描支持的候选中选择额外净空最大的位置，
   并在导航 receipt 中记录选择规则、净空和查询上限；未放宽规划约束。

两项行为均有回归测试。首轮失败日志保留在
[initial](evidence/gs-three-scans-20261116/initial/)，第二轮独立输出保留完整成功记录。
本次专用探针从起点附近采样专家目标，避免在大场景中全局随机抽样不到短路径；
其余执行及验收断言沿用 `deployment/probe_backend.py`。

## 产物与复用

- [验证汇总](evidence/gs-three-scans-20261116/summary.json)
- [渲染预览](evidence/gs-three-scans-20261116/render-previews.jpg)
- [智元原生记录](evidence/gs-three-scans-20261116/zhiyuan-native.json)、
  [智园1原生记录](evidence/gs-three-scans-20261116/zhiyuan1-native.json)、
  [智园2原生记录](evidence/gs-three-scans-20261116/zhiyuan2-native.json)
- 本机完整场景包：`outputs/gs-three-scans-20261116-v2/scenes/{scene_id}/scene.json`。
  相应 `{scene_id}.local.json` 使用本机绝对路径；原 `{scene_id}.json` 使用 A100 路径。
- A100 独立部署：`/data/nav_storage/sim-engine/deployments/gs-three-scans-20261116-v2`，
  包含冻结源码、三个服务配置、场景包、原生探针、RGB-D 与短路径记录。

以下命令可在 A100 启动已验证的智元配置（端口需空闲）：

```bash
SCAN_DEPLOY=/data/nav_storage/sim-engine/deployments/gs-three-scans-20261116-v2
GS_RUNTIME=/data/nav_storage/open-nav-harness-tools/habitat-gs/3066ae942703477c859921c1a94c55c4
PYTHONPATH="$SCAN_DEPLOY/src:$SCAN_DEPLOY/deps" \
  "$GS_RUNTIME/venv/bin/python" -m sim_engine serve-sim \
  --config "$SCAN_DEPLOY/zhiyuan.json" --port 18852
```

测试进程已退出，未切换或重启已有生产服务。
