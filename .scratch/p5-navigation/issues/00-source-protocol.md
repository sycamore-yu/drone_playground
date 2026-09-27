# P5-00：来源与协议清单

Type: task
Status: resolved
Blocked by: none
Engineering: passed
Experiment: not-applicable
Quality: not-applicable
Owner: main-session
Session: p5-main
Run: n/a

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

锁定必要依赖及几何/运动参考，记录已确认的40秒/0.5米/机体碰撞协议；交付明确的传感器、机体几何、场景参数和资源测量计划。

## 验收

来源提交/许可可追溯；场景范围为几何与运动；列出所有待测技术事实和需要用户决定的研究参数。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

来源、许可、锁定版本与已确认协议见
[P5来源清单](../../../docs/verification/p5-source-inventory.json)。

- 依赖锁定：Crazyflow 36f584d114d9d331f0cee0fe4b9066f821c0fbfd（MIT，editable）；jax 0.9.2、
  brax 0.14.2、mujoco/mjx 3.14.0、flax 0.12.6、rscope 0.0.8 实测导入。
- 新增依赖决策：mujoco-lidar 0.3.5（MIT）已装入 pixi 环境并通过 JAX 后端实测导入；
  **尚未写入 pyproject.toml/pixi.lock**，在 P5-03 冻结前必须补上。
- D435i 资产：mujoco_menagerie realsense_d435i，Apache-2.0，仅作外观与安装参考，不 vendored。
- 参考身份：SANDO 3a4450dcc5a8ed5ca825c9da7966e2642be091de（BSD-3-Clause，只读）；
  MIGHTY 163940216d056ef3e86c8d69e42adebb9a82db0f（BSD-3-Clause）；
  D.VA 01b2be4986a0851a952aa860afb4a5958e6676e2（MIT）；ego-planner
  bfda51284c8c1b476043255a8145ef925a3778a5（GPL-3.0）。
- **许可风险**：reference_repos/SUPER（2ad3419c）仓库内没有 LICENSE 文件，GitHub API 的
  license 字段为 null，默认属于保留全部权利。结论：只以未修改的外部进程/容器运行，
  不复制、不 vendored、不再授权其源码。
- 场景来源语义已按文件逐条记录（森林圆柱 1.0-1.5 m / 6.0 m、密度 0.05/0.10/0.20、
  trefoil 方程与 GLOBAL_TIME_SCALE=7.9442501919、0.65 动态比例、0.8 m 立方体、
  collision/clearance 精确公式）。
- 本机只有 ROS 2 jazzy；ROS1 Noetic 需要容器，属 P5-06。
- 尚待用户决定的三项（场景密度/速度具体值、代表策略质量门槛、实测吞吐对应的墙钟上限）
  已在本清单 `pending_user_decisions` 中列出；工程侧先给出实测表再请求确认。
