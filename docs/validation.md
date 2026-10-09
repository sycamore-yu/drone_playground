# 验证与验收证据

## 主分支合并检查：2026-10-09

已将 PPO 独立实验与控制模型改动整合。合并后的三组针对性验证分别为
**10 项 PPO 测试、17 项控制/CLI/迁移测试和 23 项验收收集器测试，共 50 项通过**。
Ruff 检查通过，56 个 Python 文件符合格式要求。该组检查验证合并点的关键调用，
没有重新运行完整 CPU、GPU 收敛和原生飞行验收。
日志和合并来源见[实施记录](plans/control-models-20261009.md)。

## 控制模型工作树：2026-10-09

独立工作树完成 21 个不重复的针对性测试：19 项模型/控制/延迟/恢复/CLI 检查，
以及 4 项迁移检查（其中 2 项与前者重叠）。Ruff 和 56 个 Python 文件格式检查通过。
sdist 和 wheel 构建成功；安装后检查通过 9 个新增模块、66 种配置及来源许可文件。
实际 acados 完成一次 0.1 s 悬停闭环，回合为 SUCCESS；样本数未达到正式验收门槛。

完整日志和执行范围见[实施记录](plans/control-models-20261009.md)。本次按用户要求
未跑全量回归、GPU 收敛矩阵或原生 S6；主目录训练和历史产物没有被修改。

## 重构前的完整验收记录

主分支此前的完整锁定环境 CPU 测试通过 **239 项**，耗时 **813.26 s**；
其中包含验收收集器的 23 项测试，覆盖初始化成本追溯、运行中状态和续训配置核对。
更早一次完整运行通过 **223 项**，耗时 **1466.77 s**。
两次完整运行都发生在本次分支合并之前，不作为合并后整套回归的证据。
完整 Ruff 规则与全部 43 个 Python 文件的格式检查通过。训练质量另按
[18 单元验收表](../results/acceptance/current.md)判断，不由单元测试替代。

EGO 和 SUPER 均已通过真实飞行 S6；Tracking、Racing、Navigation 和动态场景的
RScope 原生窗口验证已完成。Learning 的全部 18 单元已通过三种子收敛与独立冻结验收。

## 精确锁环境

Pixi 安装完成后，CPU 验证实际核验如下版本。Crazyflow 安装元数据的 Git commit 为 `62a3146a408c5fd5d0c2451de22e29bd74a71b18`，与官方仓库锁定一致。JAX 仅报告 `CpuDevice(id=0)`。

| 组件 | 实际版本 |
|---|---|
| Python | 3.12.15 |
| Crazyflow | 0.3.2，官方 Git commit 如上 |
| JAX / JAXlib | 0.11.2 / 0.11.2 |
| Flax / Optax | 0.12.10 / 0.2.8 |
| MuJoCo / MJX | 3.15.0 / 3.15.0 |
| Hydra | 1.3.7 |
| NumPy / SciPy | 2.5.3 / 1.18.1 |
| RScope | 0.0.8 |
| grpcio / protobuf | 1.84.0 / 7.36.2 |
| pytest / Ruff / build | 9.1.1 / 0.16.10 / 1.6.1 |

版本、导入路径、Git 来源及锁文件 SHA-256 记录在 `tmp/agents/package-locked-versions.json`。GPU 训练使用同一锁定环境，逐次环境身份保存在各运行的 `run.json`。

宿主 shell 的 `PYTHONPATH` 含 `/opt/ros/jazzy`，导致原始 `JAX_PLATFORMS=cpu pixi run pytest -q` 在收集前加载外部 `launch_testing` 插件并报缺少 `lark`。项目 Pixi 激活配置现已清除该外部路径；完整测试正常收集，未禁用 pytest 插件自动加载。

## 检查结果

| 检查 | 结果 | 证据 |
|---|---|---|
| 完整 CPU pytest | 239 passed，813.26 s | [`full-suite-asymmetric-final.log`](../tmp/ros-planner-refactor/full-suite-asymmetric-final.log) |
| 验收收集器专项 | 23 passed，全部包含在最终完整测试中 | [`collector-resume-green.log`](../tmp/ros-planner-refactor/collector-resume-green.log) |
| Ruff 与格式 | E/F/I/UP/B/SIM/N/D/RUF 通过；43 files already formatted | `pixi run lint`、`pixi run ruff format --check src tests tools` |
| 安装与 wheel | 真实 wheel 的五项测试通过，包含在完整 CPU 运行中 | [`test_packaging.py`](../tests/test_packaging.py) |
| 发布包构建 | 包含 ROS、评测、恢复修复和进度/传感器/特权 Critic | [`release-build-asymmetric.log`](../tmp/ros-planner-refactor/release-build-asymmetric.log) |
| 更新后的配方打包 | 独立 venv：19 通用配方 × 3 算法，加 2 个 PPO 专用特权配方，共 59 个组合通过 | [`release-install-check-asymmetric.log`](../tmp/ros-planner-refactor/release-install-check-asymmetric.log) |
| 原生 EGO / SUPER | 两者均完成 12 静态 + 12 动态回合并通过 S6 | [原生飞行记录](ros_planner.md) |
| RScope 原生窗口 | 三任务及动态场景的实际绘制、位姿核对、正常关闭通过 | [S7 验证](replay-validation.md) |
| 正式 Learning | 18/18 单元通过；36 次训练、5,400 个冻结回合 | [逐种子验收结果](../results/acceptance/current.md) |

```bash
pixi install --locked
JAX_PLATFORMS=cpu pixi run pytest -q
JAX_PLATFORMS=cpu pixi run pytest -q tests/test_acceptance_report.py
pixi run ruff check src tests tools
pixi run ruff format --check src tests tools
pixi run build
```

完整运行中的 78 条警告来自 Optax 内部的 `optax.global_norm` 弃用提示。
较早完整运行曾为 165 passed、1 failed，唯一失败为测试中未导入的 `np`；
修复后该文件 33 项重测通过，后续上述完整运行也通过。原始日志
`tmp/agents/package-cpu-pixi-clean.log`、`package-cli-fixed.log` 保持原样。

当前构建产物及 [SHA-256 清单](../dist/SHA256SUMS)：

- [`drone_playground-0.1.0-py3-none-any.whl`](../dist/drone_playground-0.1.0-py3-none-any.whl)，
  可直接安装的 Python wheel。
- [`drone_playground-0.1.0.tar.gz`](../dist/drone_playground-0.1.0.tar.gz)，
  包含源码与实验资料的源码归档。

## Wheel 的验证范围

`tests/test_packaging.py` 构建真实 wheel，以 `--no-deps` 安装到新的 venv；依赖来自当前精确锁环境的 system site-packages。子进程清除 `PYTHONPATH`，模块/场景/Actor 检查额外使用 `-I`。测试断言项目模块和场景 XML 必须来自安装目录，因此不能靠源码检出的资产回退通过。

- wheel 必须包含本仓库的场景 XML、纹理与 Hydra YAML。
- 公共 Simulation、Learning、CLI 模块从 wheel 导入。
- empty、Navigation8、LSY racing 共 10 个场景完成 MuJoCo 编译、几何身份及两个时刻的位置计算；racing 同时验证相对纹理资源可加载。
- state、depth、lidar 三种 Actor 完成参数初始化和前向计算，检查动作、记忆、辅助速度的形状与有限值。
- 实际安装的 `drone-playground` console script 使用 `--cfg job --resolve`，组合 Tracking、Racing、Navigation Depth、Navigation LiDAR 与 PPO/APG/SHAC 共 12 种配置。

较早安装验证所保留 wheel 的 SHA-256 为 `43605f65dd9096c338ddb661648dec39d5ee1cc59b87c8511d176573420d9076`。该轮完整 suite 与单独打包测试构建的 wheel 哈希相同；保留文件位于 `tmp/agents/package-wheel-locked/drone_playground-0.1.0-py3-none-any.whl`。这是包与配置运行证据，不是训练或完整飞行成绩。

首轮诊断发现 `configs/__init__.py` 缺失，导致 wheel console script 报 `Primary config module 'drone_playground.configs' not found`。补充后，重新构建的 wheel 通过诊断和精确锁环境测试。初始 `cli.py:416` 的 Ruff `E501` 也已修复。

## 验收对应关系

| 规格 | 实际证据 | 范围 |
|---|---|---|
| S1 | 锁定 Pixi、真实 wheel 安装、资源与 Hydra entrypoint 检查 | 独立 venv 共享锁环境依赖；wheel 从安装目录运行 |
| S2 | `test_forward_matches_official_crazyflow` | 同模型、初态、命令和子步，状态误差阈值 2e-6 |
| S3 | 三任务实际闭环、Navigation8 与 LSY 场景、任务事件测试 | 成功率见逐回合评测记录 |
| S4 | 原生 MuJoCo 射线对照、遮挡、动态位姿、时钟、完整设备测量测试 | 包括 1280×720 Depth 与完整 20,000 点 Mid-360；训练 Depth 使用声明的 64×48 模式 |
| S5 | 冻结 Policy 及真实 Planner→Controller→Crazyflow 运行 | 接口与任务规则共用，原生规划只接收许可测量 |
| S6 | EGO 与 SUPER 各 24 次真实 C++/ROS1 闭环 | 各静态主场景均有安全到达；全部动态失败保留 |
| S7 | 实际 RScope 原生窗口、截图、逐帧数据核对及正常退出 | 使用 Xvfb OpenGL 窗口，未声称人工桌面观看 |
| S8 | 配置、几何/权重/源码 hash、种子、失败分母、计时与回放记录 | 历史环境差异和可用证据边界在报告中明示 |
| L1—L3 | 三算法真实更新、共享网络、两种感知、控制转换、损失与导数专项 | CPU 回归和实际 GPU 训练均有记录 |
| L4 | reset、终止/截断、GRU、精确续训、场景切换恢复、C5 与冻结分区测试 | 检查点和冻结 benchmark 使用独立种子区间 |
| L5、C1—C5 | [逐种子验收表](../results/acceptance/current.md) | 全部 18 单元通过；每种子独立选模与冻结评测 |
| L6、C6 | 更新/交互/耗时/GPU 采样、恢复归档、[吞吐实验](performance.md)、递归初始化成本 | GPU 为并行作业下的整卡采样；未知历史预训练成本保留为未知 |

首发闭环配方使用零观测送达延迟和零动作延迟；非零值会明确报错。
传感器按自身时钟在物理步间插值，包含 500 Hz 物理与 30 Hz 相机的非整数周期。
当前方法执行频率须整除物理频率。Mid-360 的非重复扫描为已声明的合成近似，
不声称复制厂商专有扫描序列。完整设备采样、策略降采样和显示抽样分别记录。

早期诊断曾使用只读 Python 3.13/JAX 0.9 依赖，项目实现始终来自本仓库。
它们未计作正式种子；Racing 共同初始化来源仍保留该环境身份。正式三种子训练
使用锁定的 Python 3.12/JAX 0.11。历史参数用于初始化，不能据此声称从零训练。
