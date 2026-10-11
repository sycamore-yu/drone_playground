# 验证与验收证据

## Sensor / Replay P1 现代化：2026-10-11

**受控版本边界：**将 2026-10-10 已实现的精度修复、求交评估及传感器文档
单独提交为 `0ea6203`，再迁移策略预处理、精简测量状态并复用旧版 Crazyflie
模型和规划叠加。两个版本均只通过 `Scene.raycast()` 进行正式射线求交；
Crazyflow 动力学、扫描表、采集时钟、设备配置与 Actor 均没有改动。

**输入逐项对照：**在 `tmp/sensor-modernization/snapshot.py` 保存
`0ea6203` 版本的 `baseline.npz`，修改后同位姿、场景 D01、帧时间、
扫描相位运行生成 `candidate.npz`。共有 **19 组**仍属于测量、
Actor 观测或配置身份的张量 / JSON 字节 **完全一致**，包括 Depth 64×48
及 12×16 预处理、LiDAR 原始值与 32 点策略抽样、
重建的 `point_times` 和配置规格。新 `Measurement` 不再持久保存
`directions_body` 与 `times`：快照下方向仅用于生成命中点，逐点时间
等于广播的 `acquisition_time`，因此这是内部状态缩减，不是修改观测模型。

按 float32 数据及 bool 有效掩码计算，**32 环境 × 20,000 射线**时，
单份 `Measurement` 的逻辑缓冲从 **21,120,256 B**
降至 **10,880,256 B**，减少 **10,240,000 B（9.766 MiB，48.48%）**。
这不是 GPU 峰值显存数值；采集延迟缓冲使用相同的紧凑结构，
完整 GPU 显存和速度需要按独立工作负载另行解释。

**RTX 4090 配对更新：**对 `0ea6203` 固定源码快照与本次 P1 代码分别
启动独立进程，以相同场景、Actor 算法 APG、种子、batch、horizon 和传感器
执行真实训练；首次 XLA 编译不计入稳定更新中位数。
当时用户的其它任务同时占用 GPU（监测期间约 95%–100% 利用率），
因此这是**竞争负载下的功能及无明显退化检查**，非可直接用于精确
吞吐排名的独占 GPU 基准。

| 完整 APG 更新 | `0ea6203` 基线中位数 | P1 中位数 | 变化 |
|---|---:|---:|---:|
| D01 LiDAR，32 环境 × 32 步，20,000 rays | 1.67822 s | 1.69729 s | +1.14% 时间 |
| S01 Depth，128 环境 × 32 步，64×48 | 1.29005 s | 1.22067 s | −5.38% 时间 |

两组更新都返回非零的 Actor 梯度并实际修改网络参数。
数据在 `tmp/sensor-modernization/gpu_{baseline,candidate}_{lidar,depth}.json`。
注意进程前后 `nvidia-smi` 的显存数字包含并发任务，不能直接归因于
此次 PyTree 精简；对于显存节省，仅上述单份状态的数学大小可以确认。

**测试与冻结：**最终代码的完整 CPU `pytest -q tests` 为
**282 passed**，覆盖 MuJoCo 原生几何及深度参考、
物理/扫描采集时钟、延迟/局部 reset、六种感知 PPO/APG/SHAC
真实参数更新、训练 checkpoint 保存/恢复与无修改继续训练、完整 RScope
原生数据解码和时间对照。日志：
`tmp/sensor-modernization/all-cpu-tests.log`。
最终修改后的回放/采集/规划接口另运行 **40 passed**：
`tmp/sensor-modernization/final-targeted-tests.log`。

**六种 GPU 算法更新：**Depth/LiDAR × PPO/APG/SHAC 均以
2 个环境、2 步展开执行真正的 CUDA 更新（LiDAR 每帧保持 20,000 射线），
首次编译后再次更新；六组 Actor 梯度有限且非零、网络参数真实变化、
传感器观测有效。结果保存在
`tmp/sensor-modernization/gpu_new_{depth,lidar}_{ppo,apg,shac}.json`，
完整控制台记录为 `gpu-all-six.log`。
小批量更新主要验证 GPU 代码路径，不能证明训练收敛或性能加速。

对原训练库中的 Depth/APG Step 480、LiDAR/APG Step 3680
各用 **8 场景 × 25 回合**执行新的冻结推理（未更新权重）：

| 场景 | Depth/APG | LiDAR/APG |
|---|---:|---:|
| S01 / S02 | 25/25、25/25 | 25/25、25/25 |
| S03 | 25/25 | 0/25 |
| D01 / D02 | 25/25、25/25 | 25/25、25/25 |
| D03 | 25/25 | 23/25 |
| S06 / D06 | 25/25、21/25 | 25/25、25/25 |
| **全场景总计** | **196/200** | **173/200** |

LiDAR D03 对早先保存的 **25/25** 存在差异，因此另把
`0ea6203` 的完整源文件恢复到隔离临时目录，
用 **相同 checkpoint 和 D03 对应的完整八场景种子**
`benchmark.seed_base=2000005` 重放：
`0ea6203` 基线 **23/25**，本次 P1 **23/25**。
说明**本次状态精简没有新增 D03 完成率损失**；
此前的历史全场景报告生成于上一版几何精度修复之前，
不能用作此次 P1 的数值等价基线。
当前 D03 恰好满足主场景规定的 **≥23/25**，
S03 的 LiDAR **0/25** 是该原训练权重已有的失败，
不应被报告为本次代码重构带来的解决或退化。

回放记录：
`tmp/sensor-modernization/frozen-all8.log`、
`tmp/sensor-modernization/d03-baseline-diff.log`；
对应结构化冻结报告在
`tmp/sensor-rendering/frozen_compact_{depth,lidar}_8/eval/001/report.json`。

**回放与打包：**`tests/test_replay.py` 的 28 项回归及
`tests/test_replay.py -k 'crazyflie_visual or planner_segments'`
检查实际 MJCF/RScope 模型编译与解码、Crazyflie 可见 Mesh 和记录四元数、
规划连续线段的 MuJoCo `MjvScene` 渲染及有效时间、绿色命中点、
动态障碍、Racing 贴图和重复写入拒绝。
复用旧仓库 Crazyflie 2.x 的 STL、原始 `replay.xml`、
`LICENSE` 和 `SOURCE.txt`，仅追加可见 Mesh，保留新版的
Freejoint/碰撞几何；支持序列化为独立 `.mj_unroll`。
Python wheel 通过隔离式构建，共包含 12 个机器人资源文件，
安装到干净的 target 后能从发布包导入并导出 RScope 回放。
规划轨迹从真实 `trajectory.positions` 生成，可使用
`received_time` / `valid_until` 控制逐帧可见性，不生成或推测 SFC。
回放仍使用 Runner **真实记录的点云命中**，不会二次求交。

**checkpoint：**冻结 Actor 推理归档的观测/传感器配置身份保持不变。
完整训练状态的 `ObservationState` JAX PyTree 变更是有意的
不兼容修改：迁移前的训练 checkpoint 不支持跨此结构恢复，
训练必须使用迁移后重新保存的状态。对应真实 PPO/APG/SHAC
更新、恢复及冻结评测结果以本节追加的最终测试记录为准。

## 官方 MJX-JAX ray 求交适配评估：2026-10-10

目标：对 `mujoco.mjx.ray()` (MuJoCo/MJX 3.15.0) 验证场景覆盖、
与官方 C `mujoco.mj_ray()` 的测距一致性、Crazyflow 场景状态接入和
批量 JAX/JIT 可用性。此次只在 `tmp/mjx-ray-eval/` 放置实验程序，
不修改项目运行时几何代码、主工作树训练作业或系统驱动。

**直接接入门禁失败：**

1. 官方 MJX-JAX `ray._RAY_FUNC` 未列 Cylinder / HField，只有
   Plane、Sphere、Capsule、Ellipsoid、Box、Mesh。八个固定 Navigation 场景
   全有圆柱（例如 S01 41/45、S02 81/85、S03 162/166），因此直接替换漏检障碍物。
2. S01、S02、D01 原样 `mjx.put_model()` 均因 `Cylinder–Box collisions not implemented`
   失败；只是测距也会触发 MJX 动力学模型转换时的碰撞类型检查。
3. 导航场景 `nmat=0`；临时构造的无材质单图元模型在官方
   `mjx.ray()` 上全部抛出 JAX 空 `mat_rgba` gather 异常。
   为隔离这项问题，给候选模型补一个未使用的材质资源后，
   8 个边界案例中 6 个与 C `mj_ray` 一致，两个 Cylinder 案例仍未命中。

**不进入正式代码的研究性绕行：** 仅在临时脚本中另外加载一份
`MjModel`，将候选模型的接触对禁用以便转换成 MJX Model，再添加
一个占位 `mat_rgba` 数组；`mjx.Data.replace(geom_xpos=Scene.positions(t))`
可以接入 Crazyflow/JAX 场景动态更新。D01 移动 Box 的 0、0.5、1.0 秒射线
距离为 0.600、0.632、0.665 米，三次都与本项目求交一致；
未更新 MJX Data 时均返回约 89.96 米的陈旧值，说明手动同步不可省略。

| 已绕行的 D01 小批量诊断 | 结果 | 说明 |
|---|---:|---|
| CPU，2 个世界 × 256 条射线 | 512/512 一致 | 这些方向未触发缺失的圆柱分支 |
| GPU，默认矩阵乘法精度 | 430/512 一致 | 与局部高精度几何计算出现数值差异 |
| GPU，`JAX_DEFAULT_MATMUL_PRECISION=highest` | 512/512 一致 | 仍仅覆盖已实现图元 |
| CPU 诊断中位数 MJX / 现有路径 | 1.058 / 3.944 ms | 非几何功能等价对照；禁止推导加速倍数 |
| GPU 高精度诊断中位数 MJX / 现有路径 | 2.342 / 2.402 ms | GPU 当时存在其它高负载训练，非独占性能验收 |

该批量测试证明 `jax.jit`、`vmap` 和动态 JAX 位姿接口能够组合；
并**不**证明 20,000 条 MID360 射线 × 多世界吞吐达标，
也不能掩盖所有固定场景都含缺失的 Cylinder。
本轮不替换 `Scene.raycast()`，不引入永久混合后端、
额外 MuJoCo 模型副本、占位材质和碰撞禁用的生产适配层。
依据和选型结论写入 [ADR-0012](adr/0012-sensor.md)。

临时复现入口与结构化结果：
`tmp/mjx-ray-eval/probe.py`、`probe.json`、
`dynamic_batch.py`、`dynamic_batch_cpu.json`、
`dynamic_batch_gpu.json`、
`dynamic_batch_gpu_highest.json`。
外部依据为已安装的官方 `mujoco/mjx/_src/ray.py` 和
`mujoco/mjx/_src/io.py`；后一文件对不支持碰撞的检查发生在
`put_model` 中。

## Sensor / render 功能分支：2026-10-10

分支 `refactor/sensor-rendering` 从 `e3949a5` 开始实施 ADR-0012/0013。
这是功能分支证据，尚未合并到 `v0.2`。现已补充 CUDA 实测和两种传感器的
冻结整回合评测；来源权重的既有成绩与本分支复测结果分别记录。

已接入 MuJoCo-LiDAR 软件包中的 MID360 扫描资源，实现每世界扫描相位、快照采集和精简延迟状态。
删除了合成扫描方向、逐射线位姿回调、长位姿历史和重复去畸变点云。
相机训练参数在 YAML 中显式声明，运行及 checkpoint 记录采集语义与资源身份。
动力学、Actor、训练损失和 `Scene` 的解析求交/距离公式没有改动；
在本分支 GPU 渲染参照测试中已对相关几何坐标变换指定局部最高矩阵精度，
不影响 Actor 的全局矩阵精度设置。

### 已完成的专项检查

CPU 回归分组覆盖 **279 项**：学习 40 项、CLI 45 项、其余 194 项。
首次非学习回归有 224 项通过、10 项 CLI 测试桩缺少新增 `sensor` 字段而失败；
补齐测试桩并加入 5 项冻结传感器身份检查后，CLI 全部 45 项复测通过。
学习模块单独完成 40 项全通过。原生替换脚本的失败与这些运行路径回归分别报告。

| 检查 | 结果 | 证据 |
|---|---|---|
| 未修改的几何与环境基线 | 56 项通过 | `tmp/sensor-rendering/baseline.log` |
| 学习模块完整回归 | 40 项通过 | `tmp/sensor-rendering/learning-final.log` |
| CLI 完整复测 | 45 项通过，含缺失/变化传感器身份拒绝和显式跨模型对照 | `tmp/sensor-rendering/cli-final.log` |
| 其余模块 | 194 项通过 | `tmp/sensor-rendering/regression.log` 中 CLI 之外的全部案例 |
| 快照、运动时钟、源表相位、梯度与官方 Renderer | 12 项通过 | `tests/test_sensor_snapshot.py`、`tests/test_render_reference.py`；`snapshot-render.log` |
| Depth/LiDAR × PPO/APG/SHAC 更新与完整恢复，加具名损失检查 | 8 项通过；CNN/PointNet 参数确实更新，恢复后的下一次更新一致 | `tests/test_learning.py`；`perception-updates.log` |
| Depth 新旧同输入对照 | S01、D01 各三个世界时刻，深度、点坐标及有效 mask 一致 | `tmp/sensor-rendering/depth-migration.json` |
| 依赖和分发包 | Pixi 锁文件检查、pip 依赖检查、sdist/wheel 构建、独立 wheel 导入及两种传感器环境步进通过 | `lock-check.log`、`build.log`、`wheel-smoke.log` |

Renderer 对照覆盖正对及倾斜墙面，测试容差为 `atol=rtol=2e-5`，相机内参与像素中心一致。
本机使用已有 X11 和 Mesa 软件 OpenGL 完成对照，没有安装驱动或改变系统渲染配置。
无 OpenGL 上下文的环境会明确跳过这两项测试；本次两项均实际执行。
LiDAR 学习集成测试改为完整 20,000 条采集，仅缩小策略输入；源表开头连续 32 条不能
替代完整扫描的空间覆盖，因而不再把减少采集条数当作这个测试的加速方式。

32 个世界、每帧 20,000 条、零送达延迟时，传感器状态的逻辑数组总量从
28,854,688 降至 21,122,464 字节，减少约 26.8%。这项计算只统计状态叶子数组，
不等于峰值进程内存、显存或训练吞吐收益。

### 原生求交门禁未通过

实测版本：MuJoCo 3.15.0、MuJoCo-LiDAR 0.3.5、JAX 0.11.2。
`tools/check_sensor_migration.py` 保留了可复现检查。返回码为 1，七个指定案例有六个
不符合现有 MuJoCo 交点约定；这是边界案例检查，不是随机精度统计。

| 案例 | MuJoCo 参考距离/m | 候选距离/m |
|---|---:|---:|
| Sphere 中心向外，半径 1 | 1 | 0 |
| Box 中心向外，半尺寸 1 | 1 | 0 |
| Cylinder 中心轴向，半高 2 | 2 | 0 |
| Capsule 中心轴向，半高 2、半径 1 | 3 | 0 |
| Plane 背面向上 | −1（未命中） | 1 |
| Box 表面平行射线，起点 `[-2,1,0]` | 1 | −1（漏检） |
| Box 外部正向射线，起点 `[-2,0,0]` | 1 | 1 |

运行时没有增加候选后端、回退分支或上游副本。扫描资源已复用；原有唯一的求交
实现保留，直到替换满足原先的“无退化后删除”条件。

### GPU 恢复与完整更新吞吐：2026-10-10

NVIDIA RTX 4090，驱动 595.84，JAX 0.11.2 已识别 `CudaDevice(id=0)`。
运行 `nvidia-smi` 时 GPU 处于空闲状态，完整更新采用独立进程、禁用 JAX 预分配，
单进程先编译预热，再测量连续三次更新的中位数。原版代码固定为
`v0.2@6ec50c7` 的独立源码快照；新版为本功能分支。两者各自从同样的算法、
环境、批量和参数种子启动。LiDAR 扫描方向与采集时间语义发生变化，所以这些值是
**完整模型迁移性能**，不是第三方求交内核加速倍数。

| APG 工作负载 | v0.2 耗时/更新 | 新版耗时/更新 | 交互吞吐：v0.2 → 新版 | 加速 |
|---|---:|---:|---:|---:|
| D01 LiDAR，32 环境 × 32 步，20,000 射线/帧 | 0.81135 s | 0.73041 s | 1,262 → 1,402 /s | 1.111× |
| S01 Depth，128 环境 × 32 步，64×48 @ 30 Hz | 0.83697 s | 0.77691 s | 4,894 → 5,272 /s | 1.077× |

采样时 `nvidia-smi` 的显存占用分别为 LiDAR **2,685 → 2,693 MiB**、
Depth **1,159 → 1,161 MiB**；这是同步后采样值，并非运行过程中的峰值。
完整更新吞吐改善约 11.1% 和 7.7%，不包括首次 XLA 编译时间。
此处采用**几何局部高精度修复后的最终结果**。
此前的 13.5%/9.9% 是修复前的中间值，不代表最终代码的吞吐。
原版和新版的每个计时样本、梯度范数、设备显存、环境规模和标签分别保存在
`tmp/sensor-rendering/full_{lidar_D01,depth_S01}_{baseline,candidate}.json`。
修复后的新版记录为 `full_{lidar_D01,depth_S01}_candidate_precise.json`，
GPU 官方 Renderer 专项复测为
`tmp/sensor-rendering/gpu_explicit_precision_tests.log`（12 项通过）。

Depth/LiDAR × PPO/APG/SHAC 共六种组合另完成 GPU 实际参数更新：
每组 2 环境 × 2 步展开，LiDAR 保持完整 20,000 条采集；分别执行首次编译与
后续两次更新。六组 Actor 梯度有限且非零，网络参数发生变化，传感值有限，
正式结果保存在 `tmp/sensor-rendering/gpu_update_*.json`。这些功能更新不构成
从随机初始化训练至任务验收的证据。

同数量射线/像素的单帧 GPU 前向测试，使用动态 JIT 位姿输入并同步全输出；
四个场景组合差异较小、存在测量波动，因此不把单帧性能当成整条训练链的替代指标。
对应记录在 `tmp/sensor-rendering/gpu_forward_dynamic_input.log`。

### 冻结策略迁移评测：2026-10-10

使用原版 `results/scratch_v02/` 中真实 v2 冻结策略，不重新训练，完整运行
8 场景 × 25 回合。明确启用已有的 `benchmark.allow_environment_change=true`
并设置独立 `resume=null`，承认扫描模式变更，同时保存实际传感器身份与相同的
2,000,000 基准种子区间。未将迁移后的成功率冒充原模型的原始成绩。

| 场景 | Depth/APG Seed 0，Step 480 | LiDAR/APG Seed 0，Step 3680 |
|---|---:|---:|
| S01 / S02 / S03 | 25 / 25 / 25 | 25 / 25 / **0** |
| D01 / D02 / D03 | 25 / 25 / 25 | 25 / 25 / 25 |
| S06 / D06 | 25 / 21 | 25 / 25 |
| 合计 | **196/200** | **175/200** |

Depth 在主六场景全部 25/25，D06 为 21/25，与原版归档的冻结报告一致。
LiDAR S03 新扫描为 0/25；另用**相同权重、相同种子、原版传感器实现**
重新跑 S03，原版同样 0/25。这一场景的失败不能归因于本次扫描资源替换。
LiDAR 其余七场景均为 25/25。两份全场景报告位于
`tmp/sensor-rendering/frozen_{depth,lidar}_apg_s0_all8/eval/001/report.json`；
旧版 LiDAR S03 对照在 `frozen_lidar_apg_s0_S03_baseline/`。

### 仍未满足的独立门禁

重新启用 GPU 后，`tools/check_sensor_migration.py` 仍返回 1：
MuJoCo-LiDAR 0.3.5 原生 JAX 求交在 7 个边界案例中有 6 个与
`mujoco.mj_ray` 不一致。因此**没有替换和删除当前 `Scene.raycast()`**，
而正式训练只有这一条求交路径。此阻塞来自上游求交语义，与 GPU 可用性无关，
证据保存在 `tmp/sensor-rendering/native-gate-gpu.json`。
新 MID360 方案的从零训练收敛矩阵尚未重新完成；功能分支保持隔离，不合并到 `v0.2`。
主工作区现有训练、权重和报告未被修改。

### 复现入口

```bash
pixi run env JAX_PLATFORMS=cpu python -m pytest -q tests/test_sensor_snapshot.py tests/test_learning.py
pixi run env JAX_PLATFORMS=cpu MUJOCO_GL=glfw LIBGL_ALWAYS_SOFTWARE=1 \
  __GLX_VENDOR_LIBRARY_NAME=mesa python -m pytest -q tests/test_render_reference.py
pixi run env JAX_PLATFORMS=cpu python tools/check_sensor_migration.py \
  --output tmp/sensor-rendering/native-gate.json
```

第二条需要可用的 X11 display；第三条在当前候选版本应返回 1 并保留全部失败证据。
MID360 资源 SHA-256：`9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3`。

## v1 资产清理与 Wiki（2026-10-09）

当前代码只接受 v2 训练和冻结策略。已核对 `results/` 中的 75 个 v1
`latest.training.zip` 全部是旧训练状态，随后按明确清理要求移除，合计
**2,353,132,446 字节（约 2.19 GiB）**。**546 个 `policy.zip`** 文件的路径与大小
保持不变，`results/acceptance/current.json` 的 SHA-256 未变化；历史模型推理不在
当前支持范围内。逐文件 SHA-256、路径、大小及来源状态保存于忽略目录
`tmp/neat-freak/legacy-training-deleted-20261009.json`，清单不能恢复已删除的参数。

GitNexus 索引与 Wiki 经增量刷新，以 `deepseek/deepseek-v4.1-flash` 完成 24 页生成。
Wiki 的来源提交应与 `main` 的 HEAD、知识图谱索引一致；完成状态以
`.gitnexus/wiki/meta.json` 及命令退出码为准。生成式文档不作为未经测试的实现证据。
原 Brax 工作树仍有未提交代码，未清场。

## 主分支合并检查：2026-10-09

已将 PPO 独立实验与控制模型改动整合。合并后的三组针对性验证分别为
**10 项 PPO 测试、17 项控制/CLI/迁移测试和 23 项验收收集器测试，共 50 项通过**。
Ruff 检查通过，56 个 Python 文件符合格式要求。该组检查验证合并点的关键调用，
没有重新运行完整 CPU、GPU 收敛和原生飞行验收。
合并时的测试和提交来源保存在 Git 历史，本节给出已验证的结果摘要。

## 控制模型工作树：2026-10-09

独立工作树完成 21 个不重复的针对性测试：19 项模型/控制/延迟/恢复/CLI 检查，
以及 4 项迁移检查（其中 2 项与前者重叠）。Ruff 和 56 个 Python 文件格式检查通过。
sdist 和 wheel 构建成功；安装后检查通过 9 个新增模块、66 种配置及来源许可文件。
实际 acados 完成一次 0.1 s 悬停闭环，回合为 SUCCESS；样本数未达到正式验收门槛。

完整日志保存在当时的 `tmp/implementation/`；本次按用户要求
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

首发历史闭环使用零观测和零动作送达延迟；当前代码已实现可配置的送达缓冲区。
传感器按自身时钟在物理步间插值，包含 500 Hz 物理与 30 Hz 相机的非整数周期。
当前方法执行频率须整除物理频率。Mid-360 的非重复扫描为已声明的合成近似，
不声称复制厂商专有扫描序列。完整设备采样、策略降采样和显示抽样分别记录。

早期诊断曾使用只读 Python 3.13/JAX 0.9 依赖，项目实现始终来自本仓库。
它们未计作正式种子；Racing 共同初始化来源仍保留该环境身份。历史正式三种子训练
使用锁定的 Python 3.12/JAX 0.11。历史参数用于初始化，不能据此声称从零训练。
