# 控制器、反向模型与运行记录

当前 `v0.2` 使用独立的动作、控制器和反向动力学模型。只支持 v2 训练和冻结策略归档；
新训练应从新配置开始，不能将历史 v1 Checkpoint 当作恢复入口。以下命令在仓库根目录执行。

## 控制接口与控制器

`method` 选择 Policy 或 Planner/Controller 组合，`controller` 选择轨迹跟踪实现。
Hydra 在构造时完成组合，Runner 不再按 EGO、SUPER 或具体控制器名字分派。

**接口核对：2026-10-10T10:46:22+08:00，集成提交 `6ec50c7`。**
`PlannerController` 可以注入 Mellinger、SO3、两种 MPC 和 IdealTracking。
`PolicyMethod` 使用 `method.action` 选择控制输入；当动作是 `acceleration` 时，
`Action` 使用 Crazyflow 的 Mellinger 位置控制转换将加速度映射为姿态和推力；
直接输出姿态或角速度时则进入 Crazyflow 对应的低层控制链。
`controller` 配置用于 Planner/Controller 跟踪组合，不用于更换 Policy 的动作接口转换。
**2026-10-10 已确认该行为符合设计，不作为待解耦缺口。**

| 配置 | 实现 | 执行方式 |
|---|---|---|
| `controller=mellinger` | 官方 Crazyflow 位置控制转换 | JAX |
| `controller=so3` | EGO SO3Control 外环的 JAX 移植 | JAX |
| `controller=sampling_mpc` | Crazyflow 采样预测与精英均值更新 | 宿主调度，JAX 候选预测 |
| `controller=attitude_mpc` | LSY 非线性 OCP，acados 求解 | 宿主 |
| `controller=ideal` | SUPER 理想跟踪响应 | 宿主调度，逐物理步更新状态 |

```bash
pixi install --locked
JAX_PLATFORMS=cpu pixi run sim controller=so3 simulation.device=cpu \
  task.reference=hover task.duration=0.2 output=results/so3_smoke
JAX_PLATFORMS=cpu pixi run sim controller=sampling_mpc simulation.device=cpu \
  controller.samples=32 controller.horizon=5 controller.prediction_seconds=0.2 \
  task.reference=hover task.duration=0.2 output=results/sampling_smoke
JAX_PLATFORMS=cpu pixi run sim controller=ideal simulation.device=cpu \
  task.duration=0.2 output=results/ideal_smoke
```

两种 MPC 保留原预测模型、代价和控制范围。LSY 保留旧实现中的推力上限系数，
不是新的整机调参结果。SO3 保留原外环方程；增益按标称质量归一化，输出力在当前
机体推力方向投影后交给 Crazyflow 姿态内环。相应来源许可随安装包保存。

LSY 需要可选的 acados v0.5.1。它不进入默认训练依赖：

```bash
pixi run setup-acados
export ACADOS_SOURCE_DIR="$PWD/.cache/acados"
export LD_LIBRARY_PATH="$ACADOS_SOURCE_DIR/lib:${LD_LIBRARY_PATH:-}"
JAX_PLATFORMS=cpu pixi run sim controller=attitude_mpc simulation.device=cpu \
  task.reference=hover task.duration=0.2 output=results/attitude_mpc_smoke
```

已有匹配版本时，`setup-acados --source /path/to/acados` 可复用原生库；生成的求解器
写入本次 `output/acados`，不覆盖其它运行。该安装步骤需要编译工具和依赖下载。

SUPER worker 已启动时，理想跟踪使用相同的 Planner 入口：

```bash
pixi run benchmark task=navigation scene=S01 sensor=lidar method=super \
  controller=ideal benchmark.protocol=diagnostic 'benchmark.scenes=[S01]' \
  benchmark.episodes=1 output=results/super_ideal
```

理想跟踪按每个物理时刻采样参考并检查任务和碰撞，报告写入
`execution: ideal_tracking`。它直接更新运动状态，不运行真实电机动力学。
`Trajectory` 继续使用宿主 NumPy 表示，MPC 仍能读取完整未来时域。

## 训练与反向模型

动作来自 `method.action`，网络来自 `method.actor`，训练损失来自 `learning.loss`。
网络输出维数由动作接口确定，循环记忆由 Actor 初始化。Task 不决定动作维数。
归一化动作仅由 `Action.decode()` 定义物理尺度，控制和损失使用相同解码函数。

```bash
pixi run train experiment=tracking_point_mass learning=apg seed=0 \
  output=results/tracking_point_mass_s0
pixi run train experiment=tracking_lotf learning=shac seed=0 \
  output=results/tracking_lotf_s0
```

`learning.backward_model=null` 使用实际前向的原生导数。
`point_mass_lag` 要求净加速度输入；`lotf` 要求总推力和机体系角速度输入。
前向仍由 `simulation.dynamics` 选择官方 Crazyflow 模型。

PointMass 替换位置、速度和加速度的输出导数；LOTF 替换位置、姿态、速度及其导出的
加速度。其余输出保留原生导数。例如 PointMass 没有电机状态方程，电机状态的导数
继续来自 Crazyflow，而不是设为零。每个物理步使用实际已送达指令，反向不会提前
使用尚未送达的动作。这一混合求导保留部分原生计算，不承诺省去全部物理反向成本。

可选模型参数在 `learning.backward_options` 中声明：PointMass 使用 `time_constant`，
LOTF 使用 `mass`。不指定 LOTF 质量时使用标称机体质量。PPO 不运行反向物理模型。
完整 Checkpoint 记录网络、动作、损失和反向模型规格；冻结推理不加载训练反向模型。

## 延迟和随机化

`simulation.action_delay_s`、`sensor.config.latency` 接受固定秒数，或每回合均匀采样
一次的 `[low, high]`。命令按物理时钟送达；测量保留采集时间和采集位姿，送达前不
暴露给方法。缓冲区属于各世界的状态，随局部 reset 清理、随完整 Checkpoint 恢复。

```bash
pixi run sim simulation.action_delay_s=0.04 output=results/delayed_control
pixi run train experiment=tracking_point_mass learning=apg \
  'simulation.action_delay_s=[0.025,0.05]' \
  '+simulation.randomization.mass=[0.9,1.1]' \
  '+simulation.randomization.motor_strength=[0.95,1.05]' \
  output=results/randomized_tracking
```

物理参数通过 Crazyflow `reset_pipeline` 采样，外力和阵风通过 `step_pipeline`
施加。支持质量、惯量、电机强度，以及模型实际使用的阻力系数。控制器和动作缩放
使用标称参数，实际物理使用随机参数。`simulation.disturbance` 可声明
`force_world_n`、`torque_body_nm`、`gust_std_n` 和 `gust_period_s`。

## 结果目录与旧记录

```text
results/<run_id>/
  config.yaml
  run.json
  metrics.jsonl
  checkpoints/
    latest.training.zip
    step-000025/
      policy.zip
      report.json
      episodes.csv
  eval/001/
    report.json
    episodes.csv
    trajectories.npz    # 按需保存
    replays/             # 实际生成回放时才创建
```

`run.json` 的 `selection` 引用选中权重。选模和最终评测仍用不同种子；场景在同一张
回合表中由 `scene` 区分，失败回合不丢弃。再次评测分配新编号，不覆盖已有结果。

当前仅接受 `format_version=2`，不提供旧训练状态、旧冻结权重或旧结果目录的运行时
兼容／迁移工具。v1 产物作为只读历史证据保留在 `results/`，不能混入新实验；
独立验收收集器也只读取 v2 的 `checkpoints/step-*/`、`eval/` 和 `metrics.jsonl`。
冻结历史验收结论不等于新代码的实测收敛结论，验证范围见[验证记录](validation.md)。
