# 方法配方、环境预设与统一运行入口

日期：2026-09-28。状态：设计草案，等待用户确认目录与运行语义；可执行配置迁移另行开展。
当前事实基线：主工作树 `813a7c5`；点云论文在独立工作树继续维护。

## 已确认目标

用户通过一个方法配方和一个环境预设启动实验。方法配方选择决策实现、执行控制器、动力学、
网络、训练算法、训练目标与默认运行设置；环境预设组合场景、任务、传感测量与观测表示。
训练目标保留统一选择入口，内部区分任务奖励与算法损失；导数选择归 `algorithm.gradient`。
学习、优化和混合决策方法共享兼容的执行状态转移及评测协议，允许显式保留原生执行组合。

这份草案描述目标结构。下列新命令和配置文件名属于拟议接口；现有十组配置继续按现有代码执行。

## 配方与实现各自组织

`configs/method/` 是用户浏览的方法配方目录，保留 `project`、`custom`、`rl`、`diffrl` 四个入口。
`project/custom` 描述来源，`rl/diffrl` 提供通用训练基线。运行时通过配方解析出的能力和模块实现
决定如何训练与执行；目录分类只承担浏览与来源管理。

同一论文配方可以调用通用 PPO 或 BPTT 更新器。自定义 l2osando 可以组合 SANDO 决策实现、
可训练网络与导数规则。共享更新器、控制器、传感器和动力学各维护一个实现来源。
原始论文代码通过固定版本依赖、子模块或项目管理的原生构建接入；适配代码进入本项目的职责模块。

每个方法配方记录来源、实际实现身份、可训练能力、输入/输出契约和支持的运行方式。
来源区分“原始代码接入”“对照源码迁移”“公开论文信息重建”。
点云工作当前来源依据为独立工作树 `docs/research/pointcloud-paper-source-audit-20260928.md`，
其身份为公开信息重建；LOTF 与该工作分别归档。论文标题与配置短名通过来源表关联。

## 使用入口

配置系统沿用 Hydra，文件使用 `.yaml`。统一选取语法为 `method=... env=...`。
`scripts/` 内保留训练、评测、展示三个轻量入口，统一调用包内装配与运行模块。
整套环境通过 `env` 选择，场景和任务仍可以在高级配置中单独替换。

```bash
# 在竞速环境中重新训练 PPO。
python scripts/train.py method=rl/ppo env=racing

# 加载 DiffPhysDrone 的具名方法配方及其已核实的默认环境。
python scripts/train.py method=project/diffphysdrone

# 在平台静态避障环境中训练该方法，记录为环境迁移实验。
python scripts/train.py method=project/diffphysdrone env=static_oa

# 用同一 SUPER 配方分别评测静态与动态避障。
python scripts/eval.py method=project/super env=static_oa
python scripts/eval.py method=project/super env=dynamic_oa

# 同一个冻结策略迁移到新环境；入口校验观测与命令语义兼容。
python scripts/eval.py checkpoint=experiments/ppo-static/checkpoints/best.pkl env=dynamic_oa

# 加载检查点，默认沿用保存的环境和输入输出契约。
python scripts/play.py checkpoint=experiments/ppo-static/checkpoints/best.pkl

# 直接运行在线规划器用于观察行为。
python scripts/play.py method=project/super env=static_oa

# 显式选择轨迹导出，适用于无显示器服务器。
python scripts/play.py method=project/super env=static_oa visualization.mode=rscope

# 查看已记录轨迹；只读已有运行。
python scripts/play.py replay=experiments/super-static/rollouts
```

示例检查点与运行名表示未来实际运行的产物位置。现有检查点实际文件名与算法格式继续依据元数据读取。
`train` 调用需要更新参数的完整训练配方；原生 SANDO、EGO、SUPER 等纯优化配方声明评测与展示能力。
BPTT 配方还需指定时间展开、参数优化器和训练目标，才能构成完整训练方法。

### 配置示意

以下展示方法配方如何组合现有类型的槽位；它是目标设计示意。

```yaml
# configs/method/rl/ppo.yaml
# @package _global_
defaults:
  - /env: racing
  - /controller: crazyflow_attitude
  - /dynamics: crazyflow
  - /network: state_mlp
  - /algorithm: ppo
  - /objective: racing
  - /training: default
  - _self_

method:
  name: ppo
  implementation: neural
  kind: learned
```

环境预设的组合结果包含 `env.scene`、`env.task`、`env.sensor`、`env.observation`。
传感器拥有独立配置文件和实现；其配置归入环境内部，满足单独替换的需求。
训练相关选择继续公开 `algorithm`、`network`、`objective`、`training` 四组。
算法选择直接或代理导数，导数能力由实际控制、动力学或可微求解实现提供。

### 覆盖和兼容规则

Hydra 默认列表先组合基础槽位，再加载环境默认项和方法声明的明确覆盖，命令行显式覆盖最后生效。
原始配置、覆盖列表和最终解析配置均保存。代码只消费装配后的一份解析结果。
环境预设拥有场景和任务规则，并提供传感与观测默认选项；方法对传感和观测的要求以显式配置覆盖及
能力约束表达。SUPER 配方选定 MID-360 与原始点云输入，EGO 配方选定深度输入；静态/动态环境切换
保留各自方法要求的输入语义。配置要求无法满足时，启动阶段报告具体不兼容项。

奖励配置也接受任务兼容校验。PPO 竞速奖励依赖过门事件，跨到点到点任务时应选择对应导航目标配方；
已经有明确任务对应规则的基线可由已登记组合预设给出选择。目标函数的任务对应关系在配置中可查。
通用方法配方提供默认可执行组合；新增模态、动作或任务通过显式配方及验证扩展。

加载检查点评测时，保存的最终配置是基线。网络、归一化、动作解释和方法参数按检查点身份恢复；
显式 `env`、控制器或动力学覆盖生成新的执行配置和差异记录，并校验维度、字段意义、坐标、时间、
单位、模型状态投影及已声明导数能力。方法名称与检查点身份冲突时报告错误。
学习运行的续训与重新训练分别记录，后者创建独立运行目录；正式评测固定策略参数与归一化统计。

## 环境目录的含义

`racing`、`static_oa`、`dynamic_oa` 作为环境预设名称。
竞速预设组合门洞几何、按序穿门任务和对应观测；两类避障预设组合 Navigation8 的静态/动态视图、
点到点导航任务以及配套传感/观测默认项。几何与运动继续引用同一版本化 Navigation8 目录。
论文默认环境采用单独命名的预设，来源和已核实范围明确记录。

一个场景可以组合悬停、参考跟踪、点到点或航点访问等任务；每种组合逐项校验可行性与任务语义。
同一个任务可以使用状态、深度或点云观测，也可以为策略和价值网络配置不同的可见信息组。
学习编码器仍由网络配置拥有，环境观测负责测量选取、坐标、历史、噪声及归一化约定。

## 评测与展示共用闭环

三个入口的默认职责如下：训练负责参数更新和开发集选模；评测负责冻结方法下的确定试次及全分母统计；
展示负责少量回合交互查看，或者读取已经生成的回放。
评测和在线展示调用同一个闭环执行模块，以调用参数和输出接收方式区分行为。
展示默认输出完整回放包，有可用桌面与图形上下文时打开现有 RScope/MuJoCo 查看能力。
`visualization.mode=auto` 自动选择，`mujoco` 显式要求窗口，`rscope` 显式导出包。
自动模式中窗口创建失败时保留导出并记录具体原因；显式窗口模式给出清晰错误。

回放包包含 `.mj_unroll`、元数据、场景模型与必要资源，以及对应的事件、时间与场景身份。
图像渲染与轨迹导出分别处理；纯状态导出使用数组与文件操作。
物理状态由配置选择的前向动力学推进，显示端读取同一份状态。MuJoCo 被动查看或 RScope 显示不会
隐式增加一套用于产生轨迹的动力学。动态障碍、参考和传感器采样时间随实际记录展示。
正式计时评测默认先记录、后查看，以隔离可视化开销与规划器计算时序。

## 来源条件与跨环境实验

论文条件实验固定作者已公开且已核实的任务、模型、网络、训练目标、更新规则和预算。
平台迁移实验保留方法身份，显式替换环境或执行组合，并报告变化。
训练脚本完成预算只证明本次配方执行完成；论文结果是否复现由独立评测与来源对照判断。
原始源码可用性、算法适配完成、预算完成和策略质量分别保存。

本草案沿用当前 JAX/Brax 训练约定。共享平台中的论文重建或对照迁移可复用这些训练能力。
直接运行作者独立训练框架属于单独的接入决策，需要说明依赖隔离、状态/检查点交接和比较语义。

## 目标文件目录

下列列出完整的职责结构与主要配方；实现文件按实际接入落地，研究中的方法不通过空目录宣告完成。

```text
drone_playground/
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── AGENTS.md
├── CONTEXT.md
├── pyproject.toml
├── pixi.lock
│
├── scripts/
│   ├── train.py                      # 训练入口
│   ├── eval.py                       # 正式评测入口
│   ├── play.py                       # 在线展示或既有轨迹回放
│   └── tools/                        # 场景构建、导出和历史数据整理工具
│
├── configs/
│   ├── config.yaml                   # 全局装配默认项
│   ├── method/                       # 用户按方法浏览的配方
│   │   ├── project/
│   │   │   ├── sando.yaml
│   │   │   ├── ego_planner.yaml
│   │   │   ├── super.yaml
│   │   │   ├── pointcloud_flight.yaml # Learning to Fly from Point Clouds…
│   │   │   ├── diffphysdrone.yaml
│   │   │   └── lotf.yaml              # Learning on the Fly，独立来源
│   │   ├── custom/
│   │   │   └── l2osando.yaml
│   │   ├── rl/
│   │   │   └── ppo.yaml
│   │   └── diffrl/
│   │       ├── apg.yaml
│   │       ├── bptt.yaml
│   │       ├── shac.yaml
│   │       └── dva.yaml
│   ├── env/
│   │   ├── hovering.yaml
│   │   ├── tracking.yaml
│   │   ├── racing.yaml
│   │   ├── static_oa.yaml
│   │   ├── dynamic_oa.yaml
│   │   └── project/                  # 各论文的具名环境及核对条件
│   ├── scene/
│   │   ├── empty.yaml
│   │   ├── lsy_racing.yaml
│   │   ├── navigation8_static.yaml
│   │   ├── navigation8_dynamic.yaml
│   │   └── navigation8.json
│   ├── task/                         # 悬停、跟踪、过门、点到点规则
│   ├── sensor/                       # none、D435、MID-360、均匀扫描
│   ├── observation/                  # 状态、深度历史、点云历史、论文输入
│   ├── controller/                   # 命令执行、跟踪、飞控、加速度直通
│   ├── dynamics/                     # Crazyflow、LOTF、质点滞后
│   ├── network/                      # 状态网络、深度编码、PointNet/GRU
│   ├── algorithm/                    # 更新算法配置
│   │   ├── ppo.yaml
│   │   ├── apg.yaml
│   │   ├── bptt.yaml
│   │   ├── shac.yaml
│   │   ├── dva.yaml
│   │   └── gradient/                 # 挂载到 algorithm.gradient
│   ├── objective/                    # 训练目标：奖励项与具名损失设置
│   ├── training/                     # 种子、预算、设备、保存、恢复
│   ├── evaluation/                   # 固定试次、评测划分、统计规则
│   └── visualization/                # 自动窗口/导出和显示设置
│
├── src/drone_playground/
│   ├── app.py                        # 共用命令入口和运行路由
│   ├── composition.py                # 构造、配置解析和兼容检查
│   ├── contracts.py                  # 输入、命令、时钟、能力声明
│   ├── methods/                      # 执行时的决策实现
│   │   ├── neural.py                 # 网络推理与方法记忆
│   │   ├── optimization/
│   │   │   ├── sando.py
│   │   │   ├── ego_planner.py
│   │   │   ├── super.py
│   │   │   ├── attitude_mpc.py
│   │   │   └── sampling_mpc.py
│   │   └── hybrid/
│   │       └── l2osando.py            # 网络与求解器的具名组合
│   ├── learning/
│   │   ├── train.py                  # 训练编排、采样、选模和保存
│   │   ├── algorithms/               # PPO/APG/BPTT/SHAC/D.VA/LOTF更新
│   │   ├── networks/                 # 网络和编码器实现
│   │   └── objectives/               # 训练目标实现
│   ├── environments/
│   │   ├── environment.py            # 环境装配与任务交互
│   │   ├── scenes/                   # 几何、动态障碍、场景目录
│   │   ├── tasks/                    # 目标、参考、事件、重置
│   │   ├── sensors/                  # 测量几何、有效性和采样时刻
│   │   └── observations/             # 各接收者的输入表示
│   ├── execution/
│   │   ├── transition.py             # 完整命令执行和多速率状态转移
│   │   ├── controllers/              # 跟踪、姿态内环、混控等
│   │   └── dynamics/                 # 前向模型及其具名导数实现
│   ├── evaluation/
│   │   ├── evaluator.py              # 同一闭环，批量或逐回合执行
│   │   └── metrics.py                # 事件、成功率和耗时等统计
│   ├── visualization/
│   │   ├── viewer.py                 # 复用现有 MuJoCo/RScope 查看能力
│   │   └── rscope_io.py               # 实际状态与模型包读写
│   └── runs/
│       ├── record.py                 # 清单、来源和完整产物
│       └── checkpoints.py            # 参数、完整训练状态及恢复身份
│
├── third_party/
│   ├── sources.yaml                  # 原始来源、固定版本和获得方式
│   └── learning_on_the_fly/           # 现有原始子模块；其余按接入方式取得
├── native_planners/                  # 项目自管的 Docker/ROS 构建与启动
├── assets/                           # 公用机体、场景和显示资源
├── tests/                            # 从真实装配接口验证替换与执行
├── experiments/                      # 独立运行：配置、检查点、结果、回放
├── docs/
│   ├── architecture.md
│   ├── design/
│   ├── adr/
│   ├── research/
│   ├── verification/
│   ├── diagrams/
│   ├── status.md
│   └── backlog.md
├── .scratch/                         # 现有阶段规格和工作单
└── tmp/                              # 临时验证与构建产物
```

`configs/method/rl/ppo.yaml` 表示完整 PPO 方法配方；`configs/algorithm/ppo.yaml` 表示其中可复用的
参数更新规则。前者组合后者及网络、环境、控制与目标。其他论文配方可选同一个更新规则。
算法的数学损失与更新过程保留在算法实现中；训练目标文件声明可复用任务信号和所需具名设置。
一个来源项目能贡献多种模块，例如 LOTF 的模型与控制器进入执行模块，网络与更新进入学习模块，
原始代码保持固定依赖身份。

## 参考项目提供的具体依据

Isaac Lab 的管理器式环境将场景、观测、动作、奖励、终止等配置组合为具名环境；脚本通过任务名
选择该环境。Hydra 用环境和学习参数的独立路径提供命令行覆盖。

- [管理器式环境组合](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html)
- [Hydra 配置](https://isaac-sim.github.io/IsaacLab/main/source/features/hydra.html)
- [训练与展示脚本，固定 v2.2.0 文档](https://isaac-sim.github.io/IsaacLab/v2.2.0/source/overview/reinforcement-learning/rl_existing_scripts.html)

MuJoCo Playground 通过环境注册表读取环境默认配置，训练脚本另取对应学习参数；JAX PPO 入口中
还提供独立评测环境、仅展示模式和 RScope 输出。本项目采用环境/学习职责分离与执行复用的思想。

- [环境注册](https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/registry.py)
- [JAX PPO 训练、仅展示与 RScope](https://github.com/google-deepmind/mujoco_playground/blob/main/learning/train_jax_ppo.py)
- [MuJoCo 被动查看器](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)
- [Hydra YAML 配置](https://hydra.cc/docs/tutorials/basic/your_first_app/config_file/)
- [Hydra 组合顺序](https://hydra.cc/docs/advanced/defaults_list/)

深模块原则沿用此前已经核对的 Matt Pocock codebase-design：实验调用者只了解方法、环境与运行
方式；命令转换、状态记忆、多速率、求解失败等复杂性集中到相应模块。JAX 批量展开和外部 ROS
执行保留各自适配方式；二者从同一领域接口产生可核对的轨迹和事件。

## 后续迁移的验收切片

建议先用 PPO 竞速与 SUPER 静态导航验证完整目录设计。验收覆盖方法/环境选择、正式配置落盘、
命令单位与维度、单次控制执行、终止和超时、检查点冻结、评测/展示轨迹一致及无显示器回放读回。
随后加入 SUPER 动态场景与一个可微训练配方，验证场景切换和 `algorithm.gradient`。
文档目录与命令获批后，再制定代码迁移步骤和历史产物的一次性解释策略。
