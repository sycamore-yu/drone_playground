# Robot-learning 语义收尾审计 — 2026-10-01

审查对象：当前 `drone_playground` 未提交工作树。工作树同时包含其他会话已经完成的整改；本记录只把下面的观测/目标边界、任务/基准边界及文档修正计为本轮新增修改。没有重训、删除正式权重、改写历史成绩或提交/推送整个工作树。

## 1. 观测噪声改变了真实训练目标

**证据。** `environments/tasks/pointcloud.py::PointCloudTask.observation` 与 `environments/tasks/pointcloud_navigation.py::RecurrentNavigationEnv.observation` 原先都将 `measured` 传给 `observer.proprioception`，同时取得策略输入与 `target`。`learning/algorithms/pointcloud_bptt.py::rollout_objective` 将这个 target 写进损失的 `target_velocity`。因此，固定真实状态，仅改变测量噪声，就会改变训练目标。新增四个测试全部复现：同一状态下目标速度最大分量偏移约 0.42005 m/s。

**标准概念与依据。** Observation corruption 与 ground-truth reward/objective 是不同计算边界。[MuJoCo Playground Go1](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/locomotion/go1/joystick.py) 的 `_get_obs` 创建含噪观测，`_get_reward` 则从真实 `data` 与 command 计算奖励。这不排除其他明确研究带噪奖励的方法；本项目声明的是观测噪声，目标应保持原定义。

**分类与必要性。** 独立概念混用，必须修复；否则启用 observation noise 同时修改了学习问题，无法解释噪声实验。

**最小修改。** 保留含噪状态生成 actor 的 proprioception；另用真实状态生成 loss target。复用原有 observer，没有新接口层。点云与深度的两条观测路径均修正。

**验证。** `tests/test_measurement_target_boundary.py` 的四项用例先失败后通过；证明 actor 输入改变、真实状态不变、target 与固定状态下的速度误差损失不变。

## 2. 固定 benchmark 参数被写成通用任务限制

**证据。** `environments/tasks/pointcloud_navigation.py::validate_navigation_adaptation` 原先强制参数元组等于 `(10 Hz, 500 Hz, 300 s, 0.5 m, 0.07 m)`，并要求 `max_speed == 20.0`。即便不选择 benchmark，自定义但物理上合法的任务也被拒绝。环境说明另外写死了 `500Hz`；warm-start 来源记录还写死了 `25–50ms, 500Hz`。

**标准概念与依据。** Task specification 决定任务参数，named benchmark 固定特定比较条件。[Isaac Lab 的任务环境教程](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html) 将任务规格与通用环境实现分离；[FlightBench](https://thu-uav.github.io/FlightBench/) 单独定义 test cases，并以共同测试条件运行 baseline。这支持职责分离，不要求复制其目录或建立新的 Protocol/Manager 类型系统。

**分类与必要性。** 历史实验配方扩散成通用合法性规则，必须修复。公开元数据也必须描述实际执行。

**最小修改。** 通用递归导航校验正的时长和半径、可整除的物理子步、兼容的传感/动作时钟和命令速度范围。具体 300 秒等限制仍由现有 benchmark 校验，默认配置未修改。`physics_engine` 使用实例的实际频率；删除重复且可能错误的 warm-start 时钟描述，实际条件继续保存在现有解析配置中。

**验证。** `tests/test_navigation_task_specification_boundary.py` 六项用例覆盖：20 Hz 决策配合 500/1000 Hz 物理步，分别执行 25/50 个子步，仿真时间均推进 0.05 秒；任务时长、到达半径与碰撞半径实际来自配置；正式 benchmark 继续拒绝被修改的条件；无效半径和超出速度上限的命令继续被拒绝。测试另复现并修复了真实 1000 Hz、文字却报告 500 Hz 的情况。

## 3. Environment 运行角色的文档与代码不一致

**证据。** `composition.py::ROLE_SEEDS` 和环境工厂只接受 `train`、`eval`。`docs/status.md` 与 `notes/archive/engineering-review.md` 仍把 `train/checkpoint_eval/benchmark` 描述为三种现役运行角色。其他会话已经更新 evaluation README，本轮未覆盖该改动。

**标准概念与依据。** 环境实例的用途、训练内评测活动与正式基准规格属于不同层次。[MuJoCo Playground PPO 入口](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/learning/train_jax_ppo.py) 从同一环境定义构造 environment 与 eval_env。两个 role 名称是本项目已确认的设计，不声称为全部上游统一 API。

**分类与必要性。** 术语及公开说明不一致，需要修改文档，不需要重新重构运行器。

**最小修改。** 文档统一为 Environment role `train/eval`；checkpoint evaluation 是训练内监控/选模活动，Benchmark 是正式评测规格，二者均使用 eval 语义。历史报告和旧运行事实保留原文。

## 当前已存在并保留的机制

- LOTF 配置只出现在 `configs/dynamics/lotf_high_fidelity.yaml` 与 `lotf_simplified.yaml`，不再另设 LOTF task/method/split。
- `training/default.yaml` 已声明 reset、observation/action noise、disturbance、command/scene distribution 和 dynamics DR；环境工厂与模型有对应消费者。本轮不重复添加或统一开启。
- 当前源码和配置检索未见 `free_course_v1`、`initial_condition_randomization`、伪 curriculum 或 `EvaluationCondition`。固定训练分布保留分布语义，不新建课程调度器。
- 训练内 eval_env 与 train_env 使用同一任务实现，名义 eval 不继承训练 DR、测量/动作噪声与外力。该默认选择已通过行为测试；它不是“域随机化只能用于训练”的通用定义。
- Command 与 scene 仍可由同一具体 case 一起提供；职责区分不要求强行拆散固定 benchmark 数据。

以上取舍与 [MuJoCo Playground Go1](https://raw.githubusercontent.com/google-deepmind/mujoco_playground/main/mujoco_playground/_src/locomotion/go1/joystick.py) 分别配置 command/noise/perturbation 的做法一致；[Isaac Lab managers 文档](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html) 也分别描述 observation corruption、reset/interval events 与 curriculum。这里借鉴的是作用位置和职责，未增加相同的 Manager 架构。

## 最终验证

所有命令使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu` 与 `.pixi/envs/default/bin/python`。

| 验证集合 | 完整运行结果 | 日志 |
|---|---|---|
| 新增 measurement target 与 task specification 两个测试文件 | 10 passed，9.03 s | `tmp/rl-domain-audit-20261001/boundaries.log` |
| 原有 pointcloud navigation training 与 navigation v2 测试 | 9 passed，55.16 s | `tmp/rl-domain-audit-20261001/navigation-regression.log` |
| 名义 eval 隔离训练 effects、训练 goal 独立、点云训练内 eval 名义条件 | 3 passed，25.19 s | `tmp/rl-domain-audit-20261001/train-eval.log` |
| 三个修改的源码文件和两个新增测试文件的 Ruff | All checks passed | `tmp/rl-domain-audit-20261001/lint.log` |
| 本轮相关文件的 `git diff --check` | exit code 0 | 与 boundary/lint 同一完成回执 |

合计 22 个不同用例完成并通过。保留的 JAXopt 弃用警告没有作为测试失败处理，未为此升级依赖。

较早的扩展回归收到 SIGTERM，回执 `command-9a8c3ee7834f25ff1fbbf1c32751fbfb817f52e07a83ec5c1ffc99d8c5b7ba0d` 不计为通过；之后分组完成了上表验证。没有运行完整仓库回归、长期训练、整套正式 benchmark 或实机测试，工程回归结果不改写既有成功率。

## 本轮修改范围

源码：`environments/tasks/pointcloud.py`、`environments/tasks/pointcloud_navigation.py`、`learning/algorithms/pointcloud_navigation_bptt.py`（均在 `src/drone_playground/` 下）。

新增正式测试：`tests/test_measurement_target_boundary.py`、`tests/test_navigation_task_specification_boundary.py`。

文档：`docs/status.md`、`notes/archive/engineering-review.md` 及本记录。
