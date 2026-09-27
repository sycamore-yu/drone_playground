# 运行与远程查看

项目：`/home/tong/tongworkspace/simulation_dev/mujoco/drone_playground`。
使用本项目Pixi；shell找不到命令时改用`/home/tong/.pixi/bin/pixi`。先进入项目目录。

## 当前LOTF结果

```text
experiments/lotf-hybrid-hover-seed0-v1/independent-heldout/rollouts/
experiments/lotf-hybrid-tracking-seed0-v1/independent-heldout/rollouts/
```

在已安装RScope Viewer的VS Code/Cursor远程工作区，点击其中`.mj_unroll`即可。每文件实际保存4–5条固定/最差试次；全部128个回合在相邻`report.json`。训练过程的9个时间点在各运行的`rollouts/step-*`。

悬停执行3秒，初态随机；报告同时给出全程和最后一秒误差。八字执行5秒，使用作者CSV和随机参考起点。回放模型外形是示意几何，质量/惯量/电机坐标来自原机型；实际状态由LOTF JAX物理生成。

## 配置与训练

查看完整解析配置不会启动训练：

```bash
pixi run experiment --cfg job experiment=lotf_hybrid_hover
```

新训练必须使用新运行标识：

```bash
env -u PYTHONPATH pixi run train experiment=lotf_hybrid_hover run_id=my-lotf-hover
env -u PYTHONPATH pixi run train experiment=lotf_hybrid_tracking run_id=my-lotf-tracking
env -u PYTHONPATH pixi run train experiment=figure8_ppo run_id=my-figure8-ppo
env -u PYTHONPATH pixi run train experiment=random_apg dynamics.forward=so_rpy_rotor_drag run_id=my-random-apg
env -u PYTHONPATH pixi run train experiment=racing_shac run_id=my-racing-shac
```

LOTF默认悬停600万交互、八字2250万交互，分别使用作者200/300次更新。已有P1–P4配方名称保持清晰身份；原`--config *.json`入口已迁移到实验名和Hydra覆盖。

PPO使用`training.num_timesteps`。LOTF/APG/SHAC使用更新次数×并行环境×展开步数；
`training.num_timesteps`默认null，显式填写时必须与推导值相同。LOTF原生轨迹CSV按50Hz采样，
其频率更改需实现重采样适配，组合入口会拒绝静默改变轨迹时间。

批量组合可使用`pixi run experiment --multirun`。下面的缩小APG/SHAC组合已实际验证：

```bash
pixi run experiment --multirun experiment=figure8_apg,figure8_shac   training.num_envs=4 training.policy_updates=2 training.num_evals=3   ++algorithm.horizon_length=8 'network.hidden_sizes=[16,16]'   'run_id=probe-${algorithm.name}-new'
```

低预算只用于执行链检查，正式训练使用冻结配方。每个试验独立保存配置、曲线、参数和结果，避免共享活动显示目录干扰训练。

## P5 导航任务

P5-05—08 正式批次使用 v2。已有队列运行时不要再次启动。
主会话：学习队列 session 96099，原生规划器队列 session 27701；
会话失效时以运行目录心跳和 PID/start marker 为准，禁止只根据旧 `status=running` 推断仍在运行。
v1 已因地面碰撞遗漏中止，三份目录有 `interruption.json`，仅供诊断。

```bash
cd /home/tong/tongworkspace/simulation_dev/mujoco/drone_playground
# 查看命令，不运行矩阵
env -u PYTHONPATH /home/tong/.pixi/bin/pixi run python scripts/run_p5_learning_matrix.py --revision v2 --dry-run
env -u PYTHONPATH /home/tong/.pixi/bin/pixi run python scripts/run_p5_native_matrix.py --revision v2 --dry-run
# 运行（只在确认没有原队列时使用）
env -u PYTHONPATH /home/tong/.pixi/bin/pixi run python scripts/run_p5_learning_matrix.py --revision v2
env -u PYTHONPATH JAX_PLATFORMS=cpu /home/tong/.pixi/bin/pixi run python scripts/run_p5_native_matrix.py --revision v2
# 随时重建当前证据表；验收时增加 --require-complete
python3 scripts/summarize_p5.py --revision v2
```

8 个训练单元各 8388608 交互，单种子 0。训练期间只使用开发集选模；
每个最佳检查点单独启动最终 dev/heldout 评测，每档 32/128 回合。
四个原生规划器单元采用同样清单和预算；合计 dev 1152、heldout 4608 回合。
队列不会覆盖未完成目录或自动暖启动；成功完成的目录须通过预算/回合数核验才复用。
一个单元失败后其余独立单元继续，最终队列退出非零，保留全部错误。
全矩阵修复重跑必须使用新的 revision，不能把不同任务协议混在同一张表。

学习队列日志：`experiments/p5-matrix-v2/`；规划器队列：
`experiments/p5-native-matrix-v2/`。单元完成时写 `queue-state.json`，
过程查看各运行的 `state.json`、`metrics/metrics.jsonl` 和 `native-progress.json`。
结果表：`docs/verification/p5-results-v2/report.md` 及同目录 JSON/CSV。
统计工具核验开发选模、检查点 SHA、冻结参数、每格分母及场景身份；缺失格明确标记未完成。

原生依赖构建脚本为 `scripts/setup_p5_native.sh`。它使用已有 flightbench 容器，
仅构建外部目录中的锁定 EGO 和 SUPER 运行目标，不替换主机 ROS 或系统库。
它要求当前容器已有 ROS Noetic、catkin 与 FlightBench 的 C++/ROS 依赖；
不是从空容器安装全部依赖的脚本。实测构建是逐条执行等价命令，脚本另通过 bash 语法检查。
EGO 源码默认 `tmp/p5-refsrc/ego-planner`（`https://github.com/ZJU-FAST-Lab/ego-planner.git`），
SUPER 默认 `/home/tong/tongworkspace/reference_repos/SUPER`（`https://github.com/hku-mars/SUPER.git`）；
在新环境先准备这些 Git 仓库，或通过 `P5_EGO_SOURCE/P5_SUPER_SOURCE` 指定路径。
实际使用固定 commit 的 `git archive`，不会使用未提交修改或随分支最新版本变化。
不要在矩阵运行期间重建这些二进制。
SUPER 的离线 read_replan_log 工具不属于运行依赖；运行只需要已构建的 fsm_node。
桥工作进程默认 4 回合并行，每个独立 master；两组规划器并行时端口范围互不重叠。

独立评测回放在
`experiments/<run_id>-{dev,heldout}/rollouts/{easy,medium,hard}/case-*/`，
每档固定前四个案例，另外保留首个失败案例（若它不在前四个中）。
RScope 显示同一次运行的机体与障碍运动。学习回放含策略压缩观测；
原生回放含本体状态，完整传感器示例见 `native/<difficulty>/0/` 的第 0/150 步采样包。
这两类输入表示不能当作等像素、等点数的算法比较。

### 单个配方与工程检查

```bash
cd /home/tong/tongworkspace/simulation_dev/mujoco/drone_playground
# 只解析配置，不启动训练
/home/tong/.pixi/bin/pixi run experiment --cfg job experiment=p5_navigation_static
# 真实闭环训练（静态/动态各一个工程配方）
/home/tong/.pixi/bin/pixi run train experiment=p5_navigation_static  run_id=p5-nav-static
/home/tong/.pixi/bin/pixi run train experiment=p5_navigation_dynamic run_id=p5-nav-dynamic
```

导航协议固定：50 Hz 策略频率、40 秒上限、0.5 米到达半径、机体碰撞判失败，
碰撞与到达同一步时碰撞优先。组合入口会拒绝修改这些数值或让静态/动态任务与场景不一致。

回放按难度分档，每档若干实例，每个实例一份自带障碍动画的 `.mj_unroll`：

```text
experiments/<run_id>/rollouts/step-<step>/{easy,medium,hard}/case-000/
```

独立评测的全部回合保存在 `experiments/<run_id>/traces/{easy,medium,hard}.npz`，
按 offsets 分隔案例，包含终止帧，无批量填充；index.json 保存摘要、场景身份与步数。
记录本体、动作、事件和指标，理想测量由记录位姿、初始重置种子、场景和校准重建。
任意案例均可导出到原 RScope 格式，无需重新飞行：

```bash
env -u PYTHONPATH pixi run python scripts/export_p5_archived_case.py <run_id> hard 127
env -u PYTHONPATH pixi run python scripts/reconstruct_p5_sensor.py <run_id> easy 0 --tick 150
```

输出在该运行的 `rollouts-from-archive/`。代表回放继续位于 `rollouts/`。
传感器重建输出在 `sensor-reconstruction/`，沿用实际原生采样函数；tick 必须是终止前的发布时刻。
输出为完整理想深度图或 MID360 全射线窗口及有效掩码，不会重新飞行或更新策略。
`scripts/verify_p5_sensor_reconstruction.py --run-id <run_id> --output <json>` 可与原生输入抽样包核对。
已实际验证静态第 150 tick（3 秒）深度与有效点云最大误差均为 0，见
`docs/verification/p5-sensor-reconstruction.json`；任意案例回放验证见 `p5-archive-export.json`。
早期 v2 评测缺少全回合归档，补采使用 `scripts/run_p5_archive_repairs.py --kind native`
或 `--kind learning`，固定沿用原清单与检查点；学习补采等待原八单元训练及独立评测队列结束。
只按归档缺失决定补采，不按得分决定，原目录不改写。后缀 `-archive-v1` 的
`archive-repair.json` 记录前后结果和身份；汇总表使用补采结果，原始额外回合不重复加入分母。

开发集选模顺序为宏平均成功率 → 碰撞率 → 受限完成时间；评价阈值待协议冻结确认。

传感器预设：`observation=navigation_depth`（D435 理想深度，120×90、水平视场 85.2°、
量程 0.1–10 米、策略网格 20×15、四帧历史）与 `observation=navigation_lidar`
（MID360，复用 MuJoCo-LiDAR 图案，每次扫描取 120 点、四帧历史）。两者都给策略 2420 维输入。
环境会在构造时校验观测块与传感器标定一致，不一致直接拒绝。

吞吐实测（不要靠猜）：

```bash
pixi run python scripts/p5_throughput_probe.py --output docs/verification/p5-throughput.json
```

## 独立评测

默认从检查点恢复环境身份，网络参数及归一化冻结：

```bash
pixi run evaluate   checkpoint=experiments/lotf-hybrid-hover-seed0-v1/checkpoints/step-0006000000.pkl   training.device=cpu evaluation.split=heldout evaluation.episodes=128   run_id=my-hover-evaluation
```

显式跨模型评测选择本次实验环境：

```bash
pixi run evaluate experiment=figure8_ppo   checkpoint=experiments/p2-figure8-ppo-seed0-v2/checkpoints/step-0001310720.pkl   evaluation.environment=experiment dynamics.forward=first_principles   training.device=cpu evaluation.episodes=128 run_id=my-transfer-evaluation
```

此命令是可用能力，具体跨模型质量以新运行结果为准。输入/动作维度不兼容会在执行前拒绝。

需要结果直接放入指定独立目录时，使用同一模型加载与评测函数的CLI包装：

```bash
pixi run python -m drone_playground.cli evaluate   --checkpoint experiments/lotf-hybrid-tracking-seed0-v1/checkpoints/step-0022500000.pkl   --split heldout --episodes 128 --device cpu --output experiments/my-tracking-heldout
```

## 优化控制的同一组合入口

```bash
bash scripts/setup_acados.sh
pixi run simulate experiment=racing_attitude_mpc evaluation.episodes=32 evaluation.split=dev run_id=my-mpc
pixi run simulate experiment=racing_sampling_mpc evaluation.episodes=32 evaluation.split=dev run_id=my-sampling
```

AttitudeMPC调用真实acados，采样MPC调用真实2000候选预测；配置保留作者预测模型和控制语义。控制器耗时记录与仿真时间分别报告。P4原正式128试次仍保存在`heldout-v2`运行中。

## 恢复训练

LOTF/SHAC使用`training.resume=<完整状态文件>`，同时保持模型、任务、网络、算法、环境数和总更新数一致，并使用新的运行目录。PPO参数暖启动使用`training.warm_start=<策略文件>`；它沿用原Brax方式重新初始化优化器和随机数。

完整状态文件在`training-state/update-*.pkl`，包含参数、优化器、随机数、环境状态和已完成更新数。原始记录保持只读；重复运行和恢复结果各有独立标识。

LOTF完整实验续训还需要该源运行的`eval/step-*.json`及其`checkpoints/`，用于继承恢复时刻以前的
开发集最佳策略；更晚时间点的评估不会参与。独立的`resume-selection.json`保存筛选来源和校验值。
仅复制数值状态可用于低层恢复测试；完整实验命令需要保留原运行目录关系。

```bash
pixi run train experiment=lotf_hybrid_hover \
  training.resume=experiments/lotf-hybrid-hover-seed0-v1/training-state/update-000175.pkl \
  run_id=my-new-resume
```

CPU小规模恢复逐元素相同；正式GPU跨进程恢复175→200轮的最大参数差为4.59e-6，
独立开发32/32仍完成，单回合RMSE最大变化约3.95e-7米。该路径报告浮点近似复现，
具体证据见`docs/verification/composable-lotf-resume-check.json`。

## 指标与查看器

```bash
pixi run status
pixi run status --run-id lotf-hybrid-hover-seed0-v1
```

TensorBoard服务已监听服务器`127.0.0.1:6006`。本地Windows使用已有SSH别名：

```powershell
ssh -N -L 6006:127.0.0.1:6006 lab-gpu
```

浏览器访问`http://127.0.0.1:6006`，选择两个`lotf-hybrid-`运行。主要标签：`training/loss`、`training/gradient_norm`、`eval/rmse_all_m`、`eval/last_second_rmse_m`、`eval/completion_rate`。曲线静态图和表也在`docs/verification/composable-lotf-figures/`及共同报告中。

VS Code查看器读取文件即显示；原生客户端也继续可用：

```bash
pixi run python scripts/rscope_client.py --directory experiments/lotf-hybrid-hover-seed0-v1/independent-heldout/rollouts --show-metrics
```

Windows客户端复用`--ssh_to lab-gpu`和现有OpenSSH配置。键盘左右切试次，上下切轨迹，空格暂停/恢复。普通查看无需发布到全局活动目录；主动发布仍用`pixi run replay --directory <路径>`。

## 验证与复算

```bash
JAX_PLATFORMS=cpu pixi run test
pixi run ruff check src scripts tests
pixi run python scripts/summarize_composable_lotf.py
```

最后一条只读取训练结果、生成图表并通过已有VS Code查看器读取器验证全部LOTF回放。新环境克隆时执行`git submodule update --init --recursive`和`pixi install`。LOTF原仓库GPLv3信息保存在子模块和THIRD_PARTY_NOTICES.md，当前集成为本地研究用途。
