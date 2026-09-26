# 运行与远程查看

项目路径：`/home/tong/tongworkspace/simulation_dev/mujoco/drone_playground`。
以下命令使用本项目独立 Pixi；若 shell 找不到 Pixi，使用 `/home/tong/.pixi/bin/pixi`。
开始时进入项目目录，SSH 地址 `SERVER` 使用你平常能连接的服务器地址。

## P3/P4：任务、训练、优化控制

当前任务表在 `.scratch/drone-platform/map.md`，阶段状态在 `docs/status.md`。
学习方法通过同一个训练入口运行，配置决定方法、任务与动力学。

```bash
env -u PYTHONPATH pixi run train --config configs/experiments/figure8_shac.json --run-id my-shac-tracking
env -u PYTHONPATH pixi run train --config configs/experiments/racing_ppo.json --run-id my-racing-ppo
env -u PYTHONPATH pixi run train --config configs/experiments/racing_apg.json --run-id my-racing-apg
env -u PYTHONPATH pixi run train --config configs/experiments/racing_shac.json --run-id my-racing-shac
```

每个运行标识对应一个新目录。已有实验优先直接查看，以上命令会启动新训练。
`scripts/run_p3_matrix.py` 引用原4个P2结果并完成其它20个单元；
`scripts/run_p4_learning.py` 负责三个竞速策略和新进程独立重评。两者共用单GPU训练队列锁。
父队列进程的状态JSON指向真实子运行；子运行 `state.json` 含PID/启动标记、阶段、实际步数和心跳。

优化控制先局部构建 acados 再执行真实闭环：

```bash
bash scripts/setup_acados.sh
env -u PYTHONPATH SCIPY_ARRAY_API=1 JAX_PLATFORMS=cpu \
  .pixi/envs/default/bin/python scripts/evaluate_racing_control.py \
  --controller attitude_mpc --episodes 32 --split dev --run-id my-attitude-mpc

env -u PYTHONPATH SCIPY_ARRAY_API=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  .pixi/envs/default/bin/python scripts/evaluate_racing_control.py \
  --controller sampling_mpc --prediction-device gpu --samples 2000 \
  --episodes 32 --split dev --run-id my-sampling-mpc
```

`scripts/run_p4_optimization.py --version v2 --workers 4` 每方法4份32试次，
严格核对互斥种子后合并128试次。已有v2结果请读取而非重复运行。
预测耗时来自实际求解，首决策编译单列；当前同步评测不把墙钟延迟注入动力学。

竞速原生配置为 Level0：固定门序[1,2,3,4,2]，4个实体门，50Hz控制，最多30秒，
保留 `env.disturbances` 的动作/外力扰动。三个学习方法共用作者18.75秒参考样条，
43维状态与未来参考观测；学习回报是具名的参考跟踪配方，成功由真实过门事件判定。
开发集选模为完成数→平均门进度→误差；固定赛道的128试次是独立扰动种子，不是128条新赛道。

## VS Code 查看阶段结果

在已安装 RScope Viewer 的远程工作区，点击各运行 `independent-heldout/rollouts/` 内的 `.mj_unroll`。
优化控制合并目录的 `rollouts/shard-0/` 至 `shard-3/` 各自保留独立模型与记录，可用文件夹浏览选择。
文件内显示实际保存的4–5条轨迹（最差试次与前4重合时不重复），全部32/128回合指标保存在独立报告中。
训练时间点位于 `rollouts/step-*/`；APG 保存初始和最终策略，周期评估指标另外记录。
SHAC 还保存 `training-state/update-*.pkl` 完整环境/随机数/优化器状态，恢复一致性已有测试。

P3/P4 纹理、参考线、过门/碰撞指标随轨迹保存；各运行不会自动抢占 `/tmp/rscope/active_run`。
TensorBoard 沿用下方6006端口转发，选择 `p3-`、`p4-` 运行查看。
历史 `heldout-v1` 优化运行因扰动配置层级错误中止保留，正式优化结果为 `heldout-v2`。
修复和证据见 `docs/verification/p3-p4-engineering.md`。

读取并重新校验当前29项结果表，不执行训练：

```bash
env -u PYTHONPATH SCIPY_ARRAY_API=1 JAX_PLATFORMS=cpu pixi run python scripts/summarize_p3_p4.py
```

当前完整结果是 `docs/verification/p3-p4-results.json`。P3恢复单元使用v3显式数值失败边界与原学习参数。
SHAC竞速初始策略被开发集选中；训练结束的策略可在同运行 `rollouts/step-0000655360/` 查看。

## 现在查看结果

```bash
cd /home/tong/tongworkspace/simulation_dev/mujoco/drone_playground
pixi run status
pixi run status --run-id p2-figure8-ppo-seed0-v2
```

状态核验进程 PID、系统启动身份和进程启动时间，显示最近更新与实际步数。
已完成的训练以 `result.json` 为结果来源，活动进程以 `state.json` 和日志为来源。

TensorBoard 已在服务器 `127.0.0.1:6006` 提供服务，真实 HTTP 与标量数据已验证。
Windows PowerShell 使用已有 SSH 连接转发端口：

```powershell
ssh -N -L 6006:127.0.0.1:6006 tong@SERVER
```

随后在浏览器访问 `http://127.0.0.1:6006`。选择以 `p2-` 开头的已完成正式运行；
`p1-` 是整链探针，`p2-figure8-ppo-seed0-v1` 是为修复初始随机数键而中断的历史运行。
重点查看 `eval/completion_rate`、`eval/rmse_all_m`、`eval/return` 和 `training/*`。
APG 的原生 `eval/episode_tracking_error` 为回合误差和；32 回合位置 RMSE 使用本项目独立评测结果。

## 选择与重放

在服务器选择已经保存的结果供远程 rscope 拉取：

```bash
pixi run replay --directory experiments/p2-figure8-ppo-seed0-v2/independent-dev/rollouts
```

它发布到 `/tmp/rscope/active_run`；原始运行文件保留完整。启动一个新训练运行时可按配置自动选中，
后续检查点增量发布；查看者切换到其他运行后，训练继续保存自己的记录并跳过当前活动目录。

在服务器已有桌面上重放：

```bash
pixi run python scripts/rscope_client.py \
  --directory experiments/p2-figure8-ppo-seed0-v2/independent-dev/rollouts \
  --show-metrics
```

这个启动器仍使用 rscope 的原生查看器和交互。它针对固定的 rscope 0.0.8/MuJoCo 3.14，
在运行内移除包住 UI 方法的重入锁，只在直接写仿真状态时加锁，安装包文件保持原始字节。
直接运行旧的 `python -m rscope` 在这组版本会停在首帧；使用这里已修正的启动器。

## Windows 原生客户端

复制 `scripts/rscope_client.py` 到本地，仅安装查看器依赖即可。脚本会调用系统 `ssh -G`，
因此可以直接复用 Windows `~/.ssh/config` 中已经能工作的 Host 别名和加密私钥。

```powershell
python -m pip install "rscope==0.0.8" "mujoco==3.14.0" paramiko
python F:\code\rscope_client.py --ssh_to lab-gpu --show-metrics
```

`lab-gpu` 可以替换为任意已有 OpenSSH Host 别名，也仍支持 `username@host[:port]`。
主机公钥必须已在本地 `known_hosts` 中受信任；加密私钥会提示输入一次 passphrase。
客户端默认创建独立本地临时缓存，远端始终使用 Linux 路径；开始前先认证，失败会直接退出。
它读取 Python pickle 记录，限于自己可信的服务器与实验产物。

左右方向键切换试次，上下方向键切换保存的策略轨迹，空格暂停/继续，Shift+M 切换指标。
PPO 保存初始、中间和最终轨迹；原生 APG 提供周期评估标量，策略轨迹保存初始和最终两个时间点。
tracking rollout 默认把完整参考轨迹画成红色 3D 线；早期策略提前失败时，会从其它 checkpoint 的
同一 case 选择参考点最完整的一条，因此仍能看到完整八字/样条。`--hide-reference` 可关闭该叠加。
切换到不同模型的记录后重开查看器，使其加载对应模型资源。

## 独立重评保存策略

每次评测使用新的输出目录；已有目录会拒绝覆盖。下面的八字 PPO 检查点由开发集选中：

```bash
pixi run evaluate \
  --checkpoint experiments/p2-figure8-ppo-seed0-v2/checkpoints/step-0001310720.pkl \
  --split heldout --episodes 128 --device cpu \
  --output experiments/p2-figure8-ppo-seed0-v2/manual-heldout-01
```

已完成的独立评测位于每个正式运行的 `independent-dev/` 与 `independent-heldout/`，
包含完整回合表和自包含重放；无需为查看结果重复运行评测。
固定开发初态从 20000 开始，留出初态从 30000 开始；随机样条的参考轨迹也使用独立种子范围。

## 重复训练

使用新的运行标识，保留既有结果。四份冻结配方均完成过真实训练：

```bash
pixi run train --config configs/experiments/figure8_ppo.json --run-id my-figure8-ppo --device gpu
pixi run train --config configs/experiments/figure8_apg.json --run-id my-figure8-apg --device gpu
pixi run train --config configs/experiments/random_ppo.json --run-id my-random-ppo --device gpu
pixi run train --config configs/experiments/random_apg.json --run-id my-random-apg --device gpu
```

按配置各自消耗 2097152、655360、4194304、655360 次任务交互，物理子步另计。
回放和独立评测冻结策略及归一化；PPO 的 `--warm-start` 沿用 Brax 的参数恢复方式，
优化器、随机数和回合状态重新初始化。APG 的原生训练入口当前提供推理检查点，精确续训未实现。

## 测试与服务

```bash
JAX_PLATFORMS=cpu pixi run test
pixi run ruff check src scripts tests
pixi run demo --run-id my-native-demo --duration 10 --device cpu
pixi run metrics --port 6006
```

现有 TensorBoard 已占用 6006，查看即可；重启服务前核对所属进程。停止查看器只影响显示，
学习算法、检查点和本次运行记录独立存在。测试覆盖真实任务、导数、回合、参数重载、文件发布和查看器兼容。
