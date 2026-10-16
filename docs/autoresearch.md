# Autoresearch Gym

`tools/autoresearch.py` 把 Autoresearch Gym 的外部后端接到本项目 CLI。候选修改训练参数，
本项目继续执行 PPO、APG/BPTT 或 SHAC，保存 Orbax 权重并运行冻结评测。

## 安装

先按项目 README 安装 Pixi 环境。`uv` 和 `git` 需要在 PATH 中，然后在项目根目录执行：

```bash
pixi run autoresearch setup
```

脚本将上游克隆到 `autoresearch-gym/`，固定到
`7fd6391aecc3bf0ab8e8813cf4da7a053baa001f`，并应用
`tools/autoresearch/upstream.patch`。重复运行不会重复打补丁。
上游运行器使用其 Python 3.10 环境；训练使用本项目现有 Pixi Python 3.12。
两套依赖保持独立。克隆目录被 Git 忽略，接入代码和补丁由本项目管理。

## 运行一个候选

下面的短程配置执行 16 次 PPO 更新，共 8,192 次环境交互，然后在 S01、S02、S03、
D01、D02、D03 各运行两个完整回合。它检查训练、保存、加载和计分，不作收敛验收。

```bash
pixi run autoresearch run \
  --benchmark tools/autoresearch/navigation_depth_smoke.json \
  --tag depth-smoke --no-dashboard
```

默认配置 `tools/autoresearch/navigation_depth.json` 使用 409,600 次交互，六个主场景各评测
8 回合。运行前在该配置里确定预算；同一搜索会话保持预算和评测条件不变。

```bash
session=results/autoresearch/sessions/depth-ppo
mkdir -p "$session/candidates"
cp tools/autoresearch/seed_trainable.py "$session/candidates/pass01_baseline.py"
pixi run autoresearch run \
  --session-dir "$session" \
  --candidate "$session/candidates/pass01_baseline.py" \
  --tag pass01-baseline --no-dashboard
```

后续由 Coding Agent 根据本会话结果，逐次编写候选并调用同一命令。上游运行规则见
`autoresearch-gym/AUTORESEARCH.md`。此接入采用固定交互预算和最后一次更新的权重；
普通训练仍采用原来的三阶段收敛选模。

## 固定条件与可修改参数

任务 JSON 固定实验、算法、交互预算、评测场景和种子。候选文件的 `recipe` 可以修改
`learning.options.*`、`method.actor.*` 和 `simulation.num_envs`。
`learning.budget_only=true` 使外部候选训练按固定预算执行、不进行中途选模。
`max_interactions` 必须是 `num_envs × horizon` 的整数倍，避免改变并行度后悄悄超出预算。
普通 PPO 训练可用相同的 `max_interactions` 作为硬上限，同时保留训练期间的筛查与选模。
候选从随机权重开始，不能覆盖初始权重、恢复路径或评测条件。

搜索使用 `checkpoint_eval` 种子段，主指标为六场景平均成功率，并保留最差场景成功率和
碰撞率。评测初态不随训练种子变化。确定配方后，再按现有独立种子和正式 Benchmark
流程确认质量；反复用于搜索的评测集不能同时充当最终测试集。

换用 LiDAR、Tracking、Racing 或其他算法时，创建对应任务 JSON，指定现有实验配置和算法。
本地 BPTT 的算法配置名为 `apg`。本次提供的开箱配置为程序化 Depth PPO。

## 结果

无会话运行写入 `results/autoresearch/runs/`；会话运行写入指定会话的 `runs/`。
每轮的 `external/train/` 和 `external/eval/` 使用本项目的原生结果布局。
`summary.json`、`results.jsonl` 和 `outer_loop_log.md` 供 Coding Agent 比较候选。

`budget_exhausted` 表示完整预算已执行，导出的是该预算末尾的冻结 Actor，不表示通过
正式验收。训练曲线继续保存在原生 TensorBoard 文件中；上游汇总保留真实交互数、更新数和
最后一次训练指标。当前原生评测不报告累计 reward，上游对应字段为 `null`。

这个适配器在 GPU 主机本地运行，使用 `--no-dashboard`。回放继续使用现有原生工具。
上游补丁只处理外部 checkpoint 路径、交互预算元数据和未提供的训练回合曲线。
