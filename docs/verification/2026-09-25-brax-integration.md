# 归档与迁移说明

以下为原 Crazyflow 工作区的接入测试报告，测试条件及数值保持原身份。
原始 JSON、日志、失败记录、模型资源已复制到 `experiments/evidence/brax-20260925/`；
逐文件校验见 [证据清单](brax-20260925-evidence-manifest.json)，小型结果副本见 [结果摘要](brax-20260925-results.json)。
原工作区 tmp 的环境及脚本仍在原位置，文末复验命令以原 Crazyflow 目录执行。

原 requirements 快照记录了 Git 提交，但未包含未提交的依赖变更；现已保存
[crazyflow-dependency-changes.patch](crazyflow-dependency-changes.patch)。新平台的正式 Pixi 锁定在任务 01 完成。

---

# Crazyflow + Brax：接入验证

日期：2026-09-25。Crazyflow 提交：`36f584d114d9d331f0cee0fe4b9066f821c0fbfd`。

## 结论

Brax 可以作为本平台唯一训练库。原生 PPO 和 APG 已在真实 Crazyflow FigureEight 任务上完成参数更新，
CPU 覆盖四种动力学，RTX 4090 覆盖第一性原理动力学。rscope 原生文件写入、读取和模型重建通过。
本报告证明依赖组合、环境接入和训练更新链可执行；任务收敛、正式成功率和 SHAC/D.VA 尚未验证。

## 依赖组合

| 依赖 | 本轮通过版本 |
|---|---|
| Python | 3.14.7 |
| Crazyflow | 0.3.2，本地可编辑安装 |
| Brax | 0.14.2 |
| JAX / JAXlib / CUDA 12 插件 | 0.9.2 |
| Flax | 0.12.6 |
| Optax | 0.2.8 |
| MuJoCo / MJX | 3.14.0 |
| rscope | 0.0.8 |

独立环境 `tmp/brax_probe_20260925/venv_clean` 的依赖检查输出：
`No broken requirements found.`
完整包锁定快照见 `tmp/brax_probe_20260925/requirements-verified.txt`。

原 Pixi 环境的 JAX 0.11.1 + Flax 0.12.9 配 Brax 0.14.2 时，
PPO/APG 的四模型运行均因 `jax.device_put_replicated` 被移除而失败。
原结果保存在 `original_jax011/`。Brax 和 Crazyflow 核心代码均沿用原实现；通过版本锁定解决该兼容问题。

中间的共享依赖测试环境继承了原项目的 splax；它要求 JAX >=0.10，因而有独立包约束冲突。
最终 GPU 测试在全新、无继承的环境运行，该环境仅安装 Crazyflow、Brax、rscope 及其依赖，未包含 splax。
当前项目 Pixi 配置和锁文件的既有改动保持原状，本次没有将该验证环境覆盖到原环境。

## 临时适配范围

`probe.py::FigureEightAdapter` 只封装环境接口，不实现 PPO/APG 算法。
复用官方 FigureEight 参考轨迹、随机重置流水线、动作物理范围、奖励函数和失败函数。
Crazyflow `Sim` 仅用于构建，计算使用 `crazyflow.sim.functional` 及已构建的 step/reset 函数。
单世界内部形状通过 Brax 的向量化变成批量环境，观测展平为 43 维。

测试物理步为 500 Hz，环境决策为 50 Hz。默认机体为该环境的 `cf2x_L250`。
与原对象式环境比较的是相同初态下 20 步、终止前的前向轨迹；这不宣称所有训练重置语义等价。
Brax 默认自动重置复用首次初态；正式环境若继承 Crazyflow 每回合重新随机化，需要显式适配。

## CPU：12 个组合全部通过

每种动力学各完成前向/梯度/回合边界检查、2048 次交互的原生 PPO、4 次更新的原生 APG。
PPO 为 16 个并行环境；APG 为 8 个环境、16 步反传窗口。表中差值是初始与最终策略参数的 L2 距离。

| 动力学 | 前向与边界 | PPO 参数变化 | APG 参数变化 | 有限差分相对误差 |
|---|---|---:|---:|---:|
| `so_rpy` | 通过 | 0.05136485 | 0.11945597 | 0.00053% |
| `so_rpy_rotor` | 通过 | 0.04836278 | 0.12126925 | 0.08987% |
| `so_rpy_rotor_drag` | 通过 | 0.04827152 | 0.12142504 | 0.12062% |
| `first_principles` | 通过 | 0.04717141 | 0.12374366 | 0.20006% |

四模型对官方环境的最大观测误差：`1.519918442e-06`；最大位置误差：`7.450580597e-09`；
最大奖励误差：`0`。
所有模型都检查了 8 世界批量形状、非零有限梯度、3 步时间截断标志和后续回合计数复位。
有限差分使用 12 步高度损失、推力动作、中心差分步长 0.003；它是局部导数检查，不覆盖全部状态/接触分支。

## RTX 4090：3 个组合全部通过

设备枚举实际返回 `CudaDevice(id=0)`，使用全新隔离环境。

- 第一性原理环境的前向等价、批量、有限差分和截断复位检查通过。
- PPO：256 世界，8192 次交互；策略参数变化 `0.02613126`。
- APG：8 世界、16 步窗口、4 次策略更新及原生评估；参数变化 `0.12371258`。

参数全部为有限数。短程 PPO 的 KL 指标出现较大值（当前 GPU 探针汇总约
198.00），因此探针超参数仅用于接口验证，不作为正式训练配方。
所报 elapsed/walltime 包含首次编译等开销；本次没有形成训练吞吐性能排名。
GPU 日志包含驱动版本字符串解析告警，实际 CUDA 计算及上述断言通过。

## rscope 与现有采样 MPC

- rscope：原生 `rscope_init` / `dump_eval` / `append_unroll` / `load_model_and_data` 完成往返。
- 记录为 64 帧、1 世界、0.02 秒采样间隔，包含位置映射的 mocap、姿态、观测、回报和误差项。
- 重建模型包含 12 个资源条目，回放末帧 mocap 误差为 0.0。
- Crazyflow 已注释说明其无几何 dummy body 的惯量会在 `to_xml` 导出中丢失。
  导出适配器从已编译模型恢复该 body 的原始质量/惯量，随后 rscope 模型重建通过。
- 本轮验证文件消费者和模型状态重建；桌面窗口、Windows 与 SSH 实际连接留待部署测试。
- 原版 `examples/control/sampling.py` 在 CPU 完成 32 个候选、2 次决策的缩小接入测试，位置均有限。
  该测试保留原控制函数及预测时域，只缩小采样数和运行时长；正式避障成绩、实时性未测。

## 发现并解决的问题

1. Brax 的旧 JAX 数据复制入口：通过 JAX 0.9.2 / Flax 0.12.6 组合解决。
2. 临时环境适配器覆盖 metrics 字典，丢失 Brax 评估包装器加的 reward 字段：改为保留已有 metrics 键。
3. Crazyflow XML 重建丢失 dummy body 惯量：仅在导出层恢复模型中的精确数值。

另核验了同组竞速/MPC 的旧导入：`crazyflow.drones.load_params` 在本地 0.3.2 中不存在。
这是一处已确认的版本适配需求；本轮对竞速和 acados MPC 的其他代码属于源码检查，未声明完整运行通过。

## 重复执行

在 Crazyflow 根目录运行，验证环境已经位于本仓库的 tmp 下：

```bash
env -u PYTHONPATH SCIPY_ARRAY_API=1 JAX_PLATFORMS=cpu \
  tmp/brax_probe_20260925/venv_clean/bin/python \
  tmp/brax_probe_20260925/probe.py --case ppo --dynamics first_principles

env -u PYTHONPATH SCIPY_ARRAY_API=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  tmp/brax_probe_20260925/venv_clean/bin/python \
  tmp/brax_probe_20260925/probe.py --case apg --dynamics first_principles --device gpu

env -u PYTHONPATH SCIPY_ARRAY_API=1 JAX_PLATFORMS=cpu \
  tmp/brax_probe_20260925/venv_clean/bin/python \
  tmp/brax_probe_20260925/rscope_probe.py
```

所有原始 JSON、日志、失败记录和导出轨迹保存在 `tmp/brax_probe_20260925/`。
原始算法、环境、依赖和探针分别记录，后续正式实现只保留所需适配，并保留此报告中的兼容结论。
