# 来源、适用范围与复用选择

核查日期：2026-09-25。以下为作者文档、维护者源码或作者亲述经验。
本项目将经验落实到任务单、运行证据和验收，不照搬通用网站项目的工具架构。

## 文档与模块设计

Matt 仓库读取提交：`c55ee46073ed923f86ce59a5eb3b6d895095d1b7`。

| 来源 | 核查内容 | 本项目采用 |
|---|---|---|
| [setup](https://github.com/mattpocock/skills/blob/c55ee46073ed923f86ce59a5eb3b6d895095d1b7/skills/engineering/setup-matt-pocock-skills/SKILL.md) | 本地 Markdown 跟踪、docs/agents 配置、领域文档 | 独立仓库当前采用本地任务 |
| [本地跟踪模板](https://github.com/mattpocock/skills/blob/c55ee46073ed923f86ce59a5eb3b6d895095d1b7/skills/engineering/setup-matt-pocock-skills/issue-tracker-local.md) | .scratch/feature/spec.md、map.md、issues/NN 文件 | 当前规格和 11 个工作项的持久位置 |
| [to-tickets](https://github.com/mattpocock/skills/blob/c55ee46073ed923f86ce59a5eb3b6d895095d1b7/skills/engineering/to-tickets/SKILL.md) | 完整纵向能力、显式依赖、独立可验收 | 以完整飞行/训练/重评交付，模块分别维护 |
| [codebase-design](https://github.com/mattpocock/skills/blob/c55ee46073ed923f86ce59a5eb3b6d895095d1b7/skills/engineering/codebase-design/SKILL.md) | 小接口隐藏复杂行为、删除检验、接口即测试表面 | 五个实际模块；直接复用 Brax/rscope |
| [domain-modeling](https://github.com/mattpocock/skills/blob/c55ee46073ed923f86ce59a5eb3b6d895095d1b7/skills/engineering/domain-modeling/SKILL.md) | CONTEXT 只作术语，重要取舍简短 ADR | 术语、架构、规格、执行状态各有位置 |

`docs/status.md`、运行证据目录、TensorBoard 与质量轴是本项目针对研究工作的补充约定，
它们不是 Matt 强制规定的目录。原始 skill 中关于逐轮确认的流程已由本会话的范围确认与创建请求覆盖。

## 大型智能体项目的经验

**Anthropic：Effective harnesses for long-running agents。** 作者报告长期任务中提前宣布完成、
跨会话丢失状态与缺少实际验证的问题，采用功能清单、进度文件、初始化入口和实际检查。
本项目采用任务的外部行为验收、阶段状态及可恢复运行。文章属于工程经验，未提供本项目训练速度保证。
https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents

**OpenAI：Harness engineering。** 作者将仓库知识设为事实源，短 AGENTS 指向结构化文档，
让智能体读取运行反馈、验证真实应用行为并通过约束维持架构。
本项目采用短索引、真实运行清单和接口检查；研究协议与策略质量仍由研究者负责。
https://openai.com/index/harness-engineering/

**Simon Willison：First run the tests / Red-green TDD。** 作者强调执行测试、验证新测试确实能检出错误，
把测试作为新会话理解项目的入口。这里以与改动相关的现有测试为起点，避免重复完整耗时实验。
https://simonwillison.net/guides/agentic-engineering-patterns/first-run-the-tests/
https://simonwillison.net/guides/agentic-engineering-patterns/red-green-tdd/

**Addy Osmani：The 80% Problem in Agentic Coding。** 讨论代码生成快于理解带来的理解负担，
建议明确成功条件、自动验证和审查。这里让用户直接看完整轨迹、检查点重评及结果表，保持对研究目标的判断。
文章中的百分比属于作者讨论情境，本项目不据此预测开发效率。
https://addyosmani.com/blog/the-80-problem-in-agentic-coding/

**snarktank/ralph：公开的长期执行循环。** 仓库将任务完成状态、进度记录与 Git 历史持久化，
每轮选择一个未完成工作项、执行检查并更新状态；其默认每轮创建新上下文。
本项目借鉴持久任务和基于证据的完成标志，执行继续采用用户偏好的原会话优先恢复规则。
现有 Chat On Steroids/dsh 已承担会话执行，Ralph 此处作为设计参考。
https://github.com/snarktank/ralph

本轮同时检索 Linux.do 的大型项目/验收/长期任务讨论，未获得可稳定引用的直接原帖；
未将转述或搜索空结果作为具体技术证据。已采用的社区来源为维护者仓库和作者博客。

## 典型机器人学习项目怎样验证

| 项目 | 已核对的具体位置 | 验证方式与本项目借鉴 |
|---|---|---|
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/blob/main/learning/train_jax_ppo.py) | eval_env、deterministic inference、轨迹保存 | 独立评估环境与完整状态记录；当前 dump_rollout 注释问题需实测修复 |
| [LSY racing](https://github.com/learnsyslab/lsy_drone_racing/blob/main/scripts/sim.py) | simulate、log_episode_stats | 多回合调用原控制器、真实门序、完成时间和未完成记录 |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/develop/source/concepts/reinforcement_learning.html) | 训练/监控/检查点重放流程 | 真实整链检查→正式训练→TensorBoard→重载检查点；仅参考流程 |
| [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) | 独立评测环境、周期评估与随机种子 | 训练曲线与最终任务质量分离、检查包装器语义；仅参考评测 |
| [safe-control-gym](https://github.com/learnsyslab/safe-control-gym/blob/main/safe_control_gym/experiments/base_experiment.py) | BaseExperiment | 优化/学习控制共同的执行与轨迹指标；保持方法输入范围明确 |

## 平台已有实现清单

- [Crazyflow 函数式接口](https://learnsyslab.github.io/crazyflow/user-guide/functional-api/)：真实仿真与导数路径。
- [FigureEightEnv](https://github.com/learnsyslab/crazyflow/blob/main/crazyflow/envs/figure_8_env.py)：八字、奖励与终止。
- [RandTrajEnv](https://github.com/learnsyslab/lsy_drone_racing/blob/main/lsy_drone_racing/control/train_rl.py)：随机样条任务。
- [LSY race_core](https://github.com/learnsyslab/lsy_drone_racing/blob/main/lsy_drone_racing/envs/race_core.py)：竞速事件与状态。
- [sampling MPC](https://github.com/learnsyslab/crazyflow/blob/main/examples/control/sampling.py)：原生采样预测和真实控制执行。
- [AttitudeMPC](https://github.com/learnsyslab/lsy_drone_racing/blob/main/lsy_drone_racing/control/attitude_mpc.py)：acados＋Crazyflow 符号模型。
- [Brax](https://github.com/google/brax)：PPO/APG 训练；已继承实际短程验证报告。
- [SHAC 官方](https://github.com/NVlabs/DiffRL)、[JAX 参考](https://github.com/Andrew-Luo1/jax_shac)：算法与移植对照。
- [D.VA](https://github.com/HaoxiangYou/D.VA)：图像感知与梯度分离的方法参考，点云版本另记变体。
- [MuJoCo-LiDAR](https://github.com/discoverse-dev/MuJoCo-LiDAR)：MID-360 与射线实现候选，版本和实际几何需验证。
- [FlightBench](https://github.com/thu-uav/FlightBench)：静态视觉导航场景/指标参考。
- [SANDO](https://github.com/mit-acl/sando)：动态已知/未知障碍场景与完整方法参考。
- [rscope](https://github.com/Andrew-Luo1/rscope)：轨迹写入、原生读取、远程 SSH/SFTP 和桌面查看。

正式采用源码时固定提交与许可，复制受许可代码时保留原版权与许可证。当前只有来源入口和历史
证据固定；待接入的上游版本在各任务开始时核实，不把 main 链接当成稳定版本锁。
