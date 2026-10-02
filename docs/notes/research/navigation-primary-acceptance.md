# 导航主要场景验收

用户于2026-09-30修订第一版范围：S01／S02／S03、D01／D02／D03各自成功率≥90%即可通过；S06／D06作为扩展场景完整报告，不影响判定。三个学习训练种子仍逐个通过，跟踪、竞速要求不变。

这是原留出结果已知后的标准调整。`release-navigation-v1`报告与原判定保持不变，新增`release-navigation-primary-v1`复核明确记录调整时间、原报告摘要和原判定。没有删除失败回合、替换策略或重新选择快照。

每任务仍至少100个留出回合，覆盖四个场景。当前25次/场景，其中主要场景75、扩展25；主要场景逐个判断，不能用整体平均掩盖其中一个低于90%的场景。新校验继续检查所有八个场景、独立种子、实际初态与清单的一致性及有效结局。

已有深度32并行配方按新规则通过静态／动态两格。训练种子30／31／32的主要静态场景均100%；主要动态场景最低96%。完整任务结果仍为静态80／93／100、动态84／100／100，S06为20%／72%／100%，D06为44%／100%／100%。600份原回放凭据复用，原报告和相关认证文件重新核对摘要；见[原判定](../../artifacts/verification/stage2-depth-confirmation.json)和[新复核](../../artifacts/verification/navigation-primary-acceptance.json)。

后续学习评测在现役 experiment 上显式设置 `+evaluation.benchmark_id=navigation-primary-v1`；原生评测设置 `+evaluation.protocol=benchmarks/navigation.yaml +evaluation.benchmark_id=native-navigation-primary-v1 +evaluation.per_scene=true +evaluation.initial_conditions=null evaluation.episodes=null`。空的回合数和初态设置由 `benchmarks/navigation.yaml` 解析为每场景25回合及协议分布。原生版本身份和真实执行检查继续生效；历史结果不重新标记。

后续训练直接在对应 experiment 上设置 `training.checkpoint_eval_metric=navigation-checkpoint_eval-primary-v1`；每场景8个 checkpoint_eval 回合和种子起点由 `benchmarks/navigation.yaml` 提供。选模分数只使用六个主要场景的最差与总到达数，扩展回合不参与打破并列。正在运行的实验保持原冻结规则，完成后复核其原选择；不在中途改选模规则。checkpoint_eval通过仍需新的三个训练种子与独立留出确认。

固定Navigation8与独立初态／扰动验收不构成未见几何泛化。源码来源、传感与动力学配方差异继续分别登记；本次范围调整不改变论文原方法复现的完成状态。
