# 可组合模块工程记录

基准`bd14468`，功能分支`implementation/composable-lotf`。五模块一起实施，原P1–P4运行目录只读复用。

## 已执行检查

- 迁移前全套47项通过，日志`tmp/composable-flight/baseline-tests.log`。
- 三任务×四模型固定初态/动作12项回归通过，数据在`tests/fixtures/composition-baseline.*`，原source_commit为bd14468。
- LOTF原生前向、代理雅可比、原任务、BPTT单步及Adam/随机数、完整恢复已逐项对照。
- Hydra实际PPO4096交互；APG/SHAC各2次更新、64交互；结果在`experiments/composable-*-probe-v1`。
- 实际AttitudeMPC通过5次原生过门，857步；采样MPC通过5门，850步。完整128试次旧结果保持原身份。
- 两任务正式训练600万/2250万及各独立32+128评测完成，见共同交付表。
- RScope Viewer原读取器完整读回22个文件、103条轨迹，位置误差0，参考点逐元素相同。
- 新旧P2检查点各32回合重评，参数哈希相同，逐回合RMSE变化低于3e-9m。

## 迁移中发现并处理

1. 模块替换后旧数值故障测试仍写入已不用的`env.step_fn`。故障注入改到真正的`env.model.advance`接口，
   继续验证8.119e26角速度失败与重置；生产数值规则保持原定义，废弃任务step_fn字段移除。
2. Hydra批量预设在全局校验时以默认PPO组检查字段；不同算法共有的窗口覆盖使用显式`++algorithm.horizon_length`。
   修正命令后真实执行两项批量任务，全部达到声明的更新数。失败命令在原日志中保留。
3. 旧LOTFJVP随机键切向量与现代JAX不兼容，只调整为float0，原p/R/v切向量通过完整雅可比对照。
4. 原生气动项在零水平速度的直接导数非有限，保留原物理并用非奇异状态检查直接导数；正式采用解析代理。
5. 冻结检查点评测增加显式`evaluation.environment`选择，默认沿检查点，指定experiment时才使用当前环境配置，
   已保存网络/算法身份和输入维度分别核对。
6. 原始持久配置中的未使用时域字段与当前`rollout=full_episode`在恢复中按实际算法语义匹配。
   学习率、模型、场景、策略、任务或环境数变化会拒绝精确续训。
7. 原生CSV按索引读取，频率覆盖会改变实际参考速度。配置现在固定LOTF跟踪50Hz，其它频率需显式重采样。
8. 算法由更新数推导预算时，旧默认num_timesteps可与实际预算冲突。默认改null，显式预算必须一致。
   冻结历史检查点评测独立采用evaluate模式，保存的训练字段只作历史记录。
9. 续训重新初始化开发选模会丢失更早最佳策略；现在从源记录中仅收集恢复时刻以前的dev报告，
   校验对应策略后带入新运行。独立resume-selection.json保留继承来源，即使同一步重新评测也可追溯。
10. 非有限状态、观测或回报会使JSON和回放无法导出。现在数值异常明确计失败，保留最后有限姿态，
    无法定义的数值转移回报记0并附说明。该修正后重新评测两组各128留出，与原逐回合RMSE完全一致。

## 来源与许可

`third_party/learning_on_the_fly`固定cba6e537，原文件未改。子模块已经解除临时Git对象借用，fsck通过。
LOTF模块和改编BPTT来源按GPLv3标记。当前只有本地集成与本地提交，没有外部发布。
Crazyflow固定36f584d，版本0.3.2；Brax0.14.2/JAX0.9.2/Flax0.12.6/Optax0.2.8保持原版本。
新增Hydra1.3.7、jax-dataclasses1.6.2及原生LOTF导入依赖；Pixi锁已更新，frozen安装通过。

## 当前验证限制

独立ChatGPT代码审查已尝试并恢复原worker；本轮再次邀请worker-1后仍因浏览器未接收命令而休眠，
未产生独立审查结论。源码审查与针对性反例由主执行者完成，测试、上游对照和独立进程评测是可核验的证据。
旧Windows查看器能力继承已验收版本，本轮验证其数据读取器及生成轨迹的可视化图。
多训练种子和独占资源计时仍按总体P6计划执行。

CPU小规模保存恢复与连续更新逐元素一致；正式GPU归档快照175→200轮继续计算，最大参数差4.59e-6，
开发集均32/32，逐回合RMSE最大差3.95e-7米。早期1e-6阈值检查失败日志完整保留；
完整GPU逐元素一致尚未成立，数值状态恢复与该严格确定性指标分别报告。

最终79项测试、2个子测试通过，377.10秒；命令与结果在`tmp/composable-flight/acceptance-final-tests.log`，
原始日志另归档为[测试日志](composable-lotf-tests.log)。前一轮完整79项通过位于
`tmp/composable-flight/acceptance-tests.log`。统计、摘要和文档检查在[最终检查](composable-lotf-final-checks.json)中记录。
