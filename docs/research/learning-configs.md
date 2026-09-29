# 可复用学习配置的来源与映射

核查日期：2026-09-29。下列仓库位于项目外 `reference_repos/`，均建立了 GitNexus 索引，再核查实际配置和训练实现。配置参考不等同论文复现；网络输入、动作定义、动力学、损失和预算都必须随适配配方记录。

| 来源与固定提交 | 实际网络／训练配置 | 本项目采用方式 |
|---|---|---|
| [DiffAero](https://github.com/flyingbitac/diffaero/tree/291ea14196aefbebcf7387dd71f7e096c83878b7) | `cfg/network/mlp.yaml` 为[256,128]；`utils/nn.py`实际为LayerNorm＋ELU。SHAC actor/critic lr=0.001/0.003，Adam默认β，γ=.99、λ=.95、actor裁剪1、8个critic minibatch；目标网络每轮混入.005新参数。默认1024并行×32步×1000次=32,768,000步。 | 提供 `network=diffaero_mlp algorithm=shac_diffaero_adapter` 独立确定性JAX适配；首次用于SHAC竞速t2。早期基线128并行=4,096,000步；t2实际64并行×320次=655,360步（Hydra覆盖偏差已登记），不能按名义1000次声称同等预算。 |
| [VisFly](https://github.com/SJTU-ViSYS-team/VisFly/tree/6ff29bfcbc752da4f9474f6f0ece29d836111a22) | `exps/examples/alg_cfgs/cluttered_flight/PPO.yaml`：深度特征[128]，state/target分支[128,64]，actor/value[64,64]，ReLU；lr=5e-5，256步展开、10 epochs、batch=25600、γ=.99、GAE=.95、clip=.2、gradclip=.5、weight decay=1e-5、总计10M步；配套环境48 agents、64×64深度、.03s控制周期。 | 作为视觉PPO后续参考；目前通用PPO跟踪／竞速已经通过开发检查，没有为套用外部配置而替换其有效配方。视觉任务及网络分支需要另行适配。 |
| [VisFly-Lab/APG](https://github.com/Fanxing-LI/APG/tree/e8a3c0da81d288ac9879759dea4b354a3a8306a6) | `examples/racing/cfg/shac.yaml`：state特征[192]、actor/critic[192,96]、ReLU，lr=.001、horizon=96、100 agents、γ=.99、τ=.005、10 critic更新、10M步；tracking同类配置4M步。BPTT竞速配置为512步时域。 | 可以借鉴网络容量和时域；不直接移植其更新规则。实际 `algorithm/shac.py` 将送入target critic的obs与action都detach，actor末端bootstrap没有沿状态回传梯度。我们的SHAC保留该梯度，已有回归测试，不能把两者称作同一个实现。 |
| [DiffRacing论文](https://arxiv.org/html/2603.08019v1#S3.SS5)；[项目页固定源码](https://github.com/nbsy000/DiffRacingWeb/tree/d82683f8a064d39e45f1172f5e3964fb96a4147e) | 论文给出24×32深度、12维状态、CNN与状态MLP、192维GRU、LeakyReLU(.05)、3维机体系加速度；另有32维GRU的Delta Action模块。训练方法含AVF梯度增广。核查到的项目页中Code按钮仍为`href="#"`，没有可据此获取的训练配置。 | 网络结构可作为未来视觉竞速参考；不能杜撰优化器或预算，也不能将普通BPTT重命名为DiffRacing。用户举例中的“diffrace”按这一相关项目检索，未据此认定唯一指代。 |

## DiffAero适配中的实质差异

`shac_diffaero_adapter` 保留本项目任务、奖励、延迟、动作缩放和确定性actor，不含上游随机actor的熵正则。actor采用[256,128]和LayerNorm／ELU；critic仍为Brax无LayerNorm的V网络，初始化也沿用Brax。critic是每轮8次全批更新，不是上游的8个互不重叠minibatch。保留critic裁剪10；这些差异都可能影响收敛。

本项目 `target_critic_alpha` 定义为旧参数权重：`target = alpha*old + (1-alpha)*new`，因此对应上游 `.005` 的值是 `.995`，不能直接照抄 `.005`。演员Adam β新增配置入口，旧配方仍使用(.7,.95)，参考配方显式使用(.9,.999)。

本次t2同时改变网络宽度、critic学习率、演员优化器、调度和目标网络更新速度。这是一组来源明确的候选配方，与t0/t1比较时不能推断某一个参数单独导致差异。若t2失败，已达到本轮seed=0基线＋2次调整的上限，应携带曲线和回放共同诊断。

## 实验记录

预算和选模依据见[开发训练合同](../training-pilot.md)。每次实际运行保存解析配置、源码快照摘要、设备、训练步数、快照评测和optim-agent ask/tell记录。学习方法3个独立训练种子及每任务100个留出回合仍须按[第一版规格](../release-plan.md)完成。

深度导航t0为独立CNN／GRU适配：8并行×32×1000，434.16秒完成。第875次更新在固定Navigation8的8个开发回合全部到达，按预先约定选为候选；第1000次更新又有失败，因此不能只发布最终权重或宣称稳定收敛。这个结果没有覆盖独立初态扰动、新几何或3个训练种子。

SHAC竞速t2在实际第320次更新达到32/32开发回合通过。名义启动预算与解析配置不一致，参见[偏差记录](../training-pilot.md)；后续三种子使用实际已验证的64并行／320次配方，不把该结果伪称1000次运行。
