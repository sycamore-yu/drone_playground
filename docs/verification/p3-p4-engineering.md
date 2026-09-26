# P3/P4 工程证据

状态：工程实现和单训练种子首轮实验已完成。29项正式实验均完成，25项达到当前任务门槛。
完整交付与计时边界见 [交付记录](p3-p4-delivery.md)，最终矩阵见 [实际结果](p3-p4-results.md)。

## 来源与最小适配

- Crazyflow 0.3.2：36f584d114d9d331f0cee0fe4b9066f821c0fbfd。
- LSY：b1f5b36adb8e08e8e2adea85de790bd0e0a1d118；源码/许可/差异文件位于各 lsy_upstream 目录。
- Brax 0.14.2、JAX0.9.2、Flax0.12.6、Optax0.2.8；训练主依赖保持原版本。
- acados0.5.1：48e223e85f0408ebfd1d8c6d6fb0589e9c41b3aa；项目局部构建。
- 模板渲染器0.2.0 linux-amd64 SHA256：1973ddd1b536dcd4a059a22f7259c3b0e6b6c878cbfbe7e9b962eba195f361d9。

LSY 的 params 入口变更有独立兼容代码。acados 的矩阵、25步/0.5秒时域、SQP/HPIPM以及
源代码两处推力×4因子保留；每次运行的代码生成目录隔离。构建用 RPATH 定位私有共享库，
future-fstrings 与 matplotlib 只安装到本项目 tmp/p3p4/optimization/python-deps。

## SHAC

短窗口目标按回合片段累加折扣奖励与片段终点价值；目标网络参数冻结，终点状态梯度保留。
真实终止清零自举，时间截断使用重置前状态；TD-λ 在回合边界停止跨回合递推。
每次策略更新从脱离前一窗口梯度的实际状态开始；保存策略、价值/目标价值、优化器、归一化、
环境及随机数完整训练状态，恢复后与连续更新的最终参数逐元素完全一致。

tests/test_shac.py 覆盖闭式回报、价值参数/状态梯度、片段边界、实际 actor/critic 更新、
公共推理检查点、完整恢复。证据 tmp/p3p4/shac-resume-test.log。

## 竞速任务及导出

保留 Level0 4个实体门、[1,2,3,4,2]顺序、单向穿越与0.45米判定框，30秒最大回合。
场景接触由原生 MJX 几何检查生成。单机任务成功/失败后保留真实末状态，避免多机核心用于隔离
碰撞的“移到地面下”操作污染评测末帧。训练回报明确为 reference-tracking-v1，原生稀疏回报另存。
策略观测的未来参考按当前控制 tick 采样，训练 reward 对当前动作的参考作事后评分。

tests/test_racing.py 覆盖原生正/反向过门、门框外、顺序、真实批量步进/梯度、
完成/失败/超时分母和开发选模次序。模型导出修复了重复空默认类和纹理文件缺失；
重建后几何类型/尺寸/位姿/接触属性、纹理像素、命名刚体属性和mocap索引对齐。

## 已运行命令

```bash
env -u PYTHONPATH SCIPY_ARRAY_API=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 \
  .pixi/envs/default/bin/python -m pytest tests -q
```

2026-09-26 07:45 UTC最终功能回归：47通过，214.94秒，退出0，日志
`tmp/p3p4/final-tests-after-numerics.log`。早期37项结果及针对性诊断/修复日志保留在同目录。

两个优化器各有实际求解单元测试：acados 状态0、目标矩阵和控制量验证；采样器实际预测和暖启动。
早期无扰动开发种子20000探针：AttitudeMPC17.12秒完成5门，采样MPC17.02秒完成5门。
随后种子差异测试发现配置读取层级错误（原生为 env.disturbances）。已修复，并验证同种子完全复现、
不同种子真实状态不同，以及含原生扰动的第一性原理 SHAC 真实更新。旧 heldout-v1 中止保留，
正确的正式评测为 heldout-v2；每方法4×32试次并行，严格合并128个不同扰动种子。
早期探针不替代正式结果。CPU采样2000个候选约56.6ms/决策，CUDA约2.4ms，
正式采样器使用CUDA；全部耗时为共享服务器实测，未向仿真注入实测时延。

## 运行与结果位置

- P3：experiments/campaign-p3-seed0/state.json，docs/verification/p3-matrix.json。
- P4学习：experiments/campaign-p4-v1/state.json，docs/verification/p4-learning-v1.json。
- P4优化：experiments/p4-racing-{attitude,sampling}-mpc-heldout-v2/，各32试次分片独立保留。
- 已有 P2 实验和活动查看目录保持原始数据，P3的4个已有单元按来源引用。

两次DSH委派在配额检查处失败且未执行工具；以上实现与测试均由当前主执行会话完成。

## 最终验收补充

- 62个原生回放文件、308条实际保存轨迹已逐帧核验；源文件摘要重查见`p3-p4-replay-acceptance.json`。
- 五种竞速方法的真实VS Code标签页交互见`p4-editor-verification.json`，包括播放、定位、试次切换、指标和全景。
- 随机样条PPO阻力模型的失效积分使观测二阶矩溢出；具名v3恢复原学习参数并显式记录数值失败，
  最终预算及留出128/128完成；原失败v1/v2及首次异常输入保留，见`p3-numerical-diagnosis.md`。
- 当前计时为共享资源实测。峰值显存和统一独占资源计时仍列为P6待测项。
