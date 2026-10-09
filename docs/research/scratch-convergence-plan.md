# 从零训练收敛执行计划

2026-10-09，用户授权持续调参、增加本机并行并测试 AsymmetricPPO，直到 Depth/LiDAR × PPO/APG/SHAC × 种子 0/1/2 全部完成原 C1–C6 验收。

保留 Actor CNN/GRU192、PointNet/GRU192 与小尺度首层初始化；所有新候选从随机权重开始。已通过的运行不重训。冻结评测不用于挑选超参数。所有失败、主动停止及中断产物保留。

- [x] 核对真实运行：6/18 accepted；两个 PPO 对照运行中。
- [x] 补启 LiDAR APG/SHAC 种子 1、Depth SHAC 种子 1，暂为五路并行。
- [ ] 使用 optim-agent ask/tell 保存真实候选、命令及 checkpoint 成绩。
- [x] 按 DiffAero 的 Actor/Critic 输入分离实现特权 Critic：真实位姿、速度、目标位移、时钟、前一动作、真实全向几何距离；Actor 及冻结推理只消费原观测。
- [x] 先写失败回归，再实现窄入口；五项定向回归通过，检查真实 PPO 更新、完整恢复和推理隔离。
- [ ] PPO 根据 checkpoint 证据调参；其他算法补齐所有种子。保留连续三次完整主场景检查与独立冻结。
- [ ] 更新选定清单、验收报告、文档及发布验证，只有真实全部通过时声明完成。

优化目标：最大化最新完整 checkpoint 检查的六个主场景总成功率，分母固定 150；另记录最差场景成功率和 C5 连续通过次数。种子 0 为候选筛选，选定同一配方再复现 1/2；独立冻结成绩不回流优化。没有训练更新数或墙钟上限，停止条件为全部正式运行通过。主动淘汰无进展候选使用 pruned 并报告最新真实分数。

优化产物在 `.optim-agent-runs/`；安装工具的独立虚拟环境在 `tmp/optim-agent-venv/`。本机并行受实测 RAM/显存约束，避免重复此前内存压力造成的主机中断。不新增训练后端。

实现触及 `simulation/environment.py`（真实状态观测）、`simulation/networks.py`（独立 Critic 输入）、`learning/trainer.py`（配置、采样入口和旧 checkpoint 默认值）、对应回归测试与训练文档。PPO 的现有 GAE、终止前 bootstrap、循环策略重算及 Actor 导出流程复用。
