# 02：可加载的 PPO 跟踪策略

Type: task
Status: resolved
Blocked by: 01
Engineering: passed
Experiment: completed
Quality: passed-development32-and-heldout128
Owner: ChatGPT prime
Session: Chat On Steroids current conversation
Run: p2-figure8-ppo-seed0-v2

## 要交付

使用 Brax 原生 PPO 训练八字跟踪，展示从初始到最终的同初态轨迹与曲线，交付独立进程可加载的成功策略。

## 验收

- [x] 采用作者任务和可追溯训练配方，正式训练前冻结预算、开发试次、质量门槛与选模规则。
- [x] 核验动作缩放、观测归一化、每回合随机化、终止/时间截断与重置前状态。
- [x] 真实训练达到完整预算，保存中间/最终检查点、回报、误差、KL、动作饱和及实际吞吐。
- [x] 新进程加载指定检查点，冻结参数与归一化统计，输出全部开发试次结果。
- [x] 至少一个 PPO 检查点满足已确认质量要求，完整失败列表和代表轨迹随模型交付。
- [x] 实现可用、预算完成和策略达标分别记录；质量调试期间其他工程就绪任务继续。

## 证据与交接

真实训练完成 2097152 次交互；开发集选中 `step-0001310720.pkl`。
独立 CPU 进程开发集 32/32、RMSE 0.0275328675 米；留出集 128/128、RMSE 0.0268792909 米。
保存前后检查点、策略参数及归一化内容校验值一致；完整试次和重放位于运行内独立评测目录。
命令、文件校验和读回证据在 `docs/verification/p2-independent-evaluation.json`，实际操作见运行手册。
