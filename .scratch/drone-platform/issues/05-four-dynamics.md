# 05：四种动力学的受控比较

Type: task
Status: resolved
Blocked by: 02,03,04
Engineering: passed
Experiment: completed
Quality: 21-of-24-passed
Timing: shared-server-recorded-exclusive-profiling-pending-P6
Owner: ChatGPT prime
Session: Chat On Steroids current conversation
Run: docs/verification/p3-p4-results.json

## 要交付

从同一配置入口切换四种动力学，以共同支持的姿态接口完成已选择的学习方法实验，并重建一张可复算结果表。

## 验收

- [x] 每个单元明确训练/评测模型、机型、控制接口、频率、预算、种子和任务协议。
- [x] 原始检查点和同模型评测可重载；沿用既有导数证据，只补改变部分。
- [ ] 记录净训练时间、编译开销、实际交互、显存和全部试次结果，排除混合设备/配置身份。
- [x] 已接受实验组合完成约定预算；质量低、程序错误和不适用组合分别标识。
- [x] 保存模型内比较的表、来源运行和代表轨迹；后续跨模型重评复用相同检查点。

## 依赖说明

依赖 02/03/04 的工程接口交付；单个算法的质量结果不构成全局开工条件。

## 证据与交接

24格首轮预算及同模型独立评测已完成，复用4组P2。最终表见`docs/verification/p3-p4-results.md`，
重建命令`pixi run python scripts/summarize_p3_p4.py`。含阻力随机PPO v1/v2的数值失败保留，
v3恢复原学习参数并显式处理无法表示有限二阶矩的失效状态，留出128/128。

计时项部分完成：保存真实作业耗时、Brax原生计时、SHAC编译/更新分项、设备和全部试次。
逐运行峰值显存及统一独占资源计时仍待P6；上述复现实验完成状态不代表已完成严格性能排行。
