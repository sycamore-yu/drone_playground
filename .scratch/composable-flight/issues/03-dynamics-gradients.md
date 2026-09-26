# 03：前向动力学与反向规则

Type: task
Status: planned
Blocked by: 01
Engineering: not-started
Experiment: not-started
Quality: not-started
Owner: unassigned

## 模块职责

Crazyflow四模型继续原生复用；加入固定版本LOTF高保真模型与简化导数规则。
保持控制器与物理模块归属，梯度替换发生在完整状态转移接口上。

## 实现证据

相同初态/命令/随机数下逐步比较原生前向；比较简化模型解析导数与适配后的代理导数。
记录状态投影、动作单位、电机/角速度额外状态的切向量处理、重置和延迟。
检查原生组合内部各子步仅执行一次；保留固定经验气动项，关闭在线学习残差。

## 共同交付

与BPTT连接并完成悬停/八字训练。新模型的可微性及策略结果分别报告。
