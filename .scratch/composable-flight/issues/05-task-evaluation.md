# 05：任务、观测、独立评测与结果交付

Type: task
Status: planned
Blocked by: 01, 02, 03, 04
Engineering: not-started
Experiment: not-started
Quality: not-started
Owner: unassigned

## 模块职责

任务与场景独立配置，观测/领域随机化属于同一环境，训练目标提供唯一奖励函数。
接入LOTF状态悬停和轨迹跟踪，读取作者机型、目标、延迟、观察和初态协议。

## 实现证据

新进程重载，开发集选模，留出集参数冻结；记录所有回合、失败和最差轨迹。
rscope保存执行模型、控制器、训练前向与反向模型身份；统一TensorBoard和结果清单。
回归P1–P4的固定状态转换、检查点加载与回放数据；历史实验保持原始字节。

## 共同交付

给出两任务学习曲线、误差/失败表、训练时间、保存策略和独立评测，并区分
论文同协议复评、本项目评测协议和尚未运行的实验。
