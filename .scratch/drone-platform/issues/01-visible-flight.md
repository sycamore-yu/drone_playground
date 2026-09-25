# 01：第一条可见的完整飞行

Type: task
Status: ready-for-agent
Blocked by: none
Engineering: planned
Experiment: not-started
Quality: not-applicable
Owner: unassigned
Session: none
Run: none

## 要交付

从一个新项目命令运行真实 Crazyflie 八字参考跟踪或作者采样控制完整回合，保存相同运行身份的
轨迹、误差曲线和状态；研究者能在本地 rscope 看到远程结果，浏览器能打开相应 TensorBoard。

## 验收

- [ ] 独立 Pixi 解析并锁定兼容版本，保留历史依赖与新环境的差异；实际 CPU/GPU 设备写入清单。
- [ ] 通过作者任务/控制器完成完整回合，观测、执行状态和事件为真实计算结果。
- [ ] 一个运行身份绑定代码、配置、命令、预算、日志、模型和完整轨迹。
- [ ] rscope 原生读回、模型重建及末状态一致；Windows/SSH 图形确认单独记录。
- [ ] TensorBoard 中至少有时间、跟踪误差、执行动作、记录开销；状态可显示阶段和最近更新。
- [ ] 中断查看器后运行和已有记录可继续访问；启动/恢复/查看命令经过实际执行。

## 证据与交接

实施时填入真实命令、退出码、运行位置、窗口确认、相关提交及下一步。
继承既有 15 项接入证据，验证只补本次新依赖/新记录/新客户端路径。
