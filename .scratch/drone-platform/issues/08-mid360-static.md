# 08：可观察的 MID-360 静态导航任务

本条目保留早期范围记录。当前执行由 [P5设计](../../p5-navigation/spec.md) 的 P5-01、P5-03 承接，任务状态以新任务单为准。

Type: task
Status: ready-for-agent
Blocked by: 01
Engineering: planned
Experiment: not-started
Quality: not-applicable
Owner: unassigned
Session: none
Run: none

## 要交付

复用现成射线/扫描模式，在静态导航任务中运行真实动作并查看测量、障碍、轨迹和任务事件。

## 验收

- [ ] 核对 MuJoCo-LiDAR 与当前 JAX/MJX 的版本和批量行为，优先复用已存在实现。
- [ ] 记录 MID-360 的扫描方向、频率、外参、范围和输入编码；虚拟载荷约定明确。
- [ ] 用已知几何检查射线距离、无回波、视场、刷新频率和坐标变换，测量来自实际场景。
- [ ] 静态导航明确具体上游场景和成功/碰撞/超时协议，冻结后再运行正式方法。
- [ ] 完整一次导航执行的点云摘要、动作、轨迹与事件写入同一运行，rscope 显示策略实际可见输入。
- [ ] 记录感知开销及训练预算所需吞吐；保留改变射线预算时的输入差异。

## 证据与交接

填入依赖核对、解析几何对照、冻结任务来源、完整记录和测量时序。
