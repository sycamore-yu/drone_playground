# 01：第一条可见的完整飞行

Type: task
Status: done
Blocked by: none
Engineering: passed
Experiment: completed-server-and-client-implementation
Quality: not-applicable
UserAcceptance: accepted-by-user-2026-09-26
Owner: ChatGPT prime
Session: Chat On Steroids current conversation; implementation/p1-p2
Run: p1-native-flight-20260925; p1-ppo-update-20260925

## 要交付

从一个新项目命令运行真实 Crazyflie 八字参考跟踪或作者采样控制完整回合，保存相同运行身份的
轨迹、误差曲线和状态；研究者能在本地 rscope 看到远程结果，浏览器能打开相应 TensorBoard。

## 验收

- [x] 独立 Pixi 解析并锁定兼容版本，保留历史依赖与新环境的差异；实际 CPU/GPU 设备写入清单。
- [x] 通过作者任务/控制器完成完整回合，观测、执行状态和事件为真实计算结果。
- [x] 一个运行身份绑定代码、配置、命令、预算、日志、模型和完整轨迹。
- [x] 最小 train 入口能调用 Brax 完成真实短程更新并产出指标；正式收敛配方在任务 02 交付。
- [x] rscope 原生读回、模型重建及末状态一致，正式启动器连续播放与图表交互已实测。
- [x] 用户于2026-09-26明确确认验收 P1/P2；本轮未代替用户新增 SSH 凭据或认证操作。
- [x] TensorBoard 中有时间、跟踪误差、执行动作、记录开销；状态显示阶段和最近更新。
- [x] 停止和重开查看器后保存的记录可继续访问，查看对象切换与训练成果隔离。

## 证据与交接

原控制器 500 帧/10 秒、RMSE 0.05094382 米；PPO 整链探针实际更新 16384 次交互。
正式 rscope_client.py 窗口 Step229→237，指标曲线可见；暂停保持148后恢复151，左右试次切换通过。
TensorBoard 在 127.0.0.1:6006，HTTP 和真实时间序列读取通过；正式查看器源文件前后保持原始哈希。
记录及截图见 `docs/verification/p1-observation-checks.json/.md`；实际操作见 `docs/runbook.md`。
用户验收已解除阶段阻塞。P1/P2 原有工具端证据保持原样，新增 VS Code 内嵌查看器可直接消费这些记录。
