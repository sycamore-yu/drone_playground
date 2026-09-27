# P5-02：D435完整观测链

Type: task
Status: resolved
Blocked by: P5-01
Engineering: passed
Experiment: engineering-run
Quality: separate (P5-07)
Owner: main-session
Session: p5-main
Run: p5-static-depth-ppo-seed0-v1-eng

## 权威设计

阅读 [P5设计与执行计划](../spec.md) 中相关模块及第12节。设计已交付，任务实施待用户执行指令。

## 要交付

完成场景到深度测量、观测历史、动作执行和回放的完整路径，验证原生单环境参照及批量实现。

## 验收

深度单位/光学坐标/视场/遮挡/无效值正确；单批量一致；动态更新、采样时间和环境隔离有证据。

## 执行与恢复

先核对docs/status.md、当前工作树、既有会话与运行进程。恢复原Session/Run；临时验证放tmp/，正式测试放tests/，证据放experiments/和docs/verification/。按工程依赖推进，策略低分独立记录。保存实际命令、退出码、日志、结果及下一步；每个可验收改动本地提交。

## 证据

实现：`tasks/sensors/rays.py`（解析图元批量射线）、`tasks/sensors/depth.py`（D435 相机模型、
内外参、射线栅格、采样时刻与帧历史）、`tasks/observations.py` 的 `NavigationSensorObservation`
（量程裁剪、无效掩码、逆深度归一化、四帧历史）、组合入口的 `observation.sensor` 子组。

**后端探测结论**：本机没有 GL 上下文，原生 MuJoCo 渲染不可用（unset/egl/osmesa 三种方式
的实际报错见来源清单）；MJX 3.14 批量深度渲染需要未安装的 `warp-lang`，而且
`mjx.create_render_context` 对全部 world 共用一个 model，无法表达逐实例几何。
因此按规格第 6 节第 4 步冻结本项目对场景图元的解析批量针孔射线为训练后端。

**验收检查** `tests/test_depth_sensor.py` 共 15 项全部通过。关键一项是：
在 5 组机体位姿 × 300 像素上，解析射线与 `mujoco.mj_ray` 的**命中集合完全一致**
（双向零分歧），最大距离差 3e-6 米。其余覆盖内参（fx=65.25、cx/cy、垂直视场 69.185°）、
光学坐标（+z 前、+x 右、+y 下）、量程与无效像素、遮挡与顺序、运动障碍按实际时刻求值、
逆深度编码、25 Hz 刷新节拍与帧时间戳、重置清空历史。

真实闭环运行 `experiments/p5-static-depth-ppo-seed0-v1-eng`：1048576 交互、
`training/sps=21795.7`、退出码 0；开发集三档难度各 32 回合逐回合记录并导出回放。
该运行同时暴露了一个奖励设计缺陷：原失败代价 -20 小于最大进度收益 75，
撞毁比安全悬停回报更高。已改为失败代价 = 2×最大进度收益（-150）并新增顺序断言
（成功 > 安全超时 > 任何失败）；低分结果按实际保留。
