# Drone Playground 开发约定

本项目是配置驱动的无人机学习、规划与控制仿真平台。现役分支为 `main`；先读 `docs/status.md`、`docs/backlog.md`，再直接读取相关代码和已有运行记录。

## 边界

- 复用原会话、原运行标识和完整恢复状态。启动前核对进程及工作树，避免重复训练。
- 原点云长训练已按用户要求停止，可恢复状态为45000次更新；冻结源码与检查点保留为只读证据。旧主工作树已退役，具体状态见 `docs/research/branch-lifecycle.md`。
- 公开入口为 `method`＋`env` 与 `train`／`eval`／`play`。任务、传感器、控制器、动力学、网络和更新规则各有明确配置归属，详见 `docs/architecture.md`。
- LOTF、点云论文重建、点云控制迁移和导航适配分别记录来源。每个配方名对应真实实现。
- 几何、传感器、奖励和碰撞共用场景事实。记录实际动作单位、物理时钟、延迟及导数边界。
- 工程执行、完整预算和策略质量分别验收。失败回合保留在正式结果分母中；重训使用独立结果身份。

## 命令与位置

- 固定环境：`python3 scripts/tools/fetch_sources.py`，然后 `pixi install --locked`。
- 训练：`pixi run train method=learning/ppo env=hovering runtime.device=gpu run_id=<新标识>`。
- 测试：`JAX_PLATFORMS=cpu pixi run test`；检查：`pixi run lint`。完整 MPC 测试的 acados 条件见 `docs/runbook.md`。
- 正式训练优先 GPU，按显存和运行预算排队；CPU 用于测试、短探针和原生宿主任务。设备由配置明确选择。
- 临时探针、编译缓存和临时测试放 `tmp/`；正式回归放 `tests/`。运行产物按UTC启动日期放 `experiments/tmp/YYMMDD/<run_id>/`；最终／最好结果按目标放 `experiments/main_result/<目标>/<单元>/<日期-种子>/`，第一版目标为 `v1-18-cells`。
- 读取运行使用 `runs.layout` 的定位函数，兼容历史扁平路径；冻结工作树的原始目录不迁移。选定结果验证权重与报告摘要，开发结果和用户接受的例外显式标注。
- `experiments/`、`.pixi/`、依赖源码和调试日志通过忽略规则排除在源码发布之外。成功权重和回放在清理前独立备份并校验摘要。
- 固定依赖见 `third_party/sources.yaml`；第三方声明与许可文件跟随源码维护。

## 维护与交付

稳定术语放 `CONTEXT.md`，机制放架构文档，操作放运行手册，现役状态只放 `docs/status.md`，未完成研究只放 `docs/backlog.md`。已结束的过程记录归入 Git 历史和项目外备份。

修改前保护已有未提交内容；使用明确文件清单组织本地提交。根据影响范围验证测试、入口、文档链接、参数摘要和回放。提交报告写实际证据及剩余限制，远端推送和发布另行授权。
