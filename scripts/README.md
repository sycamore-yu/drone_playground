# 运行脚本

公开入口为 `train.py`、`eval.py`、`play.py`，分别对应 `pixi run train/eval/play`，共同调用包内装配与运行逻辑。完整方法通过 `method=...` 选择，完整环境通过 `env=...` 选择。

辅助工具位于 `tools/`。`fetch_sources.py` 取得固定依赖；`migrate_artifact.py` 显式复制迁入可信旧检查点；`run_pointcloud_pipeline.py` 编排具名点云训练阶段；`summarize_pointcloud.py` 核对实际预算和产物。各汇总与回放检查工具保留原始来源身份。

`tools/rscope_client.py` 保留 Windows/SSH 查看；`tools/setup_acados.sh` 构建项目局部依赖。历史原始命令保存在各运行目录，现役入口见 [操作手册](../docs/runbook.md)。
