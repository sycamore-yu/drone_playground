# 运行记录模块

`record.py` 与 `console.py` 保存运行标识、代码、解析配置、依赖、日志和结果；`checkpoints.py` 管理策略产物身份；`pointcloud.py` 保存具名训练和恢复证据。每次运行独占其输出目录。

`layout.py` 统一按UTC启动日期写入`experiments/tmp/YYMMDD/<run_id>/`，跨日期保持运行标识唯一。状态、评测、检查点与回放读取兼容历史扁平路径，原始报告中的路径不改写。`scripts/tools/organize_experiments.py`整理已结束运行，链接冻结来源，并将明确选定的报告、配置和权重校验后放入`experiments/main_result/<目标>/`；当前目标为`v1-18-cells`。

`migration.py` 为可信旧检查点提供显式复制迁移，校验摘要并重定位序列化类型，源文件保持独立。现役执行只消费版本 3。数值续训状态与开发集选模历史分别保存。

RScope 导出、场景包与查看位于 `visualization/`，记录侧保持数据来源身份。工程完成、实际配置预算完成及任务质量使用独立字段。完整记录规则见[开发约定](../../../docs/development.md)。
