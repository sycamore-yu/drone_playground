# 实验结果入口

本目录分为两个入口：

- [tmp/](tmp/)：按UTC启动日期组织的中间结果，例如`tmp/260930/<run_id>/`。包含开发训练、失败尝试、诊断及完整原始运行；冻结工作树的运行通过链接访问。
- [main_result/v1-18-cells/README.md](main_result/v1-18-cells/README.md)：第一版18格的最终／当前最好结果。每格按日期与训练种子分目录，包含报告、配置、来源、选定权重及回放入口。

截至2026-09-30，共16/18格达到交付要求：14格按现行规则质量通过、EGO两格由用户接受交付。点云两格仅存放开发集最好结果，尚无正式三种子确认。每条结果的`qualification.json`说明当前判定，原报告原样保存在`report.json`；深度修订标准与EGO例外分别注明。

权威第一版选择为[质量进度](../docs/verification/release-progress.json)，本地易读副本为`main_result/v1-18-cells/index.json`。学习方法的每个正式训练种子单独保留。复制权重与原始来源的SHA-256已核对；完整回放链接保留原始数据。项目外独立备份继续保留。

整理或更新选定结果：

```bash
pixi run python scripts/tools/organize_experiments.py \
  --pointcloud-run primary-pointcloud-short32-seed0-t0-20260930 --apply
```

不带`--apply`只查看迁移计划。活跃运行不移动，冻结源码及原始报告不改写；旧的`experiments/<run_id>/...`引用由现役读取接口定位到日期目录。新运行自动写入日期目录，重训必须使用独立`run_id`。

历史30格与第一版18格分别保留。历史权威选择为`docs/verification/final-acceptance/selection.json`，仍可核对：

```bash
python3 scripts/tools/summarize_final_acceptance.py \
  --selection docs/verification/final-acceptance/selection.json \
  --output tmp/final-acceptance-check.json
```

实验产物通过Git忽略规则排除，源码只跟踪此说明。新克隆需要取得结果包或自行训练；不能仅凭目录名把开发结果算作正式通过。

当前进行中：[安全余量1.5m点云实验](tmp/260930/primary-pointcloud-margin15-seed0-t1-20260930/)。保留网络，从零训练；正式验收等待开发逐场景门槛。
