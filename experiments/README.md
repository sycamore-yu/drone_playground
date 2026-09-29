# 正式结果包

此目录目前仅保留2026-09-29六方法、五任务正式矩阵所需的29个评测运行、18个训练来源及两组来源权重。每个正式评测继续保留完整终止事件和正式回放；训练目录精简为来源记录、度量、选定参数及最后完整状态。

权威选择为 `docs/verification/final-acceptance/selection.json`，数值与摘要为同目录的 `evidence.json`。成功权重和回放在项目外独立备份；主机路径迁移记录与原始文件字节保存在清理归档中。

```bash
python3 scripts/tools/summarize_final_acceptance.py \
  --selection docs/verification/final-acceptance/selection.json \
  --output tmp/final-acceptance-check.json
```

运行目录通过Git忽略规则排除。新的源码克隆需要取得结果包或自行训练。新实验使用独立 `run_id`，结果表的选择清单仅在明确验收新基线后更新。
