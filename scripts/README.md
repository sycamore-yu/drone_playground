# 执行入口

项目 CLI 已提供 train/evaluate/demo/replay/metrics/status 入口，命令见 `docs/runbook.md`。
`rscope_client.py` 是只依赖 rscope/MuJoCo 的独立查看启动器，可复制到 Windows，
修正固定版本的 UI 锁和远程路径分离问题；使用原版查看器，不修改安装包文件。
