# 执行入口

项目 CLI 已提供 train/evaluate/demo/replay/metrics/status 入口，命令见 `docs/runbook.md`。
`rscope_client.py` 是只依赖 rscope/MuJoCo/Paramiko 的独立查看启动器，可复制到 Windows，
支持 OpenSSH Host 别名，修正固定版本的 UI 锁和远程路径分离问题，并为 tracking rollout
默认绘制完整红色 3D 参考轨迹；使用原版查看器，不修改安装包文件。
