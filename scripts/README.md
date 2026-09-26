# 运行脚本

日常使用Hydra入口`pixi run train/evaluate/simulate/experiment`。
`evaluate_racing_control.py`是相同组合入口的参数包装；队列脚本也调用当前实验预设。
`summarize_composable_lotf.py`读取两项正式结果、生成曲线并校验回放；`verify_phase_replays.py`可独立读回所选目录。
`rscope_client.py`保留Windows/SSH和原生桌面查看。`setup_acados.sh`只构建项目局部依赖。
历史实验的原始命令记录保存在各自运行内，当前可复用命令见`docs/runbook.md`。
