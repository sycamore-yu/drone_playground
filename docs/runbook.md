# 运行与远程查看

## 现在可用

项目路径：`/home/tong/tongworkspace/simulation_dev/mujoco/drone_playground`。
当前交付为规格、架构、任务目录和继承证据。浏览 `docs/status.md` 和 `.scratch/drone-platform/map.md`。
原 Crazyflow 接入实验仍可按历史报告的原目录命令复验，已有充分证据无需重复运行。

## 首阶段必须交付的入口

下列为计划接口，尚未实现，当前不可作为已可执行命令：

| 入口 | 交付时必须完成的行为 |
|---|---|
| `pixi run train ...` | 保存配置和运行身份，真实训练，写指标、检查点和轨迹 |
| `pixi run evaluate ...` | 新进程加载指定策略，冻结统计，执行固定试次清单并输出逐回合结果 |
| `pixi run replay ...` | 为指定运行准备 rscope 模型与轨迹；沿用原版查看器 |
| `pixi run metrics ...` | 为指定运行启动 TensorBoard，仅监听回环地址 |
| `pixi run status` | 汇报任务与实际运行阶段、最近更新、日志及剩余预算 |

首阶段文档要用实际测试过的命令替换此表；每条命令附实测状态及证据。

## 远程使用契约

服务器负责训练、周期评估、文件保存及 TensorBoard。Windows 本地运行 rscope，使用 SSH/SFTP
获取服务器导出的活动轨迹；图形窗口在本地显示。浏览器通过 SSH 转发查看 TensorBoard 标量。
轨迹和标量共用运行标识，实际可视化客户端无需承担训练计算。

原版 rscope 使用说明提供如下形式，服务器地址和密钥来自用户自己的 SSH 设置：

```powershell
py -m rscope --ssh_to tong@SERVER --ssh_key "$env:USERPROFILE\.ssh\rscope_key" --polling_interval 5
```

这是上游命令形式；当前 Windows 本机执行、路径兼容和密钥连接尚未现场验收。
任务 01 需核对客户端 `/tmp` 路径行为、模型资源、轨迹切换、指标和远程连接；
只能在真实本地窗口出现并显示新轨迹后记录“远程查看通过”。客户端所需授权由用户完成。

TensorBoard 服务就绪后，转发形式为：

```powershell
ssh -N -L 6006:127.0.0.1:6006 tong@SERVER
```

浏览器访问 `http://127.0.0.1:6006`。6006 是计划默认端口，启动前检查占用；
当前本项目尚未启动该服务。

## 每次验收的查看顺序

从当前任务找到运行标识 → 检查配置/协议/预算 → 看开发评估曲线 →
查看固定初态与失败轨迹 → 用新进程加载检查点重评 → 对照完整回合表。
任务单记录实际命令、退出码、结果目录和用户窗口确认。
