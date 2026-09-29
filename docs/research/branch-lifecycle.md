# 分支与工作目录

2026-09-29：唯一开发位置为 `simulation_dev/drone_playground`。现役 `main` 已拥有自己的 `.git` 公共管理目录、锁定 Python 环境和本地 acados 构建；旧 `mujoco/drone_playground` 路径已退役，不再存在。

| 分支 | 当前用途 | 保留依据 |
|---|---|---|
| `main` | 唯一新增开发入口 | 现役代码、文档和正式产物 |
| `refactor/composable-architecture-v3` | 已退役 | 提交已合入 `main`；由 `archive/architecture-v3-20260929` 标签保留历史 |
| `refactor/native-planner-runtime` | 停止开发，原目录离线归档 | 提交已是 `main` 的祖先；分支仍保留，旧未提交补丁已备份 |
| `research/pointcloud-paper-navigation8` | 冻结源码与证据保留，无训练进程 | 代码已合入 `main`；`archive/pointcloud-paper-20260929` 保留源码身份；工作树关联已迁到现役 `.git` |

原点云任务为 `paper-pointcloud-seed0-full-v1`，协调器为 `paper-pointcloud-seed0-pipeline-v1`。用户选择在备份45000次完整状态后立即结束；两者均已停止。最后日志为49760次更新，恢复点仍是45000次，原50000次预算未完成，最终评测未启动。状态文件已记录停止原因，修改前副本另行保留。

完整检查点包含参数、优化器和随机状态，已核对摘要并独立备份；旧路径移除后，使用现役解释器和冻结源码读回成功。后续训练按新的质量目标建立运行身份，不自动续跑原任务。

## 清理审计结论

两个目录原来是同一仓库的工作树。旧分支提交 `1b93534` 已是现役 `main` 的祖先；旧目录另有 `docs/status.md` 与 `experiments/README.md` 两处未提交修改，已单独备份，并保留在离线目录中。

本轮已执行的退役顺序：

1. 备份45000次完整状态并核验参数、优化器和随机状态；结束训练及协调器，保留原预算与未保存进度的区别。
2. 保存全部分支、标签、Git历史、公共管理目录和各工作树未提交内容；在独立临时位置演练迁移，比较全部引用、主工作树状态及差异。
3. 将公共 Git 管理目录迁到现役位置，修复冻结工作树关联；验证 `git fsck`、引用完整性和工作树状态。现役 `.git` 已从指向旧位置的文件变为真实目录。
4. 在现役位置从固定版本独立重建 acados，替换旧符号链接；检查共享库、Python 接口及 MPC 数值回归。
5. 核查运行进程、可编辑安装和依赖引用，将旧目录整体移入项目外归档，并将其中的旧 `.git` 改为离线管理备份，避免继续作为仓库使用。

归档位于 `simulation_dev/.archives/drone_playground/cleanup-design-20260929T123143Z/`，包含 Git bundle、清理前源码、未提交补丁、检查点备份、退役凭据及 `retired-primary-worktree/`。冻结点云工作树继续保留在 `simulation_dev/.worktrees/drone_playground/pointcloud-paper`，仅作证据存放。历史正式参数和回放仍由现役 `experiments/` 及原独立备份保存。

## 首版确认工作树

正式证据与主目录开发隔离，以下detached工作树只执行各自冻结提交，不作为新的开发仓库：

| 工作树尾名 | 冻结提交 | 作用 |
|---|---|---|
| `release-control-20260929` | `ac6b861` | PPO／SHAC／BPTT三种子确认 |
| `release-navigation-20260929` | `9071156` | 深度导航独立初态扰动确认 |
| `release-solvers-20260929` | `f38fc00` | MPC与SUPER正式确认 |

均位于`simulation_dev/.worktrees/drone_playground/`，运行产物在各自`experiments/`；主目录`experiments/release-*-confirmation-20260929`的链接指向对应批次清单与进度。共享的是同一平台固定依赖环境及acados库，各方法仍有独立日志、临时生成代码、求解器实例和ROS端口。待确认阶段结束并独立备份后再退役这些工作树。
