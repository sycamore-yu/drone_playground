# 分支与工作目录

2026-09-29：现役 `main` 从 `architecture-v3` 工作树迁入 `simulation_dev/drone_playground`，Python 环境按锁文件在新路径重新安装。

| 分支 | 当前用途 | 保留依据 |
|---|---|---|
| `main` | 唯一新增开发入口 | 现役代码、文档和正式产物 |
| `refactor/composable-architecture-v3` | 已退役 | 提交已合入 `main`；由 `archive/architecture-v3-20260929` 标签保留历史 |
| `refactor/native-planner-runtime` | 停止开发，冻结运行依赖保留 | 旧主工作树的 Python 解释器仍被原点云训练和协调器使用；Git 公共管理目录与现役 MPC 所引用的 acados 构建也位于该工作树 |
| `research/pointcloud-paper-navigation8` | 停止新增开发，仅承载原冻结运行 | 代码已合入 `main`；`archive/pointcloud-paper-20260929` 保留源码身份 |

原点云任务为 `paper-pointcloud-seed0-full-v1`，协调器为 `paper-pointcloud-seed0-pipeline-v1`。其余训练已结束。当前结果及进程身份以对应 `state.json`、`result.json` 和进程启动标记核对。

原点云完整训练和评测结束后，先备份参数、源摘要和回放，再处置冻结工作树及其解释器。此步骤需要同时迁移 Git 公共管理目录和 acados 构建，保留现役 `main` 的完整历史与依赖引用。运行期间保持这些依赖原位。
