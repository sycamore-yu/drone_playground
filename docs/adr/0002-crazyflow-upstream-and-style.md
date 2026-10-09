# ADR-0002：Crazyflow、依赖管理与代码规范

**Status:** Accepted · 2026-10-08

## Context

Crazyflow 已提供四旋翼动力学、原生控制、状态管理、JAX 步进和 MuJoCo/MJX 场景支持。项目需要在其基础上实现研究任务、传感、方法及实验。

## Decision

- **Physics**：使用固定版本的官方 Crazyflow；优先直接调用 `Sim`、`SimData`、`Control`、`build_step_fn()`、`build_reset_fn()`、Pipeline 及 MJCF/MJX 功能。
- **Extensions**：在 Simulation 中增加实际缺失的场景、虚拟传感、方法适配与任务逻辑。只有必需的物理能力超出公共 API 时才评估向上游贡献或维护 fork。
- **Environment**：使用 **Python 3.12 + Pixi**，通过 `pyproject.toml` 的 `[tool.pixi]` 和 `pixi.lock` 管理依赖；统一以 `pixi run` 调用项目任务。
- **Style**：遵守 Google Python Style 和 MuJoCo 的命名与简洁代码规范；Ruff 采用 100 列和 Google-style docstrings。

## Consequences

Crazyflow 是默认物理行为的唯一来源。项目持有研究参数与训练实验配置，保持物理、传感、网络和任务参数各自的所有权。ROS1 的系统依赖使用独立运行环境。
