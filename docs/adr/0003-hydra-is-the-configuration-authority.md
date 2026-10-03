# ADR 0003：Hydra 是唯一配置与组合权威

## Status

Accepted.

## Context

项目已经使用 Hydra/OmegaConf、YAML config groups 和 `_target_`。如果再增加 Python registry 来保存环境、动力学、传感器、控制器或默认参数，就会形成第二份名称和默认值来源。

MuJoCo Playground 的 registry 同时维护 constructor、default config 和 randomizer，因为它的默认配置主要由 Python `ConfigDict` 提供。该模式不适合已经以 Hydra YAML 为配置源的本项目。

## Decision

- Hydra/OmegaConf 是唯一配置组合系统。
- YAML config groups 是 preset 和默认参数的唯一来源。
- `_target_` 是内部组件 constructor 身份，组件由 Hydra 直接实例化。
- 不建立 Environment、Dynamics、Sensor、Controller 或 Algorithm 的重复 registry。
- 不在 registry 中保存质量、学习率、任务时长或其他默认参数。
- public `load()` 只做 Hydra Compose API 的薄封装；public name 直接使用 Hydra config group 名，例如 `navigation/static`。
- 当前不维护 `navigation-static -> navigation/static` 这类重复 alias。

只有出现稳定 public alias 或第三方 runtime registration 的真实需求后，才增加对应机制。

## Consequences

- CLI、Python API、checkpoint 和实验配方使用同一套名称与配置语义。
- `configs/env/navigation/static.yaml` 本身就是 `navigation/static` 的唯一 preset 定义。
- `load()` 不提供第二套 Python 参数 API；自定义组合继续使用 Hydra override。
- 新增环境或组件只需要新增 YAML preset 和实现类，不需要同步第二张 Python 映射表。
- 配置迁移时不能把构造逻辑重新塞回 `composition.py` 或 factory switch。
