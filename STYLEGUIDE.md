# 代码与技术文档规范

采用 [MuJoCo Style Guide](https://github.com/google-deepmind/mujoco/blob/main/STYLEGUIDE.md) 和 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)，优先保持与 [Crazyflow](https://github.com/learnsyslab/crazyflow) 的公开 API 一致。

## Python

- 目标版本：**Python 3.12**。
- 统一使用 **Ruff**：100 列、Google-style docstrings，以及项目配置中声明的 lint 规则。
- API 使用清晰名称、单位和类型提示。按实际模块职责组织代码，小接口封装完整行为。
- 代码测试通过公开接口和真实物理执行验证；涉及 JAX 的测试覆盖 shape、JIT、批量和必要梯度。
- 运行命令由 Pixi 任务统一提供。

## MJCF 与 C++

- 几何、机体和场景在 MJCF 中采用 MuJoCo 正式元素；物理参数来源唯一。
- C/C++ 遵循 MuJoCo 风格及对应外部求解器/库的原生约定；C++ 格式化使用仓库明确选择的 clang-format。
- Protobuf 接口由实际 C++／ROS 服务定义，使用标准 `protoc` 生成代码。

## 文档

- **GLOSSARY**：领域术语。
- **ADR**：已批准的架构取舍及原因。
- **Spec**：任务、方法、输入输出、实验约束与可检验的完成条件。
- **Research**：第一方资料与方法参数参考。

技术文档直接陈述能力、范围和验收。每个必要限制只在相关方法或实验章节说明一次。可参考 [Anti-Defensive Writing](https://github.com/Kiterlin/anti-defensive-writing) 和 [Stop That Shit / STSS](https://github.com/lennney/stop-that-shit) 的精简表达原则。
