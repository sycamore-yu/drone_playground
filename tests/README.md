# Tests

测试结构遵循 Isaac Lab、MuJoCo Playground 和 Crazyflow 的共同做法：按被测职责组织，同时明确区分单组件测试、跨组件集成测试和冻结数值回归。目录结构不表达项目阶段、版本号或一次性验收过程。

```text
tests/
├── unit/          # 单个源码领域或稳定公共接口
├── integration/   # 配置装配、训练、评测、原生算法、CLI 等跨模块链路
├── regression/    # 冻结数值结果与历史行为基线
├── helpers/       # 无测试用例的共享构造器和稳定仓库路径
├── fixtures/      # 只读测试数据
└── conftest.py    # 全局 pytest/JAX/SciPy 测试环境
```

`unit/` 尽量镜像 `src/drone_playground` 的领域边界，例如 `control`、`dynamics`、`environments`、`learning`、`runtime` 和 `visualization`。Unit 不调用完整 Hydra `compose_experiment` / `build_environment`，也不承担真实 TensorBoard、Git、回放或跨模块训练链。`integration/` 用于这些跨组件和外部 I/O 边界；不要为了目录完整性复制 unit coverage。`regression/` 只保存需要精确比较的冻结数值或历史行为基线。

测试只依赖生产代码公共接口或 `tests.helpers`；测试文件之间不得互相 import。退役实现不保留兼容测试，现役替代接口应承接仍有价值的行为断言。临时诊断放 `tmp/`，真实运行产物放 `results/`。

完整验证使用：

```bash
JAX_PLATFORMS=cpu pixi run test
pixi run lint
```

需要快速定位时可直接选择目录，例如 `pytest tests/unit`、`pytest tests/integration/configuration` 或 `pytest tests/integration/environments`。测试通过只证明工程契约成立，不代表训练策略已经收敛。
