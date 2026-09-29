# 项目目录

现役项目为 `simulation_dev/drone_playground`，主分支为 `main`。下面给出维护中的完整一级结构和主要二级职责；依赖缓存、生成结果及项目外备份分别标注。

```text
drone_playground/
├── AGENTS.md                         # 开发边界与入口
├── CONTEXT.md                        # 稳定术语
├── README.md                         # 项目介绍、安装和正式结果表
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
├── pixi.lock
├── .gitignore
├── .gitattributes
├── assets/
│   └── scenes/
│       ├── navigation/              # 权威场景目录
│       └── archive/                 # 场景生成和回归测试所需的历史夹具
├── benchmarks/
│   └── navigation/
│       ├── v1/                      # 保留的历史协议和几何校验
│       └── v2/                      # 正式300秒、20米/秒协议
├── configs/
│   ├── config.yaml
│   ├── method/                      # learning、paper、optimization配方
│   ├── env/                         # 完整环境
│   ├── scene/
│   ├── task/
│   ├── sensor/
│   ├── observation/
│   ├── execution/
│   ├── controller/
│   ├── dynamics/
│   ├── optimal_control/
│   ├── network/
│   ├── objective/
│   ├── algorithm/
│   ├── training/
│   ├── runtime/
│   ├── evaluation/
│   └── visualization/
├── src/drone_playground/
│   ├── __init__.py
│   ├── app.py                       # Hydra入口
│   ├── cli.py                       # 查看与维护入口
│   ├── composition.py               # 装配与校验
│   ├── contracts.py
│   ├── methods/                     # 决策方法
│   ├── networks/                    # 感知、策略、价值网络
│   ├── environments/                # 场景、任务、传感和观测
│   ├── execution/                   # 控制、延迟和转移
│   ├── models/                      # 实际与预测模型实现
│   ├── learning/                    # 训练器和目标函数
│   ├── runtime/                     # JAX与宿主闭环
│   ├── evaluation/                  # 协议与统计
│   ├── integrations/                # 原生规划器适配
│   ├── runs/                        # 身份、记录、检查点和迁移
│   └── visualization/               # RScope轨迹与查看
├── native_planners/
│   ├── README.md
│   ├── versions.env
│   ├── setup.sh
│   ├── bridge/
│   ├── docker/
│   └── patches/
├── scripts/
│   ├── README.md
│   ├── train.py
│   ├── eval.py
│   ├── play.py
│   └── tools/                       # 依赖、迁移、场景、回放和正式汇总
├── tests/                           # 正式回归及历史配置夹具
├── third_party/                     # 来源、许可和固定补丁
├── docs/
│   ├── README.md
│   ├── architecture.md
│   ├── methods.md                    # 论文方法主表、接口与评测任务族
│   ├── runbook.md
│   ├── evaluation.md
│   ├── development.md
│   ├── project-tree.md
│   ├── status.md
│   ├── backlog.md
│   ├── research/
│   │   ├── pointcloud.md
│   │   ├── references.md
│   │   └── branch-lifecycle.md
│   └── verification/
│       ├── final-acceptance/         # 正式选择、摘要和完整命令
│       ├── cpu-gpu-timing.json
│       └── release-cleanup.json
├── experiments/                     # Git忽略：所选正式产物及必要权重
│   └── README.md                    # 结果包管理与验证说明
├── tmp/                             # Git忽略：临时验证和必要源码缓存
│   ├── README.md
│   ├── sources/                     # 固定Crazyflow与LOTF源码
│   └── p3p4/optimization/           # acados构建依赖
└── .pixi/                           # Git忽略：本项目Python环境
```

独立备份位于项目外 `simulation_dev/.archives/drone_playground/`。旧主工作树和 `pointcloud-paper` 仅保留原冻结训练的依赖，处置规则见[分支生命周期](research/branch-lifecycle.md)。新源码分发包仅包含版本管理中的公开文件。
