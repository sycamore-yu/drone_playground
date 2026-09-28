# Archify 架构图

本目录保存独立可交互页面与对应图源。页面在浏览器中直接打开，可缩放、搜索节点、追踪关系、
切换主题和选择导览路径。GitHub 的文件预览显示源码；下载 HTML 后由浏览器执行。

| 页面 | 内容 | 图源 |
|---|---|---|
| [双通道执行架构](composable-runtime.html) | 学习式与优化式两列，在兼容命令处进入共享飞控和动力学 | [结构化图源](composable-runtime.architecture.json) |
| [训练与导数契约](composable-learning.html) | 四类训练配置、参数更新、观测与状态转移导数 | [结构化图源](composable-learning.architecture.json) |

在项目目录中打开：

```bash
xdg-open docs/diagrams/composable-runtime.html
xdg-open docs/diagrams/composable-learning.html
```

使用本机已安装的 Archify 重新校验与生成，逐张执行：

```bash
ARCHIFY="$HOME/.agents/skills/archify/bin/archify.mjs"
node "$ARCHIFY" validate architecture docs/diagrams/composable-runtime.architecture.json --quality showcase --json
node "$ARCHIFY" deliver architecture docs/diagrams/composable-runtime.architecture.json docs/diagrams/composable-runtime.html --quality showcase --json
node "$ARCHIFY" validate architecture docs/diagrams/composable-learning.architecture.json --quality showcase --json
node "$ARCHIFY" deliver architecture docs/diagrams/composable-learning.architecture.json docs/diagrams/composable-learning.html --quality showcase --json
```

图源是编辑入口，生成器负责页面实现。执行图呈现当前 P5 具体组合；说明卡明确标记建议扩展。
正式槽位、状态所有权和实现范围见[架构文档](../architecture.md)，来源与候选见
[槽位设计依据](../research/composable-slot-design.md)。

校验与浏览器检查结果保存在[本轮证据](../verification/composable-architecture-20260928.json)。
