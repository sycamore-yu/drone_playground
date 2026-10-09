# ADR-0005：前向与反向模型范围提案（已撤回）

**Status:** Withdrawn · 2026-10-09

## 撤回原因

此前将本提案标记为 `Accepted` 是记录错误。用户提出的是问题，并未批准“只扩展训练导数、不引入额外前向模型”的范围限制。用户已明确指出，需求涉及用于反向更新的动力学模型，不能以现有梯度衰减代替。

## 当前效力

本文件不作为限制模型接入、选择配置或修改代码的依据。前向模型、反向模型和控制器的具体组合仍在讨论；此处不写入新的替代决定。

已批准的方法组合、随机化和产物布局分别由 ADR-0001、ADR-0006、ADR-0007 记录，不受此次撤回影响。

## 实现事实

截至本次核对，[Environment](../../src/drone_playground/simulation/environment.py) 使用官方 Crazyflow 前向，[Trainer._step](../../src/drone_playground/learning/trainer.py) 调用已有的 [temporal_gradient_decay](../../src/drone_playground/learning/losses.py)。现有时间梯度衰减不构成 LOTF 或 PointMass 反向动力学模型的实现。
