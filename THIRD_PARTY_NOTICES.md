# 外部实现来源

Crazyflow 以依赖调用，使用其原生任务函数、控制与动力学。当前提交
`36f584d114d9d331f0cee0fe4b9066f821c0fbfd`，许可为 MIT。

`policies/planning.py` 的随机参考构造及 `tasks/tracking.py` 的初态来自 learnsyslab/lsy_drone_racing
`control/train_rl.py::RandTrajEnv`，提交 `b1f5b36adb8e08e8e2adea85de790bd0e0a1d118`。
复用其 10 个构造点、前三点、平移尺度、三次样条及起飞导数；随机数改为每次运行私有的种子。
当前原始源码的全局随机数与旧 reset 签名通过本项目适配，任务时序和物理含义保留。

MIT License

Copyright (c) 2024 Learning Systems and Robotics Lab (LSY)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## P3/P4 增补

`tasks/lsy_upstream/` 保存同一 LSY 提交的 `race_core.py`、`randomize.py`、`utils.py`、
Level 0 配置和门/障碍资产。`controllers/lsy_upstream/` 保存其 `attitude_mpc.py` 与 Controller。
两个目录均包含原始 LICENSE、来源提交、原文件 SHA256 和最小兼容差异文件。
兼容修改限于 Crazyflow 参数导入和独立 acados 生成目录，优化矩阵、时域和原始推力系数保留。

`controllers/sampling.py` 基于上述 Crazyflow 提交的 `examples/control/sampling.py`
精英均值采样控制算法。保留候选噪声、精英均值更新、暖启动和推力估计器；任务参考和杆状障碍
由 LSY 赛道提供，采样数作为明确运行参数。它的身份是采样 MPC，不标称论文完整 MPPI/iCEM 复现。

`learning/shac.py` 是基于 SHAC 论文目标的独立 JAX 实现，数学依据为
Xu et al., Accelerated Policy Learning with Parallel Differentiable Simulation (ICLR 2022)，
官方算法参考 https://github.com/NVlabs/DiffRL 。复用 Brax 网络、动作分布和归一化，
不复制该仓库的 PyTorch 实现，也不将其作为运行依赖。

acados v0.5.1、HPIPM、BLASFEO、qpOASES 与模板渲染器通过本项目局部构建脚本获取，
各自许可证保留在下载树中；其二进制和 Python 附加依赖位于忽略的 `tmp/`，未纳入本仓库发布。

## Learning on the Fly

原仓库 https://github.com/uzh-rpg/learning_on_the_fly 固定为
`cba6e5370773ace8a08107f02810eecabf16c793`，作为Git子模块保存在
`third_party/learning_on_the_fly`。原始GPLv3许可证、作者、配置、CSV与全部来源文件保留。
该子模块原始文件保持未修改。

`learning/lotf_bptt.py` 的损失、时间展开、随机数和Adam更新逻辑改编自该源码的
`lotf/algos/bptt.py`，属于GPLv3来源的集成代码；`dynamics/gradients.py` 按其自定义JVP
定义实现，并记录现代JAX的PRNG零切向量兼容修改。模型、控制器、MLP、任务和归一化
直接调用原仓库实现。原子模块及其衍生部分的许可证信息不由其它上游的MIT声明覆盖。

本地研究集成沿用当前权限；外部发布/打包不在本轮授权中。
