# Third-Party Notices

This document records third-party source code, algorithm adaptations, runtime dependencies, and external planner integrations used by Drone Playground. Each upstream component retains its original copyright and license terms. A project-level root license does not replace the licenses attached to third-party files or submodules.

Drone Playground is distributed under **GPL-3.0-only** at the project level. The LOTF integration is a key reason for this choice: `tmp/sources/lotf` is GPLv3, and the current dynamics backend imports its quadrotor model directly and retains an explicitly selected simplified-dynamics Jacobian. LOTF does not define a separate task or learning algorithm in this project. Third-party components listed below continue to retain their own copyright and license terms.

| Component | How it is used here | Pinned identity | Upstream license |
|---|---|---|---|
| Crazyflow | Runtime/development dependency for UAV simulation, dynamics and control | `36f584d114d9d331f0cee0fe4b9066f821c0fbfd` | MIT |
| Brax | Native PPO/APG with declared pre-reset terminal-value patch | `f489566dcc44abb7aefa5a1b2d3b64246c54eb9d` | Apache-2.0 |
| LSY Drone Racing | Selected task/controller source preserved with provenance | `b1f5b36adb8e08e8e2adea85de790bd0e0a1d118` | MIT |
| Learning on the Fly (LOTF) | Pinned source-cache dependency plus quadrotor dynamics and analytical gradient integration | `cba6e5370773ace8a08107f02810eecabf16c793` | GPLv3 |
| D.VA | Algorithm semantics adapted into the JAX/Brax stack; upstream source is not copied as a runtime package | `01b2be4986a0851a952aa860afb4a5958e6676e2` | MIT |
| EGO-Planner | External ROS1 native planner process | `bfda51284c8c1b476043255a8145ef925a3778a5` | GPLv3 |
| SUPER | External ROS1 native planner process | `2ad3419c127a617c6d7df6925e81a14175a9c096` | LGPLv3-or-later headers in the planner sources used here |
| MuJoCo-LiDAR | Installed dependency for the MID-360 scan pattern | `0.3.5` | MIT |

The detailed provenance below is retained because several integrations preserve or adapt upstream implementation details rather than merely citing a paper.

## Crazyflow and LSY Drone Racing

Crazyflow is used as a dependency for its native task functions, controllers, and dynamics. The
pinned development identity is `36f584d114d9d331f0cee0fe4b9066f821c0fbfd`, under the MIT License.
The Crazyflie 2.x replay visual meshes under `assets/robots/crazyflie2x/` are copied from that same
pinned commit and retain the MIT license alongside the assets. They are display-only; the
experiment's dynamics and collision contracts remain defined by the selected runtime model.

The random-reference construction in `environments/references.py` and the initial-state logic in
`environments/tasks/tracking/rigid_body.py` originate from `learnsyslab/lsy_drone_racing`, specifically
`control/train_rl.py::RandTrajEnv` at commit `b1f5b36adb8e08e8e2adea85de790bd0e0a1d118`.
The integration preserves its ten construction points, first-three-point use, translation scale,
cubic spline, and takeoff derivatives. Global randomness and the historical reset signature are
adapted to per-run seeded execution while preserving the task timing and physical meaning.

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

## P3/P4 additions

`environments/tasks/lsy_upstream/` preserves `race_core.py`, `randomize.py`, `utils.py`, the Level 0
configuration, and gate/obstacle assets from the same LSY commit. `actions/controllers/mpc/lsy_upstream/`
preserves its `attitude_mpc.py` and Controller implementation. Both directories keep the original
LICENSE, source commit, source-file SHA256 records, and minimal compatibility patches. Compatibility
changes are limited to Crazyflow parameter imports and an isolated acados generation directory;
the optimization matrices, horizon, and original thrust coefficients are preserved.

`actions/controllers/mpc/sampling.py` is based on the elite-mean sampling controller in Crazyflow
`examples/control/sampling.py` at the pinned Crazyflow identity above. It retains candidate noise,
elite-mean updates, warm starts, and the thrust estimator. Task references and pole obstacles come
from the LSY race task, and the sample count is an explicit run parameter. This integration is
identified as sampling MPC and is not presented as a complete reproduction of MPPI or iCEM.

`learning/algorithms/shac.py` is an independent JAX implementation of the SHAC objective described by
Xu et al., *Accelerated Policy Learning with Parallel Differentiable Simulation* (ICLR 2022), with
the official algorithm repository at https://github.com/NVlabs/DiffRL used as a reference. The
implementation reuses Brax networks, action distributions, and normalization; it does not copy the
repository's PyTorch implementation or use it as a runtime dependency.

acados v0.5.1, HPIPM, BLASFEO, qpOASES, and template-generation dependencies are obtained by the
project's local build scripts. Their licenses remain in the downloaded build trees. The binaries
and auxiliary Python dependencies are stored under ignored `tmp/` paths and are not distributed by
this repository.

## Learning on the Fly

The upstream repository https://github.com/uzh-rpg/learning_on_the_fly is pinned at
`cba6e5370773ace8a08107f02810eecabf16c793` and obtained as a verified source-cache dependency at
`tmp/sources/lotf`. Its original GPLv3 license, authorship, configurations, CSV data,
and source files are retained. Declared packaging-only changes are stored in `patches/upstream/lotf-package-metadata.patch`; numerical source implementations retain the pinned identity.

`dynamics/lotf.py` imports the pinned upstream quadrotor equations and exposes only the
high-fidelity and simplified dynamics through Drone Playground's shared Task interfaces.
`dynamics/lotf.py` implements the upstream simplified-dynamics Jacobian rule with compatibility
changes required by modern JAX tangent handling. Drone Playground does not expose LOTF-specific
tasks, methods, evaluation presets, networks, or training algorithms. Permissive licenses used
elsewhere in Drone Playground do not replace the GPLv3 terms that apply to the LOTF source
dependency and derived dynamics/gradient integration code.

## P5 perception and navigation

`learning/algorithms/dva.py` implements the JAX/Brax adaptation against HaoxiangYou/D.VA commit
`01b2be4986a0851a952aa860afb4a5958e6676e2`. It preserves the algorithmic semantics of detached
observations, differentiable action-dynamics-reward propagation, terminal value estimation, and a
target critic while using Drone Playground's sensor encoders. The upstream MIT text is stored in
`docs/licenses/DVA.txt`. The point-cloud path is a project extension and is not presented as
a reproduction of a LiDAR method from the original D.VA paper.

EGO-Planner `bfda51284c8c1b476043255a8145ef925a3778a5` (GPLv3) and SUPER
`2ad3419c127a617c6d7df6925e81a14175a9c096` execute as external ROS1 processes; their planner
source code is not copied into this package. The SUPER repository root does not contain a LICENSE
file at the pinned identity, while the planner source headers used by this project state
LGPLv3-or-later; those original headers remain in the external build tree. Build scripts select
upstream ROS1 templates and run targets, then apply the explicitly documented control-transfer
patches under `ros_integrations/ros1/patches/`. The package stores the process bridge, message adaptation,
and those patches; full planner source trees remain external dependencies.

MuJoCo-LiDAR 0.3.5 is used as a pinned dependency for the MID-360 scan pattern. The active scene
catalog and geometry qualification are under `assets/scenes/navigation/` and `benchmarks/navigation/`.
The shared scene, sensor and collision contracts are described in `docs/architecture.md`.


## 版本 3 来源管理

固定来源由 `patches/sources.json` 声明，`scripts/tools/setup.py` 管理源码缓存；许可副本保存在 docs/licenses。Crazyflow 的 MuJoCo 依赖范围沿用既有本地修正；Crazyflow 与 LOTF 的静态打包元数据补丁解决 Pixi 对本地依赖版本识别的差异。LOTF 补丁还移除源码 full dynamics 中无开关的隐式推力随机化，并暴露外力／力矩输入；随机化由现役 reset 参数与条件配置显式控制。Brax 补丁修正 PPO 的截断奖励与末态 bootstrap；原生模型方程和网络结构沿用固定来源。


## Point-cloud paper reconstruction

`learning/algorithms/recurrent_bptt.py`, `networks/pointnet.py`, `dynamics/point_mass.py` and the corresponding sensor/task modules were integrated from this project's archived research snapshot `dbb660c`. They implement a reconstruction from public information for *Learning to Fly from Point Clouds via Differentiable Simulation*. The reconstruction identity and source pointers are recorded in `patches/sources.json`; inferred settings and the source audit are consolidated in `notes/research/pointcloud.md`. This work is recorded separately from Learning on the Fly.


### 原生控制迁移补丁（2026-09-29）

EGO-Planner与SUPER保持原固定来源和许可证。项目自有补丁位于`ros_integrations/ros1/patches/`：EGO交互目标保留z坐标；SUPER在显式控制迁移开关下改善初始时间猜测。修改范围、构建摘要、原生执行证据和原配方差异见`artifacts/verification/final-acceptance/`与`ros_integrations/ros1/README.md`。点云控制迁移的输入坐标条件化由独立网络预设声明，原论文重建默认保持1.0尺度。
