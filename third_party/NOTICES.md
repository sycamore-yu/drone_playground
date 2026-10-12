# Third-party notices

Drone Playground is licensed under [GPL-3.0-only](../LICENSE). External components retain their original copyright and license terms. Pinned source revisions and dependency patches are recorded in [sources.json](sources.json) and [patches/](patches/).

| Component | Upstream | License | Integration |
| --- | --- | --- | --- |
| Crazyflow | [learnsyslab/crazyflow](https://github.com/learnsyslab/crazyflow), `36f584d` | MIT | Differentiable quadrotor simulation, controllers, sampling MPC and Crazyflie 2.x visualization meshes |
| Brax | [google/brax](https://github.com/google/brax), `f489566` | Apache-2.0 | PPO and policy training; local terminal-bootstrap patch |
| LSY Drone Racing | [learnsyslab/lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing), `b1f5b36` | MIT | Racing assets, tasks, random trajectories and attitude MPC |
| Learning on the Fly (LOTF) | [uzh-rpg/learning_on_the_fly](https://github.com/uzh-rpg/learning_on_the_fly), `cba6e53` | GPLv3 | Quadrotor dynamics and simplified-dynamics Jacobian |
| D.VA | [HaoxiangYou/D.VA](https://github.com/HaoxiangYou/D.VA), `01b2be4` | MIT | JAX/Brax algorithm adaptation |
| EGO-Planner | [ZJU-FAST-Lab/ego-planner](https://github.com/ZJU-FAST-Lab/ego-planner), `bfda512` | GPLv3 | External ROS1 planner |
| SUPER | [hku-mars/SUPER](https://github.com/hku-mars/SUPER), `2ad3419` | LGPLv3-or-later in the planner source headers | External ROS1 planner |
| MuJoCo-LiDAR | version `0.3.5` | MIT | MID-360 LiDAR simulation |

## Source attribution

Crazyflow's visualization meshes and sampling MPC implementation are credited to the pinned Crazyflow source. Its MIT license is preserved in [licenses/Crazyflow.txt](licenses/Crazyflow.txt).

The LSY-derived code includes the random-reference generator in `src/drone_playground/environments/references.py`, racing task assets in `src/drone_playground/environments/tasks/lsy_upstream/`, and controllers in `src/drone_playground/control/controllers/mpc/lsy_upstream/`. The upstream source copies retain their provenance records and licenses. LSY's MIT license text is reproduced below.

LOTF is installed from the pinned source checkout with declared patches under `third_party/patches/`. Its license is preserved in [licenses/LOTF.txt](licenses/LOTF.txt). The project integrates LOTF's quadrotor equations and simplified-dynamics Jacobian.

The D.VA training algorithm is adapted to JAX/Brax. Its MIT license is preserved in [licenses/DVA.txt](licenses/DVA.txt). The SHAC implementation follows the method in Xu et al., *Accelerated Policy Learning with Parallel Differentiable Simulation* (ICLR 2022), with [NVlabs/DiffRL](https://github.com/NVlabs/DiffRL) as a reference.

The point-cloud learning implementation reconstructs *Learning to Fly from Point Clouds via Differentiable Simulation* from the published paper. The archived integration revision is recorded in [sources.json](sources.json).

EGO-Planner and SUPER are built from their pinned upstream repositories using [docker/ros1/](../docker/ros1/). The local deployment patches are under `docker/ros1/patches/`; the upstream source and its license notices remain in the external build checkout.

acados, HPIPM, BLASFEO and qpOASES are obtained during optional solver setup. Their upstream license files remain in the local source/build directories.

## LSY Drone Racing MIT license

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
