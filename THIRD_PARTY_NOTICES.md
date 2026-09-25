# 外部实现来源

Crazyflow 以依赖调用，使用其原生任务函数、控制与动力学。当前提交
`36f584d114d9d331f0cee0fe4b9066f821c0fbfd`，许可为 MIT。

`tasks/tracking.py` 的随机参考构造及初态来自 learnsyslab/lsy_drone_racing
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
