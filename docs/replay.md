# 实际记录的 RScope 回放

`simulation.replay.export_replay` 把已记录的飞行状态写成
[RScope 0.0.8](https://github.com/Andrew-Luo1/rscope) 的 `rscope.rollout.Rollout`。
它复制实际 MJCF、include、纹理和 mesh，添加无人机、实际轨迹、传感命中和规划采样点。
导出不调用物理步进，不改变 Scene、传感器或输入数组。

```python
from drone_playground.simulation.replay import export_replay
from drone_playground.simulation.scene import Scene

path = export_replay(
    "results/run-001/rollouts/episode-0001",
    Scene("D01"),  # empty、racing、Navigation8；也可直接传 MJCF Path
    times,  # (T,)，实际物理时间，单位 s，严格递增，至少两帧
    positions,  # (T, 3)，世界坐标，单位 m
    quaternions_xyzw,  # (T, 4)，body-to-world 单位四元数
    measurements=world_hits,  # 长度 T，每帧 (N, 3) 世界坐标命中点
    plans=planned_positions,  # 长度 T，每帧 (M, 3) 世界坐标规划采样点
)
```

`directory` 必须不存在或为空；每个回合独立目录，已有记录不会覆盖。
返回 `episode.mj_unroll` 的绝对路径。同目录包含 `rscope_meta.pkl`、`scene.xml`
和重命名后的资源文件。Metadata 也嵌入全部资源，因此整个目录可以搬到另一台机器。

## 测量与规划输入

`measurements` 和 `plans` 都逐帧对齐 `times`，每帧点数可变。
`None` 或空数组表示本帧无显示点，不会自动沿用上一帧。
测量还可以使用 `{"points_world": points, "mask": boolean_mask}`；
规划可以使用 `{"positions": sampled_positions}`。
无效测量先按 mask 去除；有效点、位姿和时间必须有限。

Sensor 的 `Measurement.points_body` 不是世界坐标，调用方必须用采集时的机体位姿转换。
对于 `measure(..., pose_at=None)`，其扫描期间机体位姿保持为传入的 pose：

```python
import numpy as np
from scipy.spatial.transform import Rotation

world_hits = []
for measurement, position, quaternion in zip(measurements, positions, quaternions_xyzw):
    points = np.asarray(measurement.points_body).reshape(-1, 3)
    world_hits.append({
        "points_world": Rotation.from_quat(quaternion).apply(points) + position,
        "mask": np.asarray(measurement.mask).reshape(-1),
    })
```

如果扫描使用了 `pose_at`，须用 `Measurement.times` 对应的逐点采集位姿做同样转换；
不能把移动扫描的所有点都套用最后一帧位姿。显示已有测量，不重新射线采样。
调用方决定测量何时可用、规划何时生效和失效；将实际可见内容放到对应回放帧。
规划应传入求解器实际输出的轨迹采样，导出器不生成路线或补做规划。

## 显示内容与时间

| 内容 | 保存和显示方式 |
|---|---|
| 无人机 | `qpos[T,1,7]`：世界位置和 MuJoCo `wxyz` 四元数；0.07 m 球和机体 +x 朝向标记 |
| 实际飞行轨迹 | 蓝色 MJCF 胶囊线段，连接已记录位置；这是离散采样间的显示连线 |
| 传感命中 | 绿色球形点云，mocap 按帧移动；仅包含有效命中 |
| Planner 轨迹 | 红色球形采样点，mocap 按帧切换；显示当前传入的计划 |
| 动态障碍 | 在 `time[T,1]` 的原始时间上求值 Navigation8 `body.user` 运动规律 |
| 标量 | `sensor_hits`、`plan_points`，可在 viewer 中绘图 |

保留 LSY 门的位置、姿态、纹理和真实碰撞/可视几何。Navigation8 的静态障碍和边界
保持原始几何；trefoil 与 linear bounce 使用 [资产说明](../assets/scenes/README.md)
中的公式。显式 MJCF 路径遵循同一运动约定；场景应无关节，名称 `drone*`、`replay_*`
为显示几何保留。空的 `Scene("empty")` 可直接导出。

`qvel` 是由相邻记录通过 `mujoco.mj_differentiatePos` 得到的显示用前向差分，
最后一帧沿用前一帧；不是独立采集的物理速度。API 未提供 reward，RScope 的必需
reward 字段填零并在 metadata 中标明，不能作为评测得分使用。

RScope 没有逐帧可见性字段；每种点云预留最大帧点数的 mocap 槽，未使用的槽移到
世界坐标 `(0,0,-1e6)`。导出不截断或降采样；大量点和长回合会增大模型、记录和渲染成本。
需要减小显示规模时，由调用方明确采样，并把采样规则保存在运行配置中。
无人机和叠加几何都不参与碰撞。

## 打开 viewer

在有桌面显示的机器上，CLI 会直接启动上游原生 RScope viewer：

```bash
pixi run replay replay_path=results/run-001/rollouts/episode-0001/episode.mj_unroll
```

`replay_path` 必须指向实际 `.mj_unroll`，同目录必须存在 `rscope_meta.pkl`。
CLI 在临时 `BASE_PATH` 中只链接所选文件，`META_PATH` 指向它的 metadata，
`TEMP_PATH` 使用独立临时缓存，然后调用 `rscope.main.main(ssh_enabled=False)`。
退出后清理临时目录并恢复进程内配置；不调用会清空全局活动目录的 `rscope_init`。
Linux 会话没有桌面显示时明确报错；可在桌面机器或下述 VS Code viewer 打开。

锁定的 RScope 0.0.8 与 MuJoCo 3.15 组合需要 CLI 中的兼容处理：上游 overlay 调用
已经持有内部锁，CLI 避免其外层重复加锁，并在退出时等待 viewer 线程完成 GLFW 清理。
实际原生窗口已验证 Tracking、Racing、Navigation 和动态场景的时间推进、记录位姿一致性
及正常退出；截图和验证记录见 [显示验证](replay-validation.md)。

安装了本地 **RScope Viewer** VS Code 扩展时，双击 `episode.mj_unroll`，或运行命令面板中的
`RScope` 打开命令。把 `rscope_meta.pkl` 留在同目录；远程工作区的解释器可以设为：

```json
{"rscope.pythonPath": "${workspaceFolder}/.pixi/envs/default/bin/python"}
```

原生 RScope 默认 `python -m rscope` 读取 `/tmp/rscope/active_run`，没有目录参数，
也不包含本项目对锁定版本的兼容处理。程序化播放使用相同的项目入口：

```bash
pixi run python - results/run-001/rollouts/episode-0001/episode.mj_unroll <<'PY'
import sys
from drone_playground.cli import replay

replay(sys.argv[1])
PY
```

空格暂停/播放，`Shift+M` 查看点数指标，`Shift+H` 查看帮助。
Mac 的原生 MuJoCo viewer 使用 `mjpython` 替换上面的 `python`。
RScope 0.0.8 用前两帧时间差控制固定播放间隔；不等间隔数据的位姿和原始时间仍正确保存，
但原生 viewer 的播放速度不反映每个不同的间隔。VS Code viewer 按实际时间轴播放。

## 可追溯性与验证

Metadata 的 `replay` 字段记录场景名称、Scene 的 `geometry_hash`（显式路径时为空）、
导出资源 SHA-256、帧数、四元数顺序、速度和 reward 的来源说明。
S8 所需案例、种子、方法身份、延迟、事件、失败分母和耗时由运行器保存在本回合所属的
`config.yaml`、`run.json`、`episodes.csv` 中，并记录返回的回放路径；导出器不会虚构这些数据。

```bash
JAX_PLATFORMS=cpu pixi run pytest -q tests/test_replay.py
JAX_PLATFORMS=cpu pixi run pytest -q tests/test_cli.py -k replay
pixi run ruff check src/drone_playground/simulation/replay.py tests/test_replay.py
```

测试通过上游 `rscope.rollout.append_unroll` 和
`rscope.model_loader.load_model_and_data` 解码真实文件；检查三帧姿态、原始时间、
Navigation8 几何、两种运动规律、LSY 纹理与门姿态、MuJoCo 射线命中和规划点的显示几何。
这些是格式和显示数据的测试；S7/S8 的完整验收还需实际闭环回合和运行器记录。
CLI 测试还通过真实 `rscope.main.main` 解码所选记录，仅替换桌面窗口调用；
另验证单文件选择、临时路径清理和无显示会话的明确报错，不宣称完成桌面绘制验证。
