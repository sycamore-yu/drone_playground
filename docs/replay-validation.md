# S7：真实 RScope 原生窗口验证

2026-10-08，在本机 Xvfb 上运行已安装的 RScope 0.0.8 原生 MuJoCo viewer。
这是 **headless X11 窗口的实际 OpenGL 绘制**，不是 mock、离屏替代 renderer，
也不是人工桌面点击验收。截图为原生窗口矩形的原始 PNG 像素。

## 验证范围和真实记录

选取以下三个已存在的 episode-0000；没有生成替代飞行或重新计算规划、传感器。
`SUCCESS` 来自各自的 `episodes.csv`，这里只验证播放和绘制，不重新认证控制器。

| 类别 | 实际 `.mj_unroll` 来源 | 记录结果 |
|---|---|---|
| Tracking | `results/apg_tracking_diagnostic_s0/benchmark/empty/rollouts/episode-0000/episode.mj_unroll` | policy，seed 2000000，20 s，SUCCESS；位置 RMSE 0.0337705 m |
| Racing | `results/apg_racing_diagnostic_s0/benchmark/racing/rollouts/episode-0000/episode.mj_unroll` | policy，seed 2000000，22.12 s，SUCCESS；5 次过门 |
| Navigation | `results/super_S01_diagnostic_v3/S01/episode-0000/rollouts/episode-0000/episode.mj_unroll` | 原生 super，S01，seed 100000，34.04 s，SUCCESS |

[原始 CSV 行和 CSV 哈希](../tmp/agents/replay-viewer/source-outcomes.json)保存方法和结果出处。
回放文件 SHA-256：

| 类别 | SHA-256 |
|---|---|
| Tracking | `83ae914df512420d59c947d210de5a502a3412ab7044fd515966ea37f2778069` |
| Racing | `0f466d97684a1bb0247260d4f9f4d95114c5b5b92fd0615e04515073cbb5fd40` |
| Navigation | `9c28f7965afe76c735143c76010a0d1aa03933549edd329c3b502586fb548dd5` |

## 实际窗口截图

Tracking/Racing 连续核对帧 0–30，截图时间为 0.2、0.4、0.6 s。
Navigation 连续核对帧 0–100，截图时间为 0.4、1、2 s。
这证明采样区间内播放时间和位姿持续前进，不表示已经人工观看整个回合。
蓝线是文件中完整的实际记录轨迹；正在移动的蓝色小球是当前无人机。

| 类别 | 三张可检查 PNG | 帧数据和截图 UTC/哈希 |
|---|---|---|
| Tracking | [0.2 s](../tmp/agents/replay-viewer/final-tracking/frame-0010-t0.200.png)、[0.4 s](../tmp/agents/replay-viewer/final-tracking/frame-0020-t0.400.png)、[0.6 s](../tmp/agents/replay-viewer/final-tracking/frame-0030-t0.600.png) | [evidence.json](../tmp/agents/replay-viewer/final-tracking/evidence.json)、[逐帧记录](../tmp/agents/replay-viewer/final-tracking/frames.json) |
| Racing | [0.2 s](../tmp/agents/replay-viewer/final-racing/frame-0010-t0.200.png)、[0.4 s](../tmp/agents/replay-viewer/final-racing/frame-0020-t0.400.png)、[0.6 s](../tmp/agents/replay-viewer/final-racing/frame-0030-t0.600.png) | [evidence.json](../tmp/agents/replay-viewer/final-racing/evidence.json)、[逐帧记录](../tmp/agents/replay-viewer/final-racing/frames.json) |
| Navigation | [0.4 s](../tmp/agents/replay-viewer/final-navigation/frame-0020-t0.400.png)、[1 s](../tmp/agents/replay-viewer/final-navigation/frame-0050-t1.000.png)、[2 s](../tmp/agents/replay-viewer/final-navigation/frame-0100-t2.000.png) | [evidence.json](../tmp/agents/replay-viewer/final-navigation/evidence.json)、[逐帧记录](../tmp/agents/replay-viewer/final-navigation/frames.json) |

![Tracking：帧 30，真实轨迹和无人机](../tmp/agents/replay-viewer/final-tracking/frame-0030-t0.600.png)

![Racing：帧 30，真实门框、纹理、轨迹和无人机](../tmp/agents/replay-viewer/final-racing/frame-0030-t0.600.png)

![Navigation：帧 100，真实障碍、绿色命中、红色规划和无人机](../tmp/agents/replay-viewer/final-navigation/frame-0100-t2.000.png)

Tracking 的真实场景为空，黑色背景符合其原始 MJCF。Racing 可见 LSY 门的纹理和立柱。
这两份 policy 记录的 `sensor_hits`、`plan_points` 全为零；没有人为补上点云或规划。
Tracking 机体旁的红色短杆是朝向标记，Racing 门上的绿色纹理也不是传感命中。

Navigation 的三个截图帧分别含 84/81/69 个有效命中和 67/61/57 个规划点。
绿色点分布于真实墙面/立柱，红色点列是记录中的当前计划；未绘制虚构路线。
像素计数只辅助检查颜色，不能当作点数，因为存在遮挡、投影和亚像素点。

## 方法与相机

[本地验证脚本](../tmp/agents/validate_replay_viewer.py)调用
`cli.replay → rscope.main.main → mujoco.viewer.launch_passive`。
脚本保留真实 native Handle 和原始 `sync()`，在其周围记录证据，设置相机并关闭窗口。
它不替换解码、播放循环、物理状态更新或绘制，不调用 `export_replay`。

每次真实同步都核对记录时间、`qpos`、全部 `mocap_pos/mocap_quat`，以及 MuJoCo
实际计算出的 drone 世界位置和姿态。采样时等待 2 s，让渲染线程完成当前帧后再抓图；
检查 X11 窗口树中存在 MuJoCo 窗口、画面含蓝色飞行几何、各截图哈希不同，
以及回放/metadata 在验证前后的 SHA-256 不变。嵌入资源也逐项核对 metadata 哈希。

相机通过实际 `viewer.cam` API 配置，类型为 `mjCAMERA_FREE`：

| 类别 | lookat | distance | azimuth / elevation |
|---|---|---|---|
| Tracking | 记录轨迹包围盒中心 | 4 m | 125° / −25° |
| Racing | 记录轨迹包围盒中心 | 7 m | 125° / −35° |
| Navigation | 每帧实际 `data.qpos[:3]` | 12 m | 90° / −65° |

`--follow` 是本地脚本每帧更新 `lookat`，不是声称 RScope 有某个跟随快捷键。
Navigation 原世界约 100 m 长；此处聚焦无人机附近，俯角避免立柱遮住机体。
使用原生帮助回调关闭帮助列表；原生帮助列出的空格播放/暂停、Shift+H 帮助、
Shift+M 指标没有被重新实现。截图中保留 RScope 的 Step/Status/Speed。
阴影和反射关闭以降低软件绘制开销；未改变任何几何或轨迹。
界面的 `100%` 是请求播放速度，本机软件渲染未达到实时速度。

## 环境、命令和诊断

Python 3.12.15、RScope 0.0.8、JAX 0.11.2、MuJoCo 3.15.0、Pillow 12.3.0。
`JAX_PLATFORMS=cpu`；`LIBGL_ALWAYS_SOFTWARE=1`、`LP_NUM_THREADS=2`。
Xvfb `:97` 屏幕 1600×1000×24；实际 viewer 窗口 1066×666。
脚本的抓图区域按本次无窗口管理器 Xvfb 的窗口原点 `(0,0)` 取值，未验证一般桌面布局。
验证结束后已关闭本任务启动的 Xvfb；复跑需重新执行下方启动命令。
[glxinfo](../tmp/agents/replay-viewer/glxinfo.txt)确认 Mesa llvmpipe，Accelerated: no。
没有使用训练 GPU、ROS、系统服务、硬件、付费服务或远程上传。

`sudo -n true` 返回需要密码；实际使用 `apt-get download xvfb` 和 `dpkg-deb -x`
将 Ubuntu `2:21.1.12-1ubuntu1.8` 包解压到本地，未安装系统包或修改依赖文件：

```bash
# 从项目根目录；首次准备时下载目录须存在。
mkdir -p tmp/agents/replay-viewer/xvfb
(cd tmp && apt-get download xvfb)
dpkg-deb -x tmp/xvfb_2%3a21.1.12-1ubuntu1.8_amd64.deb tmp/agents/replay-viewer/xvfb
tmp/agents/replay-viewer/xvfb/usr/bin/Xvfb :97 -screen 0 1600x1000x24 -nolisten tcp \
  > tmp/agents/replay-viewer/xvfb.log 2>&1 &
viewer_display_pid=$!
export DISPLAY=:97 JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1 PYTHONUNBUFFERED=1
```

最终截图命令如下。输出目录必须不存在；复跑时请改成新的目录名。

```bash
timeout 120s .pixi/envs/default/bin/python tmp/agents/validate_replay_viewer.py \
  results/apg_tracking_diagnostic_s0/benchmark/empty/rollouts/episode-0000/episode.mj_unroll \
  tmp/agents/replay-viewer/final-tracking --times .2 .4 .6 --distance 4 \
  > tmp/agents/replay-viewer/final-tracking.log 2>&1
timeout 120s .pixi/envs/default/bin/python tmp/agents/validate_replay_viewer.py \
  results/apg_racing_diagnostic_s0/benchmark/racing/rollouts/episode-0000/episode.mj_unroll \
  tmp/agents/replay-viewer/final-racing --times .2 .4 .6 --distance 7 --elevation -35 \
  > tmp/agents/replay-viewer/final-racing.log 2>&1
timeout 240s .pixi/envs/default/bin/python tmp/agents/validate_replay_viewer.py \
  results/super_S01_diagnostic_v3/S01/episode-0000/rollouts/episode-0000/episode.mj_unroll \
  tmp/agents/replay-viewer/final-navigation --times .4 1 2 \
  --distance 12 --azimuth 90 --elevation -65 --follow \
  > tmp/agents/replay-viewer/final-navigation.log 2>&1
pixi run ruff check --no-cache tmp/agents/validate_replay_viewer.py
pixi run ruff format --no-cache --check tmp/agents/validate_replay_viewer.py
kill "$viewer_display_pid"
```

三次最终命令均退出 0，日志最后一行为：

```text
PASS: native window, advancing time/pose, exact recorded state, distinct screenshots, unchanged source
```

| 运行 | 核对帧数 | 第一至最后核对帧的墙钟时间，含中间截图等待 | PNG 捕获 UTC，2026-10-08 |
|---|---:|---:|---|
| [Tracking 日志](../tmp/agents/replay-viewer/final-tracking.log) | 31 | 25.593 s | 16:33:54.442、16:34:03.104、16:34:13.094 |
| [Racing 日志](../tmp/agents/replay-viewer/final-racing.log) | 31 | 23.558 s | 16:35:48.465、16:35:56.801、16:36:04.846 |
| [Navigation 日志](../tmp/agents/replay-viewer/final-navigation.log) | 101 | 51.631 s | 16:38:48.356、16:39:00.573、16:39:33.210 |

Ruff 输出 `All checks passed!` 和 `1 file already formatted`。
[最终证据核查](../tmp/agents/replay-viewer/final-verification.json)再次核对九张 PNG 哈希、
三个源文件/metadata 对、Navigation 非零红绿像素、文档链接和已安装上游代码未变。
三个聚焦运行使用相同 CLI 文件哈希
`9539dbbd582e715f03839826a6d51158003005541dbba5017352c7748dd2d5ba`。

还实际运行了未经本地脚本插桩的入口：

```bash
DISPLAY=:97 JAX_PLATFORMS=cpu LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=2 \
  pixi run replay replay_path=results/apg_tracking_diagnostic_s0/benchmark/empty/rollouts/episode-0000/episode.mj_unroll
```

[原始入口日志](../tmp/agents/replay-viewer/cli-final.log)和
[运行证据](../tmp/agents/replay-viewer/cli-final.json)记录退出 0；使用 X11
`WM_DELETE_WINDOW` 正常关闭。两张实际窗口截图显示
[Step 18](../tmp/agents/replay-viewer/cli-final-before.png) →
[Step 63](../tmp/agents/replay-viewer/cli-final-after.png)。
**这两张默认相机截图只有 UI 可见，未拍到 Tracking 飞行几何**；
只能证明原始命令启动、播放推进和退出，不能冒充场景可见性通过。
本页九张主证据使用前述相机 API，已经明确看到飞行几何。
本次原始入口运行的 CLI 文件哈希为
`9feec2faf20bf99d1e3f949b1b0f21a994e9fc58d7019aa7ca7dbf7d736ab3fc`；
主线并行更新了文件，因此两个运行时点的整体文件哈希不同。

初次无兼容处理的实际窗口停在第 0 帧；
[调用栈](../tmp/agents/replay-viewer/tracking-debug.log)位于上游
`rscope.main` 的 `with viewer.lock()` 内部调用 `Handle.set_texts`，
[40 秒诊断](../tmp/agents/replay-viewer/tracking-renderflags.log)超时。
本地脚本曾将真实文字/图表调用延迟到外层锁之外，证明记录可以连续播放。
另一次 Navigation 诊断在完成截图后以 139 退出，等待 native 渲染线程结束后复验退出 0。
这些诊断文件不作为最终通过证据。

问题已交接主线，主线随后在 `cli.replay` 内处理上游嵌套锁和异步关闭。
最终脚本已移除临时叠加兼容逻辑，使用当前主线的原生启动路径；
本任务未修改 CLI、replay.py、测试、docs/replay.md、pyproject/lock 或已安装的上游文件。
各 `evidence.json` 记录实际验证时的 CLI 和脚本哈希，避免把不同实现的结果混为一谈。

## 补充验证：默认视角与动态场景

随后修复了 CLI 的默认相机：使用 MuJoCo 原生 body tracking 对准 `drone`，
距离 6 m、方位角 90°、俯角 −35°。直接运行 `pixi run replay` 已能看到 Tracking
无人机与完整八字轨迹，正常播放后通过 X11 `WM_DELETE_WINDOW` 关闭，进程退出 0。
这次没有本地脚本插桩或相机覆盖：
[截图](../results/replay_validation/default_camera/tracking.png)、
[稍后截图](../results/replay_validation/default_camera/tracking-later.png)、
[命令与哈希](../results/replay_validation/default_camera/evidence.json)。

动态场景使用已有真实 EGO D01 回合
`results/ego_S6_s0/D01/episode-0000/rollouts/episode-0000/episode.mj_unroll`，
连续核对记录的全部无人机和 mocap 位姿，捕获 0.4 s 与 1.0 s 的原生窗口，退出 0。
画面包含动态障碍、实际轨迹、有效传感命中及当前规划；源记录哈希保持不变。
[0.4 s](../results/replay_validation/production_dynamic/frame-0020-t0.400.png)、
[1.0 s](../results/replay_validation/production_dynamic/frame-0050-t1.000.png)、
[逐帧与截图证据](../results/replay_validation/production_dynamic/evidence.json)。
动态捕获使用前述验证脚本的跟随相机，区别于默认视角的直接入口检查。

## 失败案例的完整播放

另用同一公开 `cli.replay` 播放
`results/ego_S6_acceptance_s0/D03/episode-0002/rollouts/episode-0000/episode.mj_unroll`。
该回合在 24.4000015 s 以 `METHOD_FAILURE / NO_SOLUTION` 结束，原始 CSV 保持不变。
实际窗口从起点播放到失败终点，逐帧核对全部记录，随后正常关闭，进程返回 0。

- [0.4 s 原生窗口](../results/replay_validation/production_failure/frame-0020-t0.400.png)
- [24.4 s 失败终点窗口](../results/replay_validation/production_failure/frame-1220-t24.400.png)
- [来源、帧数、截图和关闭证据](../results/replay_validation/production_failure/evidence.json)
- [逐帧数据](../results/replay_validation/production_failure/frames.json)

失败终点为 `(66.1383, -1.5984, 3.0308)` m，显示 57 个记录命中和 62 个计划点。
红色点列来自记录中此前有效的规划，不代表失败的求解器返回了新解。
本次在 `:98` Xvfb、853×480 原生窗口中使用软件渲染；播放约耗时六分钟。

## 边界

证据覆盖三个任务、动态场景、失败记录和默认视角直接启动。
D03 失败记录核对了完整时长，其余显示案例核对上述采样播放区间。
人工桌面操作和全部 18 项训练验收不在本项显示验证范围中。
软件绘制速度不代表 GPU 或桌面性能。截图和 JSON 位于本地 `tmp/agents/replay-viewer/`，
清理 tmp 前须同时保留这些证据；本任务没有提交 commit。
