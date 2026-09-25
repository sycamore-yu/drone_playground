# P1 观察链实测

检查时间更新至 2026-09-25 16:29 UTC。范围仅覆盖 P1 观察链，不修改训练、任务或记录实现。

原版 `rscope 0.0.8` 已在服务器 `DISPLAY=:10.0` 打开真实 MuJoCo 窗口。进程 PID 为 `2372784`，窗口 `0x4c0000b` 的 `_NET_WM_PID` 与该进程一致，标题为 `MuJoCo : MuJoCo Model`，尺寸 `1365×703`。单独窗口截图见 `tmp/p1p2/rscope-window.png`；画面能看到四旋翼、网格地面，以及 `Eval 1/1`、`Env 1/1`、`Step 0`、`Status Play`、`Speed 100.0%`。截图只取该 MuJoCo 窗口，没有抓取整个远程桌面。

当前发布目录 `/tmp/rscope/active_run` 的来源是 `experiments/p1-native-flight-20260925/rollouts/native-flight`，包含原生 `.mj_unroll`、`rscope_meta.pkl`、`scene.xml`、模型资源和发布标记。该 rollout 内实际有 `action/0..3`、`reference_x/y/z`、`tracking_error` 八个指标。已确认 `rscope 0.0.8` 的外层 UI 锁会让窗口停在 `Step 0`：仅删除 `main.py` 中包住 UI 更新的 `with viewer.lock():` 并将其内容整体缩进恢复后，同一轨迹真实推进。`tmp/p1p2/rscope-dedented-step-a.png` 与 `tmp/p1p2/rscope-dedented-step-b.png` 相隔约 `0.6 s`，`Step` 从 `362` 变为 `371`，机体位置同步变化。随后 `Shift+M` 成功显示八个实时指标图；`tmp/p1p2/rscope-dedented-metrics.png` 与 `tmp/p1p2/rscope-dedented-metrics-later.png` 显示 `reference_z`、`tracking_error`、`action/0..3`、`reference_x/y` 持续更新。原运行时备份保存在 `tmp/p1p2/rscope-main.py.before-ui-lock-dedent`。该验证针对手工固定运行时，正式 `rscope_client.py` 集成后仍需再次复验。

上述 dedent 仅用于定位 UI 问题，曾直接修改当前项目 Pixi 环境中的安装版 `rscope/main.py`。该共享运行时在并发训练展示收尾中产生了干扰，因此验证完成后已从 `tmp/p1p2/rscope-main.py.before-ui-lock-dedent` 恢复原始字节。临时修改文件的 SHA-256 为 `1872044f8dd7b6feed99729a85a0820fbfad471a74d4e9c75214092c1932ccc2`；恢复后的安装文件与备份 SHA-256 均为 `f10f8c03208c9fd077e6c0afbe8fa75c9d13fb77ecd5a38c8d8cfb5ed76135c5`，逐字节比较一致，`with viewer.lock():` 已恢复在第 131 行。后续不再修改安装依赖或共享 `/tmp/rscope/active_run`；正式启动器只在独立副本和独立目录复验。

2026-09-25 16:29 UTC 已完成正式 `scripts/rscope_client.py` 的真实 GUI 复验。为避免影响任何共享发布状态，先把 `experiments/p2-figure8-ppo-seed0-v2/independent-dev/rollouts` 原样复制到 `tmp/p1p2/formal-client-record`，再运行 `env -u PYTHONPATH JAX_PLATFORMS=cpu DISPLAY=:10.0 pixi run python scripts/rscope_client.py --directory tmp/p1p2/formal-client-record --show-metrics`。启动日志报告本地缓存正是该独立目录，兼容层只在内存 AST 中执行 `ui_locks_removed=1`、`state_write_locks_added=1`。窗口 `0x4c0000b` 的 `_NET_WM_PID=2428689` 与正式启动器进程一致。

连续播放通过：`tmp/p1p2/formal-step-a.png` 与 `formal-step-b.png` 相隔约 `0.7 s`，`Step 229 -> 237`，机体和所有可见指标同步变化。指标图由 `--show-metrics` 默认打开，可见 `physical_thrust`、`squared_error`、`tracking_error`、`action/0..3`、`action_saturation`、`failure` 等曲线。空格暂停/恢复通过：`formal-pause-a.png` 与 `formal-pause-b.png` 相隔约 `0.8 s` 均保持 `Step 148` 且状态为 `Pause`；再次空格后 `formal-resume.png` 到 `Step 151` 且状态回到 `Play`。左右案例切换通过：右键把 `Env 1/5` 切到 `Env 2/5`，左键又恢复到 `Env 1/5`，证据分别为 `formal-env-right.png` 与 `formal-env-left.png`。

正式 GUI 验证前后，安装版 `.pixi/envs/default/lib/python3.13/site-packages/rscope/main.py` 的 SHA-256 都是 `f10f8c03208c9fd077e6c0afbe8fa75c9d13fb77ecd5a38c8d8cfb5ed76135c5`，第 131 行原始 `with viewer.lock():` 仍存在，说明正式启动器没有修改安装源码。

另执行严格 SSH 预检：`pixi run python scripts/rscope_client.py --ssh_to tong@127.0.0.1 --known_hosts tmp/p1p2/known_hosts.localhost --cache_dir tmp/p1p2/formal-ssh-cache`。在没有可用认证凭据时约 4 秒退出，返回码 1，错误为 `No authentication methods available`；本地缓存目录没有创建，后台 watcher 和 GUI 均未启动。该检查只证明严格主机校验后的认证失败能快速终止，Windows 原生客户端仍保留为待用户侧验收项。

TensorBoard 已完成真实 HTTP 与数据接口验证。PID `2367896` 只监听 `127.0.0.1:6006`；主页返回 `200`，标量标签接口 `/data/plugin/scalars/tags` 返回 `200`。当前能看到三个运行：`p1-native-flight-20260925/metrics`、`p1-ppo-update-20260925/metrics`、`p2-figure8-ppo-seed0-v1/metrics`。P1 飞行的 `flight/time_s`、`flight/tracking_error_m`、`flight/altitude_m` 各有 500 个真实时间点，时间从 `0.02 s` 到 `10.0 s`；P1 PPO 的 `training/sps=707.1865`、`training/total_loss=1.26523`、`training/walltime=23.1679 s` 均能从标量数据接口读取，当前对应 `step=16384`。原始 HTTP 响应均保存在 `tmp/p1p2/`。

安装版 `rscope` 的 SSH 接口已直接核对源码。命令形式为 `python -m rscope --ssh_to username@host[:port] --ssh_key PATH --polling_interval SECONDS`。底层使用 Paramiko：watcher 通过 SFTP 轮询 `/tmp/rscope/active_run` 中的 `.mj_unroll`，transfer 先下载到临时文件后原子重命名，模型加载器另行下载 `rscope_meta.pkl`。`ssh_connect` 对 `FLAGS.ssh_key` 无条件执行 `os.path.expanduser`，所以 SSH 模式实际需要给出 `--ssh_key`。

本机 SSH/SFTP 实测使用 `/etc/ssh/ssh_host_ed25519_key.pub` 生成仅供本次使用的 `tmp/p1p2/known_hosts.localhost`，没有读取或输出私钥。严格主机校验通过后，SSH 明确进入认证阶段并返回 `Permission denied (publickey,password)`，返回码 `255`；SFTP 返回 `Connection closed`，返回码同样为 `255`。这证明服务器 SSH 传输与主机身份链可达；当前批处理环境缺少可用认证凭据，因此 SFTP 文件拉取还没有现场通过。

上游 SSH 模式启动时会清理客户端本地 `BASE_PATH`。默认 `BASE_PATH` 为 `/tmp/rscope/active_run`；在服务器本机直接启动 SSH 客户端会与服务器发布目录重合并删除其内容。本轮按约束没有这样启动。Windows 本机运行时客户端与服务器文件系统分离，仍需现场验证路径行为、用户私钥认证、真实窗口轨迹更新和指标切换。

机器可读完整证据见 `docs/verification/p1-observation-checks.json`。P1 当前可以把“服务器 rscope 原生窗口真实推进”“指标图实时更新”“TensorBoard 页面与标量数据可读”“SSH 服务与主机身份可验证”标为已验证；“Windows 原生 rscope 经 SFTP 拉取并显示新轨迹”继续保留为用户侧验收。
