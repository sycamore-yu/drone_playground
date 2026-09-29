# 项目主表与评测任务分类

本表沿用 FlightBench 表Ⅲ的“方法—感知—规划—控制接口”组织方式，增加传感器／输入与任务类型。任务依据所列论文的实验设计归类；指标取舍与扩展范围是本项目的设计选择。现役实现及冻结结果分别见[状态](status.md)和[正式验收](verification/final-acceptance/README.md)。

## 任务族

项目采用四类核心闭环任务、两类专项闭环任务和一类独立组件评测。静态与动态合并到导航任务族，运行时仍保留各自的具名环境与结果。[1–4]

| 层次 | 任务类型 | 输入与完成目标 | 评测重点（项目拟统一） | 论文依据与安排 |
|---|---|---|---|---|
| 核心闭环 | 悬停／定点稳定 | 给定目标状态，从偏离目标的初态稳定到目标附近 | 稳态误差、稳定时间、扰动恢复、失败率 | VisFly-Lab；AC-MPC的稳定化与价值分析实验；作为基础控制诊断 [2,8] |
| 核心闭环 | 轨迹跟踪 | 给定带时标的参考轨迹，跟随位置及速度参考 | 位置／速度误差、控制平滑性、约束违反、计算时间 | VisFly-Lab；作为学习控制与MPC的直接对照 [2] |
| 核心闭环 | 竞速 | 给定门的位置和通过次序，尽快完成合法过门 | 完成率、圈速／完成时间、过门进度、控制限幅与鲁棒性 | VisFly-Lab、AC-MPC；给普通跟踪控制器配固定参考生成器，并记录其计时规则 [2,8] |
| 核心闭环 | 导航（静态／动态） | 给定起点、目标或目标序列，在障碍环境中无碰撞到达 | 到达率、碰撞率、旅行时间、路径长度、平滑性、重规划耗时、约束违反 | FlightBench、MIGHTY、SANDO、SUPER、NavRL、两篇可微飞行；规划类的主要任务 [1,3–7,11,12] |
| 专项闭环 | 着陆 | 接近着陆区，并满足接触位置、下降速度和姿态等终止要求 | 安全着陆率、落点误差、接触速度、完成时间 | VisFly-Lab有明确着陆任务；单列扩展，需定义接触与终止协议 [2] |
| 专项闭环 | 集群协同导航 | 多架受控无人机同时完成各自目标，处理相互避让 | 全体成功率、机间碰撞、完成时间、最小间距、通信条件 | EGO-Planner-v2、深度可微飞行的位置交换和共同穿越；独立多机协议 [7,13] |
| 组件评测 | 走廊约束轨迹生成／时间分配 | 固定起终状态、走廊及动力学边界，生成可行且高效的轨迹 | 求解可行率、求解时间、飞行时长、控制代价、约束违反 | AllocNet的数值基准、MIGHTY的表示比较、SANDO的标准化静态优化基准；与闭环成功率分开报告 [3,4,9] |

森林、迷宫、窄缝、薄障碍、多航点、遮挡程度、障碍密度和速度档位属于导航内部的场景或难度配置。风扰、动作延迟、传感噪声和动力学失配属于适用任务上的鲁棒性条件。该分类保留少量明确任务族，并通过具名配置形成论文常用的具体实验。[1,6–8,11,12]

## 项目主表

表中“＊”表示本项目拟采用的任务迁移或扩展评测；其余论文行的任务具有本次所引版本的实验依据。通用算法行采用VisFly-Lab的四项控制任务作为文献锚点，感知导航另作配方扩展。所有条目表示方法归类及评测范围；实际可运行性和质量验收继续独立登记。

“特权信息”指推理时额外获得环境真值或未来障碍运动；“按协议”要求明确数据来源。状态估计、目标参考、真实通信获得的队友信息，以及由感知构造的安全走廊分别按观测合同登记。通用算法的传感器和运行时域由所选网络、观测与执行配方确定。

轨迹／航点／运动命令三列记录该方法最终向下游交付的层级，“内部”仅表示方法内部存在相应中间量；“—”表示该层没有独立的对外输出。轨迹之后的参考采样与跟踪控制器由组合配置明确选择。全局／局部／滚动描述运行时决策组织；训练展开长度另外记录。[1]

| 方法 | 方法类型 | 传感器／输入 | 特权信息 | 决策时域 | 建图／环境表征 | 规划／策略模块 | 轨迹 | 航点 | 运动命令 | 任务类型／评测 |
|---|---|---|---|---|---|---|---|---|---|---|
| PPO [2] | RL | 状态；深度／LiDAR＊ | 依观测配置 | 局部策略 | 依配方 | 策略网络 | — | — | ✓ | 悬停、跟踪、竞速、着陆；导航＊ |
| BPTT [2] | diffRL | 状态；深度／LiDAR＊ | 依观测配置 | 局部策略 | 依配方 | 可微训练策略 | — | — | ✓ | 悬停、跟踪、竞速、着陆；导航＊ |
| SHAC [2] | diffRL | 状态；深度／LiDAR＊ | 依观测配置 | 局部策略 | 依配方 | 策略／价值网络 | — | — | ✓ | 悬停、跟踪、竞速、着陆；导航＊ |
| ABPT [2] | diffRL | 状态；感知输入＊ | 依观测配置 | 局部策略 | 依配方 | 策略、价值辅助与初态重采样 | — | — | ✓ | 悬停、跟踪、竞速、着陆；导航＊ |
| NavRL [6] | RL | RGB-D、状态、障碍估计 | 感知版✗ | 局部 | 占据图射线、动态障碍状态 | PPO＋速度安全修正 | — | — | 速度 | 导航（静态／动态） |
| 深度可微飞行 [7] | diffRL | 深度图、机体状态 | ✗ | 局部策略 | — | 卷积编码＋GRU | — | — | 推力加速度 | 导航（静态／动态）；集群协同 |
| 点云可微飞行 [11] | diffRL | LiDAR点云／深度转点云、状态 | ✗ | 局部策略 | — | PointNet式编码＋GRU | — | — | 加速度 | 导航（静态／动态） |
| Fast-Planner [1] | 轨迹优化 | 深度、里程计 | ✗ | 全局引导／重规划 | 栅格＋ESDF | 搜索＋样条优化 | ✓ | — | — | 导航（静态；动态＊） |
| EGO-Planner [1] | 轨迹优化 | 深度、里程计 | ✗ | 局部 | 占据栅格 | 搜索＋B样条优化 | ✓ | — | — | 导航（静态；动态＊） |
| EGO-Planner-v2 [13] | 协同轨迹优化 | 深度、状态、队友通信 | 按通信／感知协议 | 局部 | 占据栅格、队友轨迹 | MINCO协同规划 | ✓ | — | — | 集群协同；单机静态导航；普通动态障碍＊ |
| SUPER [5] | 轨迹优化 | LiDAR、状态 | ✗ | 滚动 | 局部占据地图 | 探索轨迹＋安全备用轨迹 | ✓ | — | — | 导航（静态；动态＊） |
| SANDO [4] | 硬约束轨迹优化 | LiDAR／D435、障碍运动信息 | 按真值／感知协议 | 滚动 | 占据图＋时空安全走廊 | 三次Bézier＋MIQP | ✓ | — | — | 导航（静态／动态）；轨迹生成组件 |
| MIGHTY [3] | 软约束轨迹优化 | LiDAR、状态、障碍运动信息 | 按协议 | 滚动 | 安全走廊、障碍模型 | Hermite样条时空优化 | ✓ | — | — | 导航（静态／动态）；轨迹生成组件 |
| AllocNet [9] | 学习＋可微优化 | 起终状态、安全走廊；前端可用深度 | 按走廊来源 | 有限走廊／局部重规划 | 安全走廊 | 时间分配网络＋QP | ✓ | — | — | 走廊轨迹生成／时间分配；静态导航 |
| AttitudeMPC [14] | MPC控制 | 状态、参考 | 按任务协议 | 滚动 | — | 跟踪优化 | — | — | 姿态／推力 | 跟踪、竞速（项目控制基线） |
| SamplingMPC [14] | 采样式MPC控制 | 状态、参考 | 按任务协议 | 滚动 | — | 精英采样优化 | — | — | 姿态／推力 | 跟踪、竞速（项目控制基线） |
| AC-MPC [8] | RL＋可微MPC | 状态、目标门 | 依任务观测 | 滚动 | — | 代价网络＋MPC | 内部 | — | 总推力／角速度 | 竞速；悬停消融；跟踪＊ |
| AERO-MPPI [12] | 采样式规划—控制 | LiDAR、状态 | ✗ | 滚动 | 局部点云距离表征 | 锚点引导＋并行MPPI | 内部 | 内部 | 总推力／角速度 | 导航（静态；动态＊） |
| LOONG [10] | 模仿学习＋MPCC | LiDAR、状态 | 按感知协议 | 滚动 | 局部地图、安全走廊 | 时间分配网络＋避障MPCC | 内部 | 内部 | 执行控制量 | 导航（静态、时间关键；动态＊） |

“深度可微飞行”对应《Learning vision-based agile flight via differentiable physics》；“点云可微飞行”对应《Learning to Fly from Point Clouds via Differentiable Simulation》。通用diffRL＋LiDAR导航可以选择点云论文的完整配方，也可以独立比较PPO、BPTT和SHAC配方。PointNet、GRU、动力学和损失等模块配置共同确定论文身份。[7,11]

NavRL原论文的障碍感知使用D435i；LiDAR惯性里程计提供状态估计。其静态输入是地图上的虚拟射线，动态输入包括估计的障碍位置、尺寸和速度；完整方法还包含速度安全修正。LiDAR感知移植单独登记为项目扩展。[6]

SANDO采用三次Bézier与MIQP硬约束规划，MIGHTY采用Hermite样条时空优化。SANDO的动态实验分别提供真实未来障碍运动和纯感知估计两种条件；MIGHTY动态实验的障碍位置函数同样需要明确来源。“安全走廊”属于结构化表征，特权标签按其信息来源确定。[3,4]

LOONG、AERO-MPPI以完整导航系统参加导航组。抽取其中的MPCC或MPPI组件研究跟踪、竞速时，建立独立控制配方。AllocNet以轨迹生成和时间分配作为组件主评测；接入固定感知前端与控制链后，采用静态导航验证完整闭环。[9,10,12]

## 导航评测的组织依据

MIGHTY第IV-D节以静态场景对比SUPER和EGO系列，第IV-E节测试纯动态及静动态混合障碍。SANDO第IX-A/C节测试标准化静态问题与森林，第IX-D/F节分别测试动态真值和感知闭环。因而，静态／动态导航可以作为这些轨迹规划方法的主要系统评测；轨迹表示和优化器自身的贡献再用固定走廊问题补充隔离。[3,4]

SUPER的文献锚点是未知障碍环境中的安全高速导航，其探索／备用机制应在未知空间、窄缝、细障碍和速度变化条件下考核。动态障碍统一测试属于本项目的扩展协议。FlightBench的森林、迷宫、多航点，以及点云论文的静态、移动、细障碍实验，都可以作为导航任务内的具名子场景。[1,5,11]

本项目导航组拟统一三类信息条件：传感器观测驱动的闭环；给定当前局部几何的规划组件；给定真实未来障碍运动的受控实验。分别报告成功与失败分母、飞行时间／路径质量、规划与控制时间以及约束违反。密度、可通行宽度、遮挡、转弯、动态障碍速度和预测误差形成测试配置；具体阈值在相应评测协议中冻结。

## 框架与底座

| 名称 | 项目角色 | 对应评测范围 |
|---|---|---|
| FlightBench [1] | 导航基准和分层接口参照 | 静态导航的森林、迷宫、多航点等场景；任务难度与系统性能指标 |
| VisFly-Lab [2] | 多任务一阶学习参照，提供ABPT设计 | 悬停、跟踪、着陆、竞速；训练步数、墙钟时间及多种子表现 |
| Crazyflow [15] | JAX可微动力学与批量执行基础 | 提供任务执行基础；另做动力学、导数和吞吐量验证，按本项目底座协议记录 |

实现顺序采用悬停、跟踪、竞速、静态／动态导航的现役核心，随后按明确研究需求扩展着陆与集群协同。走廊轨迹生成作为优化组件专项接入。任务分类调整只改变设计与文档归属，已有具名环境、冻结协议和历史结果继续使用各自身份。

## 文献与依据

以下固定版本及章节支持上表的机制和原始任务。带“＊”项及“项目控制基线”行属于本项目选择，完成状态以独立验收为准。

1. [FlightBench: Benchmarking Learning-based Methods for Ego-vision-based Quadrotors Navigation](https://arxiv.org/html/2406.05687v3)，第III节、表Ⅱ／Ⅲ、图4：导航场景、接口层级、信息条件及六类性能指标；与用户提供的2025年版本一致。
2. [VisFly-Lab: Unified Differentiable Framework for First-Order Reinforcement Learning of Quadrotor Control](https://arxiv.org/html/2603.21123v1)，第III-C节、表Ⅰ及第V节：悬停、跟踪、着陆、竞速与PPO/BPTT/SHAC/ABPT比较。着陆以接触和下降速度区分终止结果。
3. [MIGHTY: Hermite Spline-based Efficient Trajectory Planning](https://arxiv.org/html/2511.10822v1)，第IV-A/B节：轨迹表示组件比较；第IV-D/E节：静态导航与两类动态障碍；第V节：LiDAR硬件实验。
4. [SANDO: Safe Autonomous Trajectory Planning for Dynamic Unknown Environments](https://arxiv.org/html/2604.07599v1)，轨迹优化节：MIQP与三次Bézier；第IX-A/C/D/F节：标准化静态、森林、动态真值和纯感知动态；第X节：硬件验证。
5. [SUPER作者仓库及论文入口](https://github.com/hku-mars/SUPER)，论文《Safety-assured high-speed navigation for MAVs》，Science Robotics 2025；ROG-MAP与探索／安全备用轨迹。其静态系统对照也见MIGHTY第IV-D节和LOONG第IV-A节。
6. [NavRL: Learning Safe Flight in Dynamic Environments](https://arxiv.org/html/2409.15634v2)，第III节：表征、PPO与速度安全修正；第IV节：D435i感知、静态／动态导航和课程训练。
7. [Learning vision-based agile flight via differentiable physics](https://doi.org/10.1038/s42256-025-01048-0)，用户提供版本的Results及Methods：深度感知、加速度策略、单机静态／动态导航及集群位置交换／共同穿越。
8. [Actor-Critic Model Predictive Control: Differentiable Optimization meets Reinforcement Learning for Agile Flight](https://arxiv.org/html/2306.09852v8)，第IV节及第V-A/D节：多赛道竞速和动力学扰动；第V-E/F节：悬停训练消融及代价／价值关系分析。
9. [Deep Learning for Optimization of Trajectories for Quadrotors](https://arxiv.org/html/2309.15191v2)，第III节：AllocNet时间分配与可微QP；第IV-A/B节：数值优化基准；第IV-C节：室内外静态导航闭环。
10. [LOONG: Online Time-Optimal Autonomous Flight for MAVs in Cluttered Environments](https://arxiv.org/html/2601.07434v1)，第II/III节：学习时间分配和含避障MPCC；第IV-A/B节：时间关键、密集森林及真实障碍环境导航。
11. 《Learning to Fly from Point Clouds via Differentiable Simulation》，用户提供的IROS 2026论文，第III-C/E节：PointNet式编码、GRU和可微物理训练；第IV-B节与图6/7：静态、移动和细障碍导航。来源及重建假设见[点云方法说明](research/pointcloud.md)。
12. [AERO-MPPI: Anchor-Guided Ensemble Trajectory Optimization for Agile Mapless Drone Navigation](https://arxiv.org/html/2509.17340v2)，第III/IV节：CTBR、局部LiDAR与多锚点MPPI；实验节：不同密度及障碍结构的导航。动态环境列为项目扩展。
13. [EGO-Planner-v2作者仓库](https://github.com/ZJU-FAST-Lab/EGO-Planner-v2)，对应《Swarm of micro flying robots in the wild》。单机静态测试亦见MIGHTY第IV-D节；普通动态障碍的额外预测输入按所选适配登记。
14. 项目已有[AttitudeMPC配方](../configs/method/optimization/attitude_mpc.yaml)与[SamplingMPC配方](../configs/method/optimization/sampling_mpc.yaml)。本行评测范围是项目控制基线设计；SamplingMPC的精英采样身份单独保留。
15. [Crazyflow作者仓库](https://github.com/learnsyslab/crazyflow)与项目[固定依赖清单](../third_party/sources.yaml)：JAX无人机仿真、可微动力学与批量执行接口。
