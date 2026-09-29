# 版本 3 完整实际目录

目录依据 `refactor/composable-architecture-v3` 的受版本控制文件及本轮交付文件生成。基础重组提交为 `48cb031`。包含当前配置、源码、测试、文档与验证摘要；运行产物、依赖环境和缓存按下方约定保存。未来 LOONG、AERO-MPPI、AC-MPC 的计划登记在来源清单和待办中，新增文件随真实实现交付。

## 仓库维护文件

```text
drone_playground/
├── .scratch/
│   ├── architecture-v3/
│   │   ├── implementation.md
│   │   ├── progress.md
│   │   └── spec.md
│   ├── composable-flight/
│   │   ├── issues/
│   │   │   ├── 01-composition.md
│   │   │   ├── 02-policy-control.md
│   │   │   ├── 03-dynamics-gradients.md
│   │   │   ├── 04-training.md
│   │   │   └── 05-task-evaluation.md
│   │   ├── implementation.md
│   │   ├── map.md
│   │   └── spec.md
│   ├── drone-platform/
│   │   ├── issues/
│   │   │   ├── 01-visible-flight.md
│   │   │   ├── 02-ppo-tracking.md
│   │   │   ├── 03-apg-random-tracking.md
│   │   │   ├── 04-shac-tracking.md
│   │   │   ├── 05-four-dynamics.md
│   │   │   ├── 06-native-racing-mpc.md
│   │   │   ├── 07-racing-policy.md
│   │   │   ├── 08-mid360-static.md
│   │   │   ├── 09-static-navigation-policy.md
│   │   │   ├── 10-dynamic-navigation.md
│   │   │   └── 11-heldout-release.md
│   │   ├── map.md
│   │   ├── p1-p2-execution.md
│   │   └── spec.md
│   ├── final-acceptance/
│   │   ├── plan.md
│   │   ├── progress.md
│   │   └── spec.md
│   ├── p5-navigation/
│   │   ├── issues/
│   │   │   ├── 00-source-protocol.md
│   │   │   ├── 01-scenes-navigation.md
│   │   │   ├── 02-depth.md
│   │   │   ├── 03-lidar-throughput.md
│   │   │   ├── 04-perception-ppo.md
│   │   │   ├── 05-dva.md
│   │   │   ├── 06-native-planners.md
│   │   │   ├── 07-formal-training.md
│   │   │   └── 08-delivery.md
│   │   └── spec.md
│   └── pointcloud-paper/
│       ├── implementation.md
│       ├── progress.md
│       └── spec.md
├── assets/
│   └── scenes/
│       ├── archive/
│       │   ├── p5_fixed_catalog_v1.json
│       │   ├── p5_fixed_catalog_v2.json
│       │   ├── p5_fixed_catalog_v3.json
│       │   └── p5_fixed_catalog_v4.json
│       └── navigation/
│           └── catalog.json
├── benchmarks/
│   └── navigation/
│       ├── v1/
│       │   ├── geometry-verification.json
│       │   ├── protocol.yaml
│       │   └── splits.yaml
│       └── v2/
│           └── protocol.yaml
├── configs/
│   ├── algorithm/
│   │   ├── apg.yaml
│   │   ├── bptt.yaml
│   │   ├── dva.yaml
│   │   ├── lotf_bptt.yaml
│   │   ├── none.yaml
│   │   ├── pointcloud_bptt.yaml
│   │   ├── ppo.yaml
│   │   └── shac.yaml
│   ├── controller/
│   │   ├── acceleration_passthrough.yaml
│   │   ├── crazyflow_attitude.yaml
│   │   ├── lotf_betaflight.yaml
│   │   └── trajectory_tracking.yaml
│   ├── dynamics/
│   │   ├── crazyflow.yaml
│   │   ├── lotf.yaml
│   │   └── paper_point_mass.yaml
│   ├── env/
│   │   ├── navigation/
│   │   │   ├── dynamic.yaml
│   │   │   └── static.yaml
│   │   ├── paper/
│   │   │   ├── control/
│   │   │   │   ├── hovering.yaml
│   │   │   │   ├── racing.yaml
│   │   │   │   └── tracking.yaml
│   │   │   ├── lotf_hover.yaml
│   │   │   ├── lotf_tracking.yaml
│   │   │   ├── pointcloud_flight.yaml
│   │   │   ├── pointcloud_navigation.yaml
│   │   │   └── pointcloud_navigation_v2.yaml
│   │   ├── tracking/
│   │   │   └── random.yaml
│   │   ├── hovering.yaml
│   │   ├── racing.yaml
│   │   └── tracking.yaml
│   ├── evaluation/
│   │   ├── default.yaml
│   │   ├── navigation_v1.yaml
│   │   ├── navigation_v2.yaml
│   │   ├── paper_pointcloud.yaml
│   │   └── pointcloud_navigation_v2.yaml
│   ├── execution/
│   │   ├── acceleration.yaml
│   │   ├── attitude_thrust.yaml
│   │   ├── lotf.yaml
│   │   └── trajectory_tracking.yaml
│   ├── method/
│   │   ├── learning/
│   │   │   ├── apg.yaml
│   │   │   ├── bptt.yaml
│   │   │   ├── dva.yaml
│   │   │   ├── ppo.yaml
│   │   │   └── shac.yaml
│   │   ├── optimization/
│   │   │   ├── attitude_mpc.yaml
│   │   │   └── sampling_mpc.yaml
│   │   └── paper/
│   │       ├── ego_planner.yaml
│   │       ├── lotf.yaml
│   │       ├── pointcloud_flight.yaml
│   │       └── super.yaml
│   ├── network/
│   │   ├── brax_apg.yaml
│   │   ├── brax_ppo.yaml
│   │   ├── brax_shac.yaml
│   │   ├── lotf_mlp.yaml
│   │   ├── none.yaml
│   │   ├── paper_pointnet_gru.yaml
│   │   ├── paper_pointnet_gru_conditioned.yaml
│   │   └── perception_ppo.yaml
│   ├── objective/
│   │   ├── lotf_hover.yaml
│   │   ├── lotf_tracking.yaml
│   │   ├── navigation.yaml
│   │   ├── paper_pointcloud.yaml
│   │   ├── pointcloud_control.yaml
│   │   └── tracking_exp.yaml
│   ├── observation/
│   │   ├── lotf_state.yaml
│   │   ├── navigation_depth.yaml
│   │   ├── navigation_lidar.yaml
│   │   ├── navigation_state.yaml
│   │   ├── paper_pointcloud.yaml
│   │   └── state_reference.yaml
│   ├── optimal_control/
│   │   ├── attitude_mpc.yaml
│   │   └── sampling_mpc.yaml
│   ├── runtime/
│   │   ├── host.yaml
│   │   └── jax.yaml
│   ├── scene/
│   │   ├── navigation/
│   │   │   ├── catalog.yaml
│   │   │   ├── dynamic.yaml
│   │   │   └── static.yaml
│   │   ├── empty.yaml
│   │   ├── lotf_world.yaml
│   │   ├── lsy_racing.yaml
│   │   └── paper_primitives.yaml
│   ├── sensor/
│   │   ├── d435.yaml
│   │   ├── mid360.yaml
│   │   ├── none.yaml
│   │   └── paper_uniform.yaml
│   ├── task/
│   │   ├── figure8.yaml
│   │   ├── hovering.yaml
│   │   ├── lotf_hover.yaml
│   │   ├── lotf_tracking.yaml
│   │   ├── navigation.yaml
│   │   ├── paper_transfer.yaml
│   │   ├── pointcloud_avoidance.yaml
│   │   ├── pointcloud_control.yaml
│   │   ├── racing.yaml
│   │   └── random.yaml
│   ├── training/
│   │   └── default.yaml
│   ├── visualization/
│   │   ├── headless.yaml
│   │   └── rscope.yaml
│   └── config.yaml
├── docs/
│   ├── adr/
│   │   ├── 0001-independent-project.md
│   │   ├── 0002-brax-rscope.md
│   │   ├── 0003-upstream-task-semantics.md
│   │   ├── 0004-composable-flight.md
│   │   ├── 0005-lotf-training-scope.md
│   │   ├── 0006-composable-slot-language.md
│   │   ├── 0007-method-execution-protocol.md
│   │   └── 0008-composable-architecture-v3.md
│   ├── agents/
│   │   ├── domain.md
│   │   ├── issue-tracker.md
│   │   └── workflow.md
│   ├── design/
│   │   ├── method-environment-layout.md
│   │   └── p3-p4-implementation.md
│   ├── diagrams/
│   │   ├── README.md
│   │   ├── composable-learning.architecture.json
│   │   ├── composable-learning.html
│   │   ├── composable-runtime.architecture.json
│   │   └── composable-runtime.html
│   ├── licenses/
│   │   └── dva-LICENSE.md
│   ├── research/
│   │   ├── archive/
│   │   │   ├── runbook-before-architecture-v3.md
│   │   │   └── status-before-architecture-v3.md
│   │   ├── alignment-history.md
│   │   ├── architecture-v3-timing.md
│   │   ├── branch-lifecycle.md
│   │   ├── composable-platform-references.md
│   │   ├── composable-slot-design.md
│   │   ├── p5-scene-reference-comparison.md
│   │   ├── pointcloud-paper-module-map.md
│   │   ├── pointcloud-paper-source-audit-20260928.md
│   │   └── references.md
│   ├── verification/
│   │   ├── architecture-v3/
│   │   │   ├── runs/
│   │   │   │   ├── attitude_mpc/
│   │   │   │   │   ├── report-summary.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── ego_dynamic/
│   │   │   │   │   ├── report-summary.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── ego_static/
│   │   │   │   │   ├── report-summary.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── play/
│   │   │   │   │   ├── report-summary.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── pointcloud_eval/
│   │   │   │   │   ├── report-summary.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── pointcloud_resume/
│   │   │   │   │   └── result.json
│   │   │   │   ├── pointcloud_train/
│   │   │   │   │   └── result.json
│   │   │   │   ├── ppo_dynamic_eval/
│   │   │   │   │   ├── report-summary.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── ppo_dynamic_train/
│   │   │   │   │   └── result.json
│   │   │   │   ├── ppo_hover_eval/
│   │   │   │   │   ├── report-summary.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── ppo_hover_train/
│   │   │   │   │   └── result.json
│   │   │   │   ├── ppo_racing_eval/
│   │   │   │   │   ├── report-summary.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── ppo_racing_train/
│   │   │   │   │   └── result.json
│   │   │   │   ├── ppo_static_eval/
│   │   │   │   │   ├── report-summary.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── ppo_static_train/
│   │   │   │   │   └── result.json
│   │   │   │   ├── ppo_tracking_eval/
│   │   │   │   │   ├── report-summary.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── ppo_tracking_train/
│   │   │   │   │   └── result.json
│   │   │   │   ├── sampling_mpc/
│   │   │   │   │   ├── report-summary.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── super_dynamic/
│   │   │   │   │   ├── report-summary.json
│   │   │   │   │   └── result.json
│   │   │   │   └── super_static/
│   │   │   │       ├── report-summary.json
│   │   │   │       └── result.json
│   │   │   ├── README.md
│   │   │   ├── current-doc-links.json
│   │   │   ├── evidence.json
│   │   │   ├── final-additions.log
│   │   │   ├── final-additions.xml
│   │   │   ├── final-checks.json
│   │   │   ├── format-recovered.log
│   │   │   ├── frozen-final.log
│   │   │   ├── frozen-final.xml
│   │   │   ├── last-fixes.log
│   │   │   ├── last-fixes.xml
│   │   │   ├── pixi-locked-2.log
│   │   │   ├── pointcloud-golden-result.json
│   │   │   ├── recipe-deduplication.json
│   │   │   ├── recovery-final.log
│   │   │   ├── recovery-final.xml
│   │   │   ├── recovery-overrides.log
│   │   │   ├── recovery-overrides.xml
│   │   │   ├── red-recovery-overrides.log
│   │   │   ├── review.md
│   │   │   ├── ruff-recovered.log
│   │   │   ├── smoke-envs.json
│   │   │   ├── source-cache-verify-2.log
│   │   │   ├── source-patches-verified.json
│   │   │   ├── suite-2.log
│   │   │   ├── suite-2.xml
│   │   │   ├── suite-final.log
│   │   │   ├── suite-final.xml
│   │   │   ├── suite-recovered.log
│   │   │   └── suite-recovered.xml
│   │   ├── composable-lotf-figures/
│   │   │   ├── lotf-hybrid-hover-seed0-v1-error.png
│   │   │   ├── lotf-hybrid-hover-seed0-v1-loss.png
│   │   │   ├── lotf-hybrid-hover-seed0-v1-trajectory.png
│   │   │   ├── lotf-hybrid-tracking-seed0-v1-error.png
│   │   │   ├── lotf-hybrid-tracking-seed0-v1-loss.png
│   │   │   └── lotf-hybrid-tracking-seed0-v1-trajectory.png
│   │   ├── final-acceptance/
│   │   │   ├── runs/
│   │   │   │   ├── final-acceptance-bptt-hovering-t0/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-bptt-navigation-dynamic-smoke-v3/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-bptt-navigation-static-smoke-v3/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-bptt-racing-t0/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-bptt-tracking-t0/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-bptt-hovering-v1/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-bptt-navigation-dynamic-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-bptt-navigation-static-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-bptt-racing-v1/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-bptt-tracking-v1/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-pointcloud-hovering-v1/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-pointcloud-racing-v1/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-pointcloud-tracking-v1/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-ppo-hovering-v1/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-ppo-navigation-dynamic-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-ppo-navigation-static-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-ppo-racing-v1/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-ppo-tracking-v1/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-shac-hovering-v1/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-shac-navigation-dynamic-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-shac-navigation-static-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-shac-racing-trained-v1/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-heldout-shac-tracking-v1/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-native-ego_planner-dynamic-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-native-ego_planner-hovering-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-native-ego_planner-racing-v3/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-native-ego_planner-static-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-native-ego_planner-tracking-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-native-super-dynamic-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-native-super-hovering-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-native-super-racing-v3/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-native-super-static-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-native-super-tracking-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-pointcloud-hovering-t2/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-pointcloud-navigation-30000-v2/
│   │   │   │   │   ├── eval/
│   │   │   │   │   │   └── report.json
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-pointcloud-racing-t1/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-pointcloud-tracking-t1/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-ppo-hovering-t0/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-ppo-navigation-dynamic-smoke-v2/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-ppo-navigation-static-smoke-v2/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-ppo-racing-t0/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-ppo-tracking-t0/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-shac-hovering-t0/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-shac-navigation-dynamic-smoke-v3/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-shac-navigation-static-smoke-v3/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   ├── final-acceptance-shac-racing-t1/
│   │   │   │   │   ├── command.txt
│   │   │   │   │   ├── manifest.json
│   │   │   │   │   ├── resolved-config.json
│   │   │   │   │   └── result.json
│   │   │   │   └── final-acceptance-shac-tracking-t0/
│   │   │   │       ├── command.txt
│   │   │   │       ├── manifest.json
│   │   │   │       ├── resolved-config.json
│   │   │   │       └── result.json
│   │   │   ├── tuning/
│   │   │   │   ├── final-acceptance-bptt-hovering-t0/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-bptt-navigation-dynamic-smoke-v2/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-bptt-navigation-dynamic-smoke-v3/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-bptt-navigation-static-smoke-v2/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-bptt-navigation-static-smoke-v3/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-bptt-racing-t0/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-bptt-tracking-t0/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-pointcloud-hovering-t0/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-pointcloud-hovering-t0-cli-v2/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-pointcloud-hovering-t1/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-pointcloud-hovering-t2/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-pointcloud-racing-t1/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-pointcloud-tracking-t0/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-pointcloud-tracking-t1/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-ppo-hovering-t0/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-ppo-navigation-dynamic-smoke-v2/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-ppo-navigation-static-smoke-v2/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-ppo-racing-t0/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-ppo-tracking-t0/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-shac-hovering-t0/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-shac-navigation-dynamic-smoke-v2/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-shac-navigation-dynamic-smoke-v3/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-shac-navigation-static-smoke-v2/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-shac-navigation-static-smoke-v3/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-shac-racing-t0/
│   │   │   │   │   └── observation.json
│   │   │   │   ├── final-acceptance-shac-racing-t1/
│   │   │   │   │   └── observation.json
│   │   │   │   └── final-acceptance-shac-tracking-t0/
│   │   │   │       └── observation.json
│   │   │   ├── verification-sources/
│   │   │   │   ├── finalize_verified_delivery.py.txt
│   │   │   │   ├── prepare_delivery.py.txt
│   │   │   │   ├── render_delivery.py.txt
│   │   │   │   └── verify_delivery_replays.py.txt
│   │   │   ├── README.md
│   │   │   ├── commands.md
│   │   │   ├── current-doc-links.json
│   │   │   ├── diff-check.log
│   │   │   ├── ego-3d-goals-build.log
│   │   │   ├── evidence-audit.log
│   │   │   ├── evidence.json
│   │   │   ├── final-additions-complete.log
│   │   │   ├── final-additions-complete.xml
│   │   │   ├── final-regression.log
│   │   │   ├── final-regression.xml
│   │   │   ├── format.log
│   │   │   ├── full-regression-recovery.log
│   │   │   ├── green-pointcloud-conditioning.log
│   │   │   ├── green-sensor-gradients.log
│   │   │   ├── legacy-checkpoints.json
│   │   │   ├── native-control-exits-v2.json
│   │   │   ├── original-paper-snapshot.json
│   │   │   ├── pointcloud-feature-diagnosis.json
│   │   │   ├── red-acceptance-summary.log
│   │   │   ├── red-native-provenance.log
│   │   │   ├── red-native-takeoff.log
│   │   │   ├── red-pointcloud-conditioning.log
│   │   │   ├── red-sensor-gradients.log
│   │   │   ├── remaining-frozen-evaluations.json
│   │   │   ├── replay-readback.json
│   │   │   ├── ruff.log
│   │   │   ├── selection.json
│   │   │   ├── shac-trained-endpoint-selection.json
│   │   │   ├── source-hashes.json
│   │   │   ├── summary-guard-final.log
│   │   │   ├── super-control-initial-time-build-v2.log
│   │   │   ├── terminal-readback-complete.log
│   │   │   └── verification.json
│   │   ├── images/
│   │   │   ├── p4-ppo.png
│   │   │   ├── p4-sampling-mpc.png
│   │   │   ├── p4-vscode-apg.png
│   │   │   └── p4-vscode-attitude-mpc.png
│   │   ├── p5-results-v2/
│   │   │   ├── cells.csv
│   │   │   ├── development-curves.csv
│   │   │   ├── development-curves.pdf
│   │   │   ├── development-curves.png
│   │   │   ├── development-diagnostics.pdf
│   │   │   ├── development-diagnostics.png
│   │   │   ├── episodes.csv
│   │   │   ├── heldout-matrix.pdf
│   │   │   ├── heldout-matrix.png
│   │   │   ├── replays.json
│   │   │   ├── report.md
│   │   │   ├── scene-splits.json
│   │   │   ├── stage-status.md
│   │   │   ├── subtypes.csv
│   │   │   ├── summary.json
│   │   │   ├── training.csv
│   │   │   └── units.csv
│   │   ├── pointcloud-paper-stage1/
│   │   │   ├── verified-delivery/
│   │   │   │   ├── episodes.csv
│   │   │   │   ├── module-slots.json
│   │   │   │   ├── reconstruction-assumptions.json
│   │   │   │   ├── report.md
│   │   │   │   └── summary.json
│   │   │   ├── README.md
│   │   │   ├── coordinator-recovery.json
│   │   │   ├── delivery-verification.json
│   │   │   ├── evidence.json
│   │   │   ├── supervisor-source-audit.json
│   │   │   ├── training-loss.png
│   │   │   └── training-speed.png
│   │   ├── 2026-09-25-brax-integration.md
│   │   ├── brax-20260925-evidence-manifest.json
│   │   ├── brax-20260925-results.json
│   │   ├── composable-architecture-20260928.json
│   │   ├── composable-design-check.json
│   │   ├── composable-legacy-checkpoints.json
│   │   ├── composable-lotf-delivery.md
│   │   ├── composable-lotf-engineering.md
│   │   ├── composable-lotf-final-checks.json
│   │   ├── composable-lotf-replays.json
│   │   ├── composable-lotf-results.json
│   │   ├── composable-lotf-resume-check.json
│   │   ├── composable-lotf-review-evaluation.json
│   │   ├── composable-lotf-review-replays.json
│   │   ├── composable-lotf-tests.log
│   │   ├── crazyflow-dependency-changes.patch
│   │   ├── navigation8.json
│   │   ├── p1-observation-checks.json
│   │   ├── p1-observation-checks.md
│   │   ├── p1-p2-delivery.md
│   │   ├── p1-p2-final-checks.json
│   │   ├── p1-p2-results.json
│   │   ├── p2-independent-evaluation.json
│   │   ├── p2-independent-evaluation.md
│   │   ├── p3-matrix.json
│   │   ├── p3-numerical-diagnosis.json
│   │   ├── p3-numerical-diagnosis.md
│   │   ├── p3-p4-delivery.md
│   │   ├── p3-p4-engineering.md
│   │   ├── p3-p4-final-checks.json
│   │   ├── p3-p4-replay-acceptance.json
│   │   ├── p3-p4-results.json
│   │   ├── p3-p4-results.md
│   │   ├── p3-p4-tests.log
│   │   ├── p3-recovery-replay-verification.json
│   │   ├── p3-replay-verification.json
│   │   ├── p4-editor-verification.json
│   │   ├── p4-learning-v1.json
│   │   ├── p4-optimization-v2.json
│   │   ├── p4-replay-verification.json
│   │   ├── p4-visual-check.json
│   │   ├── p5-archive-export.json
│   │   ├── p5-dva-gpu-resume.json
│   │   ├── p5-fixed-scenes-review-v1.json
│   │   ├── p5-fixed-scenes-review-v3.json
│   │   ├── p5-fixed-scenes-review-v4.json
│   │   ├── p5-implementation-review.md
│   │   ├── p5-native-build.json
│   │   ├── p5-noise-parameterization.json
│   │   ├── p5-sensor-reconstruction.json
│   │   ├── p5-source-inventory.json
│   │   ├── p5-tests.json
│   │   ├── p5-throughput.json
│   │   ├── pointcloud-paper-method-tests.xml
│   │   ├── pointcloud-summary-checks-20260928.json
│   │   └── project-setup-checks.json
│   ├── architecture.md
│   ├── backlog.md
│   ├── evaluation.md
│   ├── project-tree.md
│   ├── runbook.md
│   └── status.md
├── experiments/
│   └── README.md
├── native_planners/
│   ├── bridge/
│   │   └── worker.py
│   ├── docker/
│   │   └── Dockerfile.ros1
│   ├── patches/
│   │   ├── ego-3d-goals.patch
│   │   └── super-control-initial-time.patch
│   ├── README.md
│   ├── setup.sh
│   └── versions.env
├── scripts/
│   ├── tools/
│   │   ├── build_navigation.py
│   │   ├── export_navigation.py
│   │   ├── export_p5_archived_case.py
│   │   ├── fetch_sources.py
│   │   ├── migrate_artifact.py
│   │   ├── p5_throughput_probe.py
│   │   ├── plot_p5.py
│   │   ├── reconstruct_p5_sensor.py
│   │   ├── rscope_client.py
│   │   ├── run_p5_archive_repairs.py
│   │   ├── run_pointcloud_pipeline.py
│   │   ├── setup_acados.sh
│   │   ├── setup_p5_native.sh
│   │   ├── summarize_composable_lotf.py
│   │   ├── summarize_final_acceptance.py
│   │   ├── summarize_p3_p4.py
│   │   ├── summarize_p5.py
│   │   ├── summarize_pointcloud.py
│   │   ├── verify_p5_replays.py
│   │   ├── verify_p5_scene_splits.py
│   │   ├── verify_p5_sensor_reconstruction.py
│   │   └── verify_phase_replays.py
│   ├── README.md
│   ├── eval.py
│   ├── play.py
│   └── train.py
├── src/
│   └── drone_playground/
│       ├── environments/
│       │   ├── observations/
│       │   │   └── __init__.py
│       │   ├── scenes/
│       │   │   ├── __init__.py
│       │   │   ├── catalog.py
│       │   │   ├── control_geometry.py
│       │   │   ├── navigation.py
│       │   │   └── pointcloud.py
│       │   ├── sensors/
│       │   │   ├── __init__.py
│       │   │   ├── depth.py
│       │   │   ├── lidar.py
│       │   │   ├── pointcloud.py
│       │   │   └── rays.py
│       │   ├── tasks/
│       │   │   ├── lsy_upstream/
│       │   │   │   ├── assets/
│       │   │   │   │   ├── gate.xml
│       │   │   │   │   ├── gate_bottom.png
│       │   │   │   │   ├── gate_left.png
│       │   │   │   │   ├── gate_right.png
│       │   │   │   │   ├── gate_top.png
│       │   │   │   │   └── obstacle.xml
│       │   │   │   ├── LICENSE
│       │   │   │   ├── __init__.py
│       │   │   │   ├── compat.py
│       │   │   │   ├── compatibility.patch
│       │   │   │   ├── level0.toml
│       │   │   │   ├── provenance.json
│       │   │   │   ├── race_core.py
│       │   │   │   ├── randomize.py
│       │   │   │   └── utils.py
│       │   │   ├── README.md
│       │   │   ├── __init__.py
│       │   │   ├── lotf.py
│       │   │   ├── navigation.py
│       │   │   ├── pointcloud.py
│       │   │   ├── pointcloud_control.py
│       │   │   ├── racing.py
│       │   │   └── tracking.py
│       │   ├── __init__.py
│       │   └── environment.py
│       ├── evaluation/
│       │   ├── README.md
│       │   ├── __init__.py
│       │   ├── evaluator.py
│       │   ├── lotf.py
│       │   ├── native_control.py
│       │   ├── native_planners.py
│       │   ├── navigation.py
│       │   ├── optimization.py
│       │   ├── pointcloud.py
│       │   ├── pointcloud_control.py
│       │   ├── protocols.py
│       │   ├── racing.py
│       │   ├── reporting.py
│       │   ├── trace_archive.py
│       │   └── tracking.py
│       ├── execution/
│       │   ├── controllers/
│       │   │   ├── README.md
│       │   │   ├── __init__.py
│       │   │   ├── acceleration.py
│       │   │   ├── crazyflow.py
│       │   │   ├── demo.py
│       │   │   ├── lotf.py
│       │   │   └── trajectory.py
│       │   ├── __init__.py
│       │   ├── delay.py
│       │   └── transition.py
│       ├── integrations/
│       │   ├── __init__.py
│       │   └── native_planner.py
│       ├── learning/
│       │   ├── algorithms/
│       │   │   ├── __init__.py
│       │   │   ├── bptt.py
│       │   │   ├── dva.py
│       │   │   ├── lotf_bptt.py
│       │   │   ├── pointcloud_bptt.py
│       │   │   ├── pointcloud_control.py
│       │   │   └── shac.py
│       │   ├── objectives/
│       │   │   ├── __init__.py
│       │   │   └── pointcloud.py
│       │   ├── README.md
│       │   ├── __init__.py
│       │   ├── env_adapter.py
│       │   └── train.py
│       ├── methods/
│       │   ├── optimal_control/
│       │   │   ├── lsy_upstream/
│       │   │   │   ├── LICENSE
│       │   │   │   ├── __init__.py
│       │   │   │   ├── attitude_mpc.py
│       │   │   │   ├── compatibility.patch
│       │   │   │   ├── controller.py
│       │   │   │   └── provenance.json
│       │   │   ├── __init__.py
│       │   │   ├── factory.py
│       │   │   ├── lsy_mpc.py
│       │   │   └── sampling.py
│       │   ├── planners/
│       │   │   ├── __init__.py
│       │   │   └── reference.py
│       │   ├── __init__.py
│       │   └── neural.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── crazyflow.py
│       │   ├── gradients.py
│       │   ├── lotf.py
│       │   └── point_mass.py
│       ├── networks/
│       │   ├── __init__.py
│       │   ├── encoders.py
│       │   ├── pointcloud.py
│       │   └── policies.py
│       ├── runs/
│       │   ├── README.md
│       │   ├── __init__.py
│       │   ├── checkpoints.py
│       │   ├── console.py
│       │   ├── migration.py
│       │   ├── pointcloud.py
│       │   └── record.py
│       ├── runtime/
│       │   ├── __init__.py
│       │   ├── devices.py
│       │   ├── host_runner.py
│       │   └── jax_runner.py
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── lotf_scene.py
│       │   ├── navigation_scene.py
│       │   ├── rscope_io.py
│       │   └── viewer.py
│       ├── __init__.py
│       ├── app.py
│       ├── cli.py
│       ├── composition.py
│       └── contracts.py
├── tests/
│   ├── fixtures/
│   │   ├── architecture-v2-recipes.json
│   │   ├── architecture-v3-recipes.json
│   │   ├── composition-baseline.json
│   │   └── composition-baseline.npz
│   ├── README.md
│   ├── __init__.py
│   ├── reference_configs.py
│   ├── test_acceptance_summary.py
│   ├── test_architecture_v3.py
│   ├── test_artifact_migration_v3.py
│   ├── test_bptt_training.py
│   ├── test_checkpoint.py
│   ├── test_cli.py
│   ├── test_component_regression.py
│   ├── test_composition.py
│   ├── test_console.py
│   ├── test_control_geometry.py
│   ├── test_depth_sensor.py
│   ├── test_device_ownership.py
│   ├── test_dva_navigation.py
│   ├── test_evaluation.py
│   ├── test_execution_config.py
│   ├── test_final_acceptance.py
│   ├── test_fixed_navigation_scenes.py
│   ├── test_frozen_contract_v3.py
│   ├── test_lidar_sensor.py
│   ├── test_live_replay.py
│   ├── test_lotf.py
│   ├── test_lotf_review.py
│   ├── test_lotf_training.py
│   ├── test_native_control.py
│   ├── test_native_planners.py
│   ├── test_navigation.py
│   ├── test_navigation_archive.py
│   ├── test_nested_composition.py
│   ├── test_numerical_boundary.py
│   ├── test_optimization.py
│   ├── test_perception_ppo.py
│   ├── test_pointcloud_conditioning.py
│   ├── test_pointcloud_control.py
│   ├── test_pointcloud_control_pipeline.py
│   ├── test_pointcloud_evaluation.py
│   ├── test_pointcloud_method.py
│   ├── test_pointcloud_navigation_v2.py
│   ├── test_pointcloud_physics.py
│   ├── test_pointcloud_pipeline.py
│   ├── test_pointcloud_protocol_guards.py
│   ├── test_pointcloud_summary.py
│   ├── test_pointcloud_training.py
│   ├── test_protocols_v3.py
│   ├── test_racing.py
│   ├── test_reporting.py
│   ├── test_rscope_client.py
│   ├── test_runs.py
│   ├── test_runtime_terminal_compute.py
│   ├── test_runtime_v3.py
│   ├── test_sensor_gradients.py
│   ├── test_shac.py
│   ├── test_shac_warm_start.py
│   └── test_tracking.py
├── third_party/
│   ├── licenses/
│   │   ├── Crazyflow.txt
│   │   └── LOTF.txt
│   ├── patches/
│   │   ├── crazyflow-mujoco-version.patch
│   │   └── lotf-package-metadata.patch
│   └── sources.yaml
├── tmp/
│   └── README.md
├── .gitattributes
├── .gitignore
├── AGENTS.md
├── CONTEXT.md
├── LICENSE
├── README.md
├── THIRD_PARTY_NOTICES.md
├── pixi.lock
└── pyproject.toml
```

## 运行时目录

```text
experiments/<run-id>/
├── manifest.json
├── command.txt
├── state.json
├── console.log
├── result.json
├── metrics/
├── checkpoints/
├── training-state/             # 具名训练器需要时生成
├── eval/
└── rollouts/

tmp/
├── sources/                    # 固定上游源码与打包补丁
├── architecture-v3/            # 本次探针、命令日志与中间证据
└── hydra/

.pixi/envs/default/             # 本工作树独立依赖环境
```

`env` 下的配置组合场景、任务、传感器、观测与执行层。`dynamics` 的实现位于 `models/`；预测模型属于具体方法，训练导数选择属于 `algorithm.gradient`。PPO/APG 复用 Brax 更新器，其余具名更新器位于 `learning/algorithms/`。源码物理目录只表达实际已有实现。
