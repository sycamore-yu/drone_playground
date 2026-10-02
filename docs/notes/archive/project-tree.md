# 当前完整源码与配置目录

缓存、实验产物和第三方内部文件不展开。职责见 [目录说明](directory-layout.md)。

```text
src/drone_playground/
├── artifacts/
│   ├── README.md
│   ├── __init__.py
│   ├── checkpoints.py
│   ├── console.py
│   ├── decisions.py
│   ├── layout.py
│   ├── record.py
│   ├── reporting.py
│   ├── schema.py
│   ├── source.py
│   ├── traces.py
│   └── training_state.py
├── environments/
│   ├── observations/
│   │   ├── __init__.py
│   │   └── flight_state.py
│   ├── scenes/
│   │   ├── __init__.py
│   │   ├── catalog.py
│   │   ├── empty.py
│   │   ├── generated_navigation.py
│   │   ├── geometry.py
│   │   ├── mujoco_geometry.py
│   │   ├── primitives.py
│   │   └── racing.py
│   ├── sensors/
│   │   ├── __init__.py
│   │   ├── depth.py
│   │   ├── lidar.py
│   │   └── rays.py
│   ├── tasks/
│   │   ├── lsy_upstream/
│   │   ├── navigation/
│   │   │   ├── __init__.py
│   │   │   ├── acceleration.py
│   │   │   ├── initialization.py
│   │   │   ├── recurrent.py
│   │   │   └── rigid_body.py
│   │   ├── tracking/
│   │   │   ├── __init__.py
│   │   │   ├── acceleration.py
│   │   │   └── rigid_body.py
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── racing.py
│   ├── README.md
│   ├── __init__.py
│   ├── environment.py
│   ├── randomization.py
│   └── references.py
├── evaluation/
│   ├── navigation/
│   │   ├── __init__.py
│   │   ├── acceleration.py
│   │   ├── cases.py
│   │   ├── external.py
│   │   ├── policy.py
│   │   └── recurrent.py
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── acceleration.py
│   │   ├── external.py
│   │   └── policy.py
│   ├── README.md
│   ├── __init__.py
│   ├── benchmarks.py
│   ├── mpc.py
│   ├── racing.py
│   └── run.py
├── execution/
│   ├── controllers/
│   │   ├── mpc/
│   │   │   ├── lsy_upstream/
│   │   │   ├── __init__.py
│   │   │   ├── delay_prediction.py
│   │   │   ├── factory.py
│   │   │   ├── lsy_mpc.py
│   │   │   └── sampling.py
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── acceleration.py
│   │   ├── bodyrates.py
│   │   ├── crazyflow.py
│   │   ├── demo.py
│   │   ├── trajectory.py
│   │   ├── trajectory_jax.py
│   │   └── velocity.py
│   ├── __init__.py
│   ├── commands.py
│   ├── delay.py
│   ├── geometric_policy.py
│   ├── native_tracking.py
│   └── transition.py
├── integrations/
│   ├── __init__.py
│   ├── grpc_service.py
│   ├── pipeline.py
│   └── ros1.py
├── learning/
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── bptt.py
│   │   ├── dva.py
│   │   ├── recurrent_bptt.py
│   │   ├── recurrent_navigation_bptt.py
│   │   ├── recurrent_tracking.py
│   │   └── shac.py
│   ├── objectives/
│   │   ├── __init__.py
│   │   ├── acceleration_flight.py
│   │   ├── navigation.py
│   │   └── navigation_acceleration.py
│   ├── README.md
│   ├── __init__.py
│   ├── env_adapter.py
│   └── train.py
├── models/
│   ├── __init__.py
│   ├── crazyflow.py
│   ├── lotf.py
│   └── point_mass.py
├── networks/
│   ├── README.md
│   ├── __init__.py
│   ├── cnn_gru.py
│   ├── encoders.py
│   ├── physical_outputs.py
│   ├── pointnet_gru.py
│   ├── policies.py
│   └── range_encoder.py
├── planning/
│   ├── __init__.py
│   └── minimum_jerk.py
├── rpc/
│   ├── proto/
│   │   ├── __init__.py
│   │   ├── algorithm.proto
│   │   └── algorithm_pb2.py
│   ├── README.md
│   ├── __init__.py
│   ├── client.py
│   ├── geometry.py
│   ├── ros_trajectory.py
│   ├── ros_visualization.py
│   ├── server.py
│   ├── waypoints.py
│   └── wire.py
├── runtime/
│   ├── __init__.py
│   ├── devices.py
│   ├── host_runner.py
│   ├── jax_runner.py
│   └── timing.py
├── visualization/
│   ├── __init__.py
│   ├── layers.py
│   ├── navigation_scene.py
│   ├── rscope_io.py
│   ├── sensor_hits.py
│   └── viewer.py
├── __init__.py
├── app.py
├── cli.py
└── composition.py
```

```text
configs/
├── algorithm/
│   ├── apg.yaml
│   ├── bptt.yaml
│   ├── dva.yaml
│   ├── none.yaml
│   ├── ppo.yaml
│   ├── recurrent_bptt.yaml
│   ├── recurrent_navigation_bptt.yaml
│   ├── shac.yaml
│   └── shac_diffaero_adapter.yaml
├── controller/
│   ├── acceleration_passthrough.yaml
│   ├── attitude_mpc.yaml
│   ├── bodyrates.yaml
│   ├── crazyflow_attitude.yaml
│   ├── jax_trajectory_tracking.yaml
│   ├── jax_waypoint_tracking.yaml
│   ├── sampling_mpc.yaml
│   ├── trajectory_curve_tracking.yaml
│   ├── trajectory_tracking.yaml
│   ├── velocity_yaw.yaml
│   └── waypoint_tracking.yaml
├── domain_randomization/
├── dynamics/
│   ├── crazyflow_first_principles.yaml
│   ├── crazyflow_first_principles_cf21b_500.yaml
│   ├── crazyflow_so_rpy.yaml
│   ├── crazyflow_so_rpy_cf21b_500.yaml
│   ├── crazyflow_so_rpy_rotor.yaml
│   ├── crazyflow_so_rpy_rotor_drag.yaml
│   ├── lotf_high_fidelity.yaml
│   ├── lotf_simplified.yaml
│   └── point_mass_lag.yaml
├── env/
│   ├── hovering/
│   │   └── acceleration.yaml
│   ├── navigation/
│   │   ├── acceleration_catalog.yaml
│   │   ├── acceleration_procedural.yaml
│   │   ├── dynamic.yaml
│   │   ├── dynamic_velocity.yaml
│   │   ├── recurrent_acceleration.yaml
│   │   ├── recurrent_depth.yaml
│   │   ├── recurrent_dynamic.yaml
│   │   ├── recurrent_static.yaml
│   │   ├── static.yaml
│   │   └── static_velocity.yaml
│   ├── racing/
│   │   └── acceleration.yaml
│   ├── tracking/
│   │   ├── acceleration.yaml
│   │   └── random.yaml
│   ├── hovering.yaml
│   ├── racing.yaml
│   └── tracking.yaml
├── evaluation/
│   ├── default.yaml
│   ├── native_navigation_benchmark.yaml
│   ├── native_navigation_primary_benchmark.yaml
│   ├── navigation_acceleration.yaml
│   ├── navigation_acceleration_benchmark.yaml
│   ├── navigation_benchmark.yaml
│   ├── navigation_primary_benchmark.yaml
│   ├── navigation_v2.yaml
│   ├── racing_benchmark.yaml
│   └── tracking_benchmark.yaml
├── execution/
│   ├── acceleration.yaml
│   ├── attitude_thrust.yaml
│   ├── trajectory_tracking.yaml
│   ├── velocity_yaw.yaml
│   └── waypoint_tracking.yaml
├── experiment/
│   ├── learning/
│   │   ├── apg.yaml
│   │   ├── bptt.yaml
│   │   ├── dva.yaml
│   │   ├── geometric.yaml
│   │   ├── navigation_bptt.yaml
│   │   ├── navigation_ppo.yaml
│   │   ├── navigation_shac.yaml
│   │   ├── ppo.yaml
│   │   └── shac.yaml
│   ├── optimization/
│   │   ├── attitude_mpc.yaml
│   │   └── sampling_mpc.yaml
│   ├── papers/
│   │   ├── depth_diffphysics.yaml
│   │   ├── ego_planner.yaml
│   │   ├── pointcloud_diffsim.yaml
│   │   └── super.yaml
│   ├── transfers/
│   │   ├── pointcloud_acceleration_benchmark.yaml
│   │   ├── pointcloud_hovering.yaml
│   │   ├── pointcloud_racing.yaml
│   │   └── pointcloud_tracking.yaml
│   ├── native.yaml
│   └── pipeline.yaml
├── network/
│   ├── depth_cnn_gru.yaml
│   ├── gaussian_mlp_64.yaml
│   ├── mlp_256_128.yaml
│   ├── mlp_32.yaml
│   ├── mlp_64.yaml
│   ├── none.yaml
│   ├── pointnet_gru.yaml
│   ├── pointnet_gru_scaled.yaml
│   ├── polar_range_mlp_128.yaml
│   ├── sensor_fusion_mlp_128.yaml
│   ├── sensor_fusion_mlp_64.yaml
│   └── sensor_fusion_mlp_64_velocity_init.yaml
├── objective/
│   ├── acceleration_flight.yaml
│   ├── navigation.yaml
│   ├── navigation_acceleration.yaml
│   ├── navigation_motion.yaml
│   ├── reference_tracking.yaml
│   └── tracking_exp.yaml
├── observation/
│   ├── flight_body.yaml
│   ├── flight_state.yaml
│   ├── navigation_depth.yaml
│   ├── navigation_lidar.yaml
│   ├── navigation_state.yaml
│   └── state_reference.yaml
├── optimal_control/
│   ├── attitude_mpc.yaml
│   └── sampling_mpc.yaml
├── runtime/
│   ├── host.yaml
│   └── jax.yaml
├── scene/
│   ├── navigation/
│   │   ├── catalog.yaml
│   │   ├── dynamic.yaml
│   │   └── static.yaml
│   ├── empty.yaml
│   ├── primitives.yaml
│   └── racing.yaml
├── sensor/
│   ├── d435.yaml
│   ├── mid360.yaml
│   ├── none.yaml
│   ├── pinhole_depth.yaml
│   └── uniform_lidar.yaml
├── task/
│   ├── tracking/
│   │   ├── figure8.yaml
│   │   └── random.yaml
│   ├── hovering.yaml
│   ├── navigation.yaml
│   └── racing.yaml
├── training/
│   ├── default.yaml
│   ├── navigation_checkpoint_eval_primary.yaml
│   ├── navigation_checkpoint_eval_v2.yaml
│   ├── navigation_convergence.yaml
│   ├── navigation_independent.yaml
│   ├── navigation_initial_states.yaml
│   ├── navigation_recurrent.yaml
│   └── release_pilot.yaml
├── visualization/
│   ├── headless.yaml
│   └── rscope.yaml
└── config.yaml
```

```text
native/
├── include/
│   └── drone_native/
│       └── algorithm.h
├── src/
│   └── service.cc
├── tests/
│   └── interop_server.cc
├── CMakeLists.txt
├── README.md
├── pixi.lock
└── pixi.toml
```

```text
ros1_planners/
├── bridge/
│   └── worker.py
├── docker/
│   └── Dockerfile.ros1
├── patches/
│   ├── ego-3d-goals.patch
│   ├── super-control-initial-time.patch
│   └── super-heartbeat-lock.patch
├── README.md
├── setup.sh
└── versions.env
```

```text
scripts/
├── experiments/
│   ├── confirm_control_learning.py
│   ├── confirm_control_solvers.py
│   ├── confirm_navigation_learning.py
│   ├── run_pointcloud_pipeline.py
│   ├── summarize_final_acceptance.py
│   ├── summarize_pointcloud.py
│   └── verify_phase_replays.py
├── tools/
│   ├── build_navigation.py
│   ├── enhance_replay.py
│   ├── export_navigation.py
│   ├── export_source.py
│   ├── fetch_sources.py
│   ├── organize_experiments.py
│   ├── rscope_client.py
│   └── setup_acados.sh
├── README.md
├── eval.py
├── play.py
└── train.py
```

```text
assets/scenes/
├── navigation/
│   └── catalog.json
└── README.md
```
