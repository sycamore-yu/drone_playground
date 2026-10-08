"""Replay-only MuJoCo model for a navigation scenario instance.

The replay model is generated from the frozen scene bank, so rscope shows the
same primitives the sensor, the reward and the collision test consumed. A
navigation batch mixes instances, so one model is built per exported episode and
the obstacles are animated through mocap bodies by the rollout exporter.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from drone_playground.environments.scenes.geometry import SceneBank, obstacle_positions
from drone_playground.environments.scenes.mjcf import instance_spec
from drone_playground.resources import resource_path


def active_indices(bank: SceneBank, scenario_id: int) -> np.ndarray:
    """Capacity indices that carry geometry for one instance, in replay order."""
    return np.flatnonzero(np.asarray(bank.active[scenario_id]))


def obstacle_track(bank: SceneBank, scenario_id: int, times: np.ndarray) -> np.ndarray:
    """Obstacle centers over time with shape ``[T, capacity, 3]``."""
    positions = jax.vmap(lambda time: obstacle_positions(bank, jnp.int32(scenario_id), time))(
        jnp.asarray(np.asarray(times, np.float32))
    )
    return np.asarray(positions)


def create_replay_model(env, scenario_id: int):
    """MuJoCo replay model for one navigation instance of a built environment."""
    bank = env.bank
    indices = active_indices(bank, scenario_id)
    corridor_low = np.asarray(bank.world_low, float)
    corridor_high = np.asarray(bank.world_high, float)
    centre = (corridor_low + corridor_high) / 2.0
    half = (corridor_high - corridor_low) / 2.0
    start = np.asarray(bank.start[scenario_id], float)
    goal = np.asarray(bank.goal[scenario_id], float)
    dt = env.dt
    xml = f"""<mujoco model="navigation-replay">
  <compiler angle="radian"/>
  <option timestep="{dt}" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1600" offheight="1000"/>
    <map znear="0.001"/>
  </visual>
  <worldbody>
    <light pos="8 0 9" dir="0 0 -1"/>
    <geom name="ground" type="plane" pos="0 0 {corridor_low[2]}"
          size="{half[0]} {half[1]} .1" rgba=".86 .88 .9 1"/>
    <geom name="floor_marker" type="box" pos="{centre[0]} {centre[1]} {corridor_low[2] - 0.02}"
          size="{half[0]} {half[1]} 0.02" rgba=".78 .82 .86 1"/>
    <geom name="start_marker" type="sphere" pos="{start[0]} {start[1]} {start[2]}"
          size="0.12" rgba="0.2 0.8 0.3 0.8"/>
    <geom name="goal_marker" type="sphere" pos="{goal[0]} {goal[1]} {goal[2]}"
          size="0.5" rgba="0.9 0.8 0.2 0.35"/>
  </worldbody>
</mujoco>"""
    spec = mujoco.MjSpec.from_string(xml)
    robot_path = resource_path("assets/robots/crazyflie2x/replay.xml")
    robot_spec = mujoco.MjSpec.from_file(str(robot_path))
    robot_spec.compiler.meshdir = str(robot_path.parent)
    spec.compiler.meshdir = str(robot_path.parent)
    spec.attach(robot_spec, prefix="", suffix="", frame=spec.worldbody.add_frame())
    scene_spec = instance_spec(bank, scenario_id)
    spec.attach(scene_spec, prefix="", suffix="", frame=spec.worldbody.add_frame())
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    from drone_playground.visualization.layers import ReplayLayers, sensor_view

    sensor = getattr(env, "sensor", None)
    experiment_config = getattr(env, "experiment_config", {})
    visualization = (
        experiment_config.get("visualization", {}) if isinstance(experiment_config, dict) else {}
    )
    from drone_playground.visualization.sensor_hits import ReplaySensorContext, supports_hit_overlay

    replay_sensor_context = None
    if (
        sensor is not None
        and supports_hit_overlay(sensor)
        and visualization.get("sensor_hits", True)
    ):
        replay_sensor_context = ReplaySensorContext(
            sensor=sensor,
            bank=bank,
            scenario_id=int(scenario_id),
            max_points=int(visualization.get("max_sensor_points", 2400)),
        )
    return SimpleNamespace(
        spec=spec,
        mj_model=model,
        replay_visualization=ReplayLayers(
            sensor=sensor_view(sensor.calibration()) if sensor is not None else None
        ),
        replay_sensor_context=replay_sensor_context,
        data=SimpleNamespace(
            core=SimpleNamespace(
                drone_mocap_ids=np.array([int(model.body("drone").mocapid[0])]),
                obstacle_mocap_ids=np.array(
                    [int(model.body(f"obstacle_{i}").mocapid[0]) for i in range(len(indices))]
                ),
            )
        ),
        mjx_data=SimpleNamespace(
            qpos=data.qpos[None].copy(),
            qvel=data.qvel[None].copy(),
            mocap_pos=data.mocap_pos[None].copy(),
            mocap_quat=data.mocap_quat[None].copy(),
        ),
        component_identity={
            **getattr(env, "component_identity", {}),
            "visual_geometry": "MJCF scene assets and Crazyflie 2.x P250 visual meshes",
            "physics_engine": getattr(
                env,
                "physics_engine",
                "Crazyflow JAX; MuJoCo is used only for replay",
            ),
            "scenario": bank.describe(scenario_id),
        },
    )
