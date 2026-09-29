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

from drone_playground.environments.scenes.navigation import (
    KIND_CAPSULE,
    KIND_CYLINDER,
    KIND_SPHERE,
    MOTION_NAMES,
    MOTION_STATIC,
    SceneBank,
    obstacle_positions,
)

DRONE_MOTOR_RADIUS_M = 0.0325
ROTOR_RADIUS_M = 0.02355


def active_indices(bank: SceneBank, scenario_id: int) -> np.ndarray:
    """Capacity indices that carry geometry for one instance, in replay order."""
    return np.flatnonzero(np.asarray(bank.active[scenario_id]))


def instance_obstacles(bank: SceneBank, scenario_id: int) -> list[dict]:
    """Host-side obstacle list for one instance, used by the replay model."""
    obstacles = []
    for index in active_indices(bank, scenario_id):
        kind = int(np.asarray(bank.kind[scenario_id, index]))
        size = np.asarray(bank.size[scenario_id, index], float)
        origin = np.asarray(bank.origin[scenario_id, index], float)
        motion = int(np.asarray(bank.motion[scenario_id, index]))
        params = np.asarray(bank.params[scenario_id, index], float)
        obstacles.append(
            {
                "kind": kind,
                "size": size,
                "origin": origin,
                "motion": motion,
                "params": params,
                "motion_name": MOTION_NAMES[motion],
                "rotation": None
                if bank.rotations is None
                else np.asarray(bank.rotations[scenario_id, index]),
            }
        )
    return obstacles


def obstacle_track(bank: SceneBank, scenario_id: int, times: np.ndarray) -> np.ndarray:
    """Obstacle centres over time with shape ``[T, capacity, 3]``."""
    positions = jax.vmap(lambda time: obstacle_positions(bank, jnp.int32(scenario_id), time))(
        jnp.asarray(np.asarray(times, np.float32))
    )
    return np.asarray(positions)


def _obstacle_xml(obstacles: list[dict]) -> str:
    bodies = []
    for index, obstacle in enumerate(obstacles):
        x, y, z = (float(value) for value in obstacle["origin"])
        rgba = "0.35 0.42 0.5 1" if obstacle["motion"] == MOTION_STATIC else "0.95 0.45 0.12 1"
        if obstacle["kind"] == KIND_CAPSULE:
            radius, height = float(obstacle["size"][0]), float(obstacle["size"][1])
            geometry = f'<geom name="obstacle_geom_{index}" type="capsule" size="{radius} {height / 2.0}" rgba="{rgba}"/>'
        elif obstacle["kind"] == KIND_CYLINDER:
            radius, height = float(obstacle["size"][0]), float(obstacle["size"][1])
            geometry = f'<geom name="obstacle_geom_{index}" type="cylinder" size="{radius} {height / 2.0}" rgba="{rgba}"/>'
        elif obstacle["kind"] == KIND_SPHERE:
            radius = float(obstacle["size"][0])
            geometry = (
                f'<geom name="obstacle_geom_{index}" type="sphere" size="{radius}" rgba="{rgba}"/>'
            )
        else:
            hx, hy, hz = (float(value) for value in obstacle["size"])
            geometry = f'<geom name="obstacle_geom_{index}" type="box" size="{hx} {hy} {hz}" rgba="{rgba}"/>'
        if obstacle.get("rotation") is not None:
            from scipy.spatial.transform import Rotation

            xyzw = Rotation.from_matrix(obstacle["rotation"]).as_quat()
            quaternion = " ".join(str(float(value)) for value in xyzw[[3, 0, 1, 2]])
            geometry = geometry.replace("<geom ", f'<geom quat="{quaternion}" ', 1)
        bodies.append(
            f'<body name="obstacle_{index}" mocap="true" pos="{x} {y} {z}">{geometry}</body>'
        )
    return "".join(bodies)


def _drone_xml() -> str:
    geoms = [
        '<geom name="body_sphere" type="sphere" size="0.07" rgba="0.85 0.25 0.25 0.55"/>',
        '<geom name="board" type="box" size="0.03 0.03 0.006" rgba="0.2 0.25 0.3 1"/>',
    ]
    for index, (x, y) in enumerate(((1, -1), (-1, -1), (-1, 1), (1, 1))):
        px, py = x * DRONE_MOTOR_RADIUS_M, y * DRONE_MOTOR_RADIUS_M
        geoms.append(
            f'<geom name="rotor{index}" type="cylinder" pos="{px} {py} 0.012" '
            f'size="{ROTOR_RADIUS_M} 0.0015" rgba="0.2 0.5 0.8 0.75"/>'
        )
    return (
        '<body name="drone" mocap="true" pos="0 0 2">'
        '  <geom name="drone_collision" type="sphere" size="0.07" '
        '        pos="0 0 0.005" rgba="0.9 0.4 0.2 0.25"/>'
        f"  {''.join(geoms)}"
        "</body>"
    )


def create_replay_model(env, scenario_id: int):
    """MuJoCo replay model for one navigation instance of a built environment."""
    bank = env.bank
    obstacles = instance_obstacles(bank, scenario_id)
    corridor_low = np.asarray(bank.world_low, float)
    corridor_high = np.asarray(bank.world_high, float)
    centre = (corridor_low + corridor_high) / 2.0
    half = (corridor_high - corridor_low) / 2.0
    start = np.asarray(bank.start[scenario_id], float)
    goal = np.asarray(bank.goal[scenario_id], float)
    dt = env.dt
    xml = f'''<mujoco model="navigation-replay">
  <compiler angle="radian"/>
  <option timestep="{dt}" gravity="0 0 -9.81"/>
  <visual><global offwidth="1600" offheight="1000"/></visual>
  <worldbody>
    <light pos="8 0 9" dir="0 0 -1"/>
    <geom name="ground" type="plane" pos="0 0 {corridor_low[2]}" size="{half[0]} {half[1]} .1" rgba=".86 .88 .9 1"/>
    <geom name="floor_marker" type="box" pos="{centre[0]} {centre[1]} {corridor_low[2] - 0.02}" size="{half[0]} {half[1]} 0.02" rgba=".78 .82 .86 1"/>
    <geom name="start_marker" type="sphere" pos="{start[0]} {start[1]} {start[2]}" size="0.12" rgba="0.2 0.8 0.3 0.8"/>
    <geom name="goal_marker" type="sphere" pos="{goal[0]} {goal[1]} {goal[2]}" size="0.5" rgba="0.9 0.8 0.2 0.35"/>
    {_drone_xml()}
    {_obstacle_xml(obstacles)}
  </worldbody>
</mujoco>'''
    spec = mujoco.MjSpec.from_string(xml)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(
        spec=spec,
        mj_model=model,
        data=SimpleNamespace(
            core=SimpleNamespace(
                drone_mocap_ids=np.array([0]),
                obstacle_mocap_ids=np.arange(1, 1 + len(obstacles)),
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
            "visual_geometry": "analytic scene primitives plus a schematic quadrotor",
            "physics_engine": getattr(
                env, "physics_engine", "Crazyflow JAX; MuJoCo is used only for replay"
            ),
            "scenario": env.scenario(scenario_id),
        },
    )
