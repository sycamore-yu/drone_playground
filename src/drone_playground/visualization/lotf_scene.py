"""A replay-only MuJoCo model from the actual LOTF mass and motor coordinates."""

from types import SimpleNamespace

import mujoco
import numpy as np


def create_replay_model(task):
    native = task.model.native
    motors = np.asarray([native._tbm_fr, native._tbm_bl, native._tbm_br, native._tbm_fl])
    inertia = " ".join(str(float(x)) for x in native._inertia)
    geoms = []
    for i, p in enumerate(motors):
        x, y, z = map(float, p)
        geoms.append(
            f'<geom name="arm{i}" type="capsule" fromto="0 0 0 {x} {y} {z}" size="0.006" rgba="0.18 0.18 0.20 1"/>'
        )
        geoms.append(
            f'<geom name="rotor{i}" type="cylinder" pos="{x} {y} {z + 0.015}" size="0.027 0.0015" rgba="0.2 0.5 0.8 0.7"/>'
        )
    xml = f'''<mujoco model="LOTF-example-quad-replay">
      <compiler angle="radian"/>
      <option timestep="{task.dt}" gravity="0 0 -9.81"/>
      <visual><global offwidth="1200" offheight="800"/></visual>
      <worldbody><light pos="0 0 5"/>
        <geom name="ground" type="plane" size="5 5 .1" rgba=".9 .9 .9 1"/>
        <body name="lotf_quad" mocap="true" pos="0 0 1.5">
          <inertial pos="0 0 0" mass="{native._mass}" diaginertia="{inertia}"/>
          <geom name="center" type="box" size=".025 .018 .008" rgba=".25 .3 .35 1"/>
          {"".join(geoms)}
        </body>
      </worldbody></mujoco>'''
    spec = mujoco.MjSpec.from_string(xml)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return SimpleNamespace(
        spec=spec,
        mj_model=model,
        data=SimpleNamespace(core=SimpleNamespace(drone_mocap_ids=np.array([0]))),
        mjx_data=SimpleNamespace(
            qpos=data.qpos[None].copy(),
            qvel=data.qvel[None].copy(),
            mocap_pos=data.mocap_pos[None].copy(),
            mocap_quat=data.mocap_quat[None].copy(),
        ),
        component_identity={
            **task.component_identity,
            "visual_geometry": "schematic; native mass/inertia/motor coordinates",
            "physics_engine": "LOTF JAX, MuJoCo is used only for replay",
        },
    )
