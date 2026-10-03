"""Synthetic scene builders shared by sensor and navigation tests."""

import jax.numpy as jnp
import mujoco
import numpy as np

from drone_playground.environments.scenes.geometry import (
    KIND_CYLINDER,
    MOTION_STATIC,
    SceneBank,
)

WORLD_LOW = np.array([0.0, -5.0, 0.0], np.float32)
WORLD_HIGH = np.array([20.0, 5.0, 5.0], np.float32)


def synthetic_bank(
    obstacles,
    capacity=4,
    start=(0.5, 0.0, 2.0),
    goal=(15.5, 0.0, 2.0),
):
    kind = np.zeros((1, capacity), np.int32)
    size = np.zeros((1, capacity, 3), np.float32)
    origin = np.zeros((1, capacity, 3), np.float32)
    motion = np.zeros((1, capacity), np.int32)
    params = np.zeros((1, capacity, 5), np.float32)
    active = np.zeros((1, capacity), bool)
    for index, obstacle in enumerate(obstacles):
        kind[0, index] = obstacle["kind"]
        size[0, index] = obstacle["size"]
        origin[0, index] = obstacle["origin"]
        motion[0, index] = obstacle.get("motion", MOTION_STATIC)
        params[0, index] = obstacle.get("params", (0.0,) * 5)
        active[0, index] = True
    return SceneBank(
        kind=jnp.asarray(kind),
        size=jnp.asarray(size),
        origin=jnp.asarray(origin),
        motion=jnp.asarray(motion),
        params=jnp.asarray(params),
        active=jnp.asarray(active),
        start=jnp.asarray([start], jnp.float32),
        goal=jnp.asarray([goal], jnp.float32),
        difficulty=jnp.zeros((1,), jnp.int32),
        subtype=jnp.zeros((1,), jnp.int32),
        world_low=jnp.asarray(WORLD_LOW),
        world_high=jnp.asarray(WORLD_HIGH),
        subtype_names=("synthetic",),
    )


def mujoco_scene(obstacles):
    bodies = []
    for obstacle in obstacles:
        x, y, z = obstacle["origin"]
        if obstacle["kind"] == KIND_CYLINDER:
            radius, height = obstacle["size"][0], obstacle["size"][1]
            geometry = f'<geom type="cylinder" size="{radius} {height / 2.0}"/>'
        else:
            hx, hy, hz = obstacle["size"]
            geometry = f'<geom type="box" size="{hx} {hy} {hz}"/>'
        bodies.append(f'<body pos="{x} {y} {z}">{geometry}</body>')
    xml = (
        '<mujoco model="sensor-parity"><compiler angle="radian"/>'
        '<option timestep="0.002"/><worldbody>'
        '<geom name="ground" type="plane" size="110 105 .1" pos="10 0 0"/>'
        '<body name="mjx_dummy" pos="500 500 500"><freejoint/>'
        '<geom name="mjx_dummy_geom" type="sphere" size="0.01" mass="0.001"/>'
        "</body>" + "".join(bodies) + "</worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def synthetic_navigation_env(obstacles, duration=2.0, **kwargs):
    from dataclasses import asdict
    from types import SimpleNamespace

    from drone_playground.configuration import load_config
    from drone_playground.environments.environment import build_environment

    config = load_config("environment", ["env=navigation/static"])
    config["env"]["sensor"] = None
    config["env"]["task"]["observation"] = {
        "_target_": "drone_playground.environments.observations.state.NavigationObservation"
    }
    config["env"]["dynamics"].update(
        forward=kwargs.pop("dynamics", "first_principles"), drone=kwargs.pop("drone", "cf2x_L250")
    )
    config["env"]["task"].update(duration=duration, freq=kwargs.pop("freq", 50))
    collision = kwargs.pop("training_collision_mode", "terminate")
    objective = kwargs.pop("objective", None)
    if objective is not None:
        config["env"]["task"]["reward"] = {
            "_target_": type(objective).__module__ + "." + type(objective).__name__,
            **asdict(objective),
        }
    config["env"]["task"].update(kwargs)
    config["training"] = {"navigation_collision_mode": collision}
    bank = synthetic_bank(obstacles)
    scene = SimpleNamespace(build=lambda: (bank, {"source": "synthetic test scene"}))
    return build_environment(config, "cpu", role="train", scene=scene)


def place(state, env, position, velocity=(0.0, 0.0, 0.0)):
    data = state.pipeline_state
    states = data.sim_data.states.replace(
        pos=jnp.asarray(position, jnp.float32)[None, None],
        vel=jnp.asarray(velocity, jnp.float32)[None, None],
        quat=jnp.broadcast_to(env.identity_quat, data.sim_data.states.quat.shape),
        ang_vel=jnp.zeros_like(data.sim_data.states.ang_vel),
    )
    return state.replace(pipeline_state=data.replace(sim_data=data.sim_data.replace(states=states)))
