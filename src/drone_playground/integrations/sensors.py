"""Physical sensor packets shared by native tracking and navigation execution."""

import base64

import numpy as np
from scipy.spatial.transform import Rotation


def pack_array(value):
    """Pack a numeric sensor array for external method transport."""
    return base64.b64encode(np.asarray(value, dtype="<f4").tobytes()).decode("ascii")


def sensor_packet(env, method, sample, state):
    """Build a timestamped sensor observation packet for the selected method."""
    import jax

    data = state.pipeline_state
    body = env.controller_observation(state)
    packet = {
        "time": int(data.step_index) * env.dt,
        "policy_observation": np.asarray(state.obs).tolist(),
        "position": body["pos"].tolist(),
        "quaternion": body["quat"].tolist(),
        "velocity": body["vel"].tolist(),
        "angular_velocity": body["ang_vel"].tolist(),
    }
    if method == "none" or int(data.step_index) % env.sensor_period:
        return packet
    if method in ("ego", "depth"):
        depth, pos, rotation = jax.tree.map(np.asarray, sample(data))
        # Ray grid is width-major; ROS images are row-major.
        depth = depth.reshape(env.sensor.width, env.sensor.height).T
        packet.update(
            depth=pack_array(depth),
            width=env.sensor.width,
            height=env.sensor.height,
            camera_position=pos.tolist(),
            camera_quaternion=Rotation.from_matrix(rotation).as_quat().tolist(),
        )
    else:
        frame = jax.tree.map(np.asarray, sample(data))
        points = frame.points_world[frame.valid]
        points = np.c_[points, np.ones(len(points), dtype=np.float32)]
        packet.update(points=pack_array(points), point_count=len(points))
    return packet


def packet_measurement(packet):
    """Convert one captured sensor packet to the shared protobuf measurement."""
    import base64

    from drone_playground.integrations.rpc.proto import algorithm_pb2 as pb
    from drone_playground.integrations.rpc.wire import vec3

    if "depth" in packet:
        return pb.Measurement(
            time=packet["time"],
            frame="camera_optical",
            depth=pb.DepthImage(
                float32_le=base64.b64decode(packet["depth"], validate=True),
                width=packet["width"],
                height=packet["height"],
                camera_position=vec3(packet["camera_position"]),
                camera_quaternion_xyzw=packet["camera_quaternion"],
            ),
        )
    if "points" in packet:
        return pb.Measurement(
            time=packet["time"],
            frame="world",
            point_cloud=pb.PointCloud(
                float32_le=base64.b64decode(packet["points"], validate=True),
                count=packet["point_count"],
                channels=4,
            ),
        )
    return None
