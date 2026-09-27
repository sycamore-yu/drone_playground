#!/usr/bin/env python3
"""ROS1-only native planner worker. JSON lines are the sole host dependency.

Run in an isolated ROS master. Each worker owns one episode and destroys its
planner/map processes at EOF, including when the simulator fails.
"""
import argparse
import base64
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PROTOCOL = sys.stdout
sys.stdout = sys.stderr

def stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()

def launch_file(method, calibration, goal, folder):
    root = ET.Element("launch")
    if method == "ego":
        source = "/tmp/p5-native/ego/src/ego-planner/src/planner/plan_manage/launch/advanced_param.xml"
        node = ET.parse(source).getroot().find("node")
        k = calibration["intrinsics"]
        args = dict(map_size_x_=40, map_size_y_=20, map_size_z_=6,
                    odometry_topic="/p5/odom", camera_pose_topic="/p5/camera_pose",
                    depth_topic="/p5/depth", cloud_topic="/p5/unused_cloud",
                    cx=k["cx_px"], cy=k["cy_px"], fx=k["fx_px"], fy=k["fy_px"],
                    max_vel=2, max_acc=3, planning_horizon=7.5, flight_type=2, point_num=1)
        for i in range(5):
            for j, axis in enumerate("xyz"):
                args["point%d_%s" % (i, axis)] = goal[j]
        for child in node:
            for attr, value in list(child.attrib.items()):
                for key, replacement in args.items():
                    value = value.replace("$(arg %s)" % key, str(replacement))
                child.set(attr, value)
        overrides = {"grid_map/pose_type": "1", "grid_map/virtual_ceil_height": "4.9",
                     "grid_map/depth_filter_maxdist": "10.0", "grid_map/max_ray_length": "10.0"}
        for child in node.findall("param"):
            if child.get("name") in overrides:
                child.set("value", overrides[child.get("name")])
        root.append(node)
        server = ET.SubElement(root, "node", pkg="ego_planner", name="traj_server",
                               type="traj_server", output="screen")
        ET.SubElement(server, "remap", **{"from": "/position_cmd", "to": "/p5/command"})
        ET.SubElement(server, "remap", **{"from": "/odom_world", "to": "/p5/odom"})
        ET.SubElement(server, "param", name="traj_server/time_forward", value="1.0")
    else:
        import yaml
        source = "/tmp/p5-native/super/src/SUPER/super_planner/config/click_smooth_ros1.yaml"
        with open(source) as handle:
            cfg = yaml.safe_load(handle)
        cfg["fsm"].update(click_goal_topic="/p5/goal", click_height=float(goal[2]),
                          cmd_topic="/p5/command", mpc_cmd_topic="/p5/polynomial")
        cfg["super_planner"]["visualization_en"] = False
        cfg["traj_opt"]["boundary"].update(max_vel=2.0, max_acc=3.0)
        cfg["rog_map"]["ros_callback"].update(cloud_topic="/p5/cloud", odom_topic="/p5/odom")
        cfg["rog_map"]["visualization"]["enable"] = False
        cfg["rog_map"]["virtual_ceil_height"] = 4.9
        cfg["rog_map"]["map_size"] = [40, 20, 6]
        target = folder / "super.yaml"
        with target.open("w") as handle:
            yaml.safe_dump(cfg, handle)
        node = ET.SubElement(root, "node", pkg="super_planner", name="fsm_node",
                             type="fsm_node", output="screen")
        ET.SubElement(node, "param", name="config_path", value=str(target))
    path = folder / "planner.launch"
    ET.ElementTree(root).write(str(path))
    return str(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ego", "super"], required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:%d" % args.port
    os.environ["ROS_IP"] = "127.0.0.1"
    os.environ.pop("ROS_HOSTNAME", None)
    os.environ.pop("ROS_NAMESPACE", None)
    # Refuse an occupied master port before setting any ROS parameters.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    core = planner = None
    with tempfile.TemporaryDirectory(prefix="p5-ros-") as temporary:
        os.environ["ROS_LOG_DIR"] = temporary
        try:
            core = subprocess.Popen(["roscore", "-p", str(args.port)],
                                    stdout=sys.stderr, stderr=sys.stderr, start_new_session=True)
            import rosgraph
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    rosgraph.Master("/p5_probe").getPid()
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                raise RuntimeError("Private ROS master failed to start")
            import rospy
            from geometry_msgs.msg import PoseStamped
            from nav_msgs.msg import Odometry
            from nav_msgs.msg import Path as PathMsg
            from quadrotor_msgs.msg import PositionCommand
            from rosgraph_msgs.msg import Clock
            from sensor_msgs.msg import Image, PointCloud2, PointField
            rospy.set_param("/use_sim_time", True)
            rospy.init_node("p5_bridge", anonymous=False, disable_signals=True)
            pubs = {
                "clock": rospy.Publisher("/clock", Clock, queue_size=1, latch=True),
                "odom": rospy.Publisher("/p5/odom", Odometry, queue_size=1),
                "pose": rospy.Publisher("/p5/camera_pose", PoseStamped, queue_size=1),
                "depth": rospy.Publisher("/p5/depth", Image, queue_size=1),
                "cloud": rospy.Publisher("/p5/cloud", PointCloud2, queue_size=1),
                "goal": rospy.Publisher("/p5/goal", PoseStamped, queue_size=1, latch=True),
                "waypoint": rospy.Publisher("/waypoint_generator/waypoints", PathMsg, queue_size=1),
            }
            latest = {"reference": None, "commands": 0, "trajectories": 0}
            def command(msg):
                latest["reference"] = {
                    "position": [msg.position.x, msg.position.y, msg.position.z],
                    "velocity": [msg.velocity.x, msg.velocity.y, msg.velocity.z],
                    "acceleration": [msg.acceleration.x, msg.acceleration.y, msg.acceleration.z],
                    "yaw": msg.yaw, "time": msg.header.stamp.to_sec() - 1.0}
                latest["commands"] += 1
            def trajectory(msg):
                if args.method == "ego" or msg.piece_num_pos > 0:
                    latest["trajectories"] += 1
            rospy.Subscriber("/p5/command", PositionCommand, command, queue_size=1)
            if args.method == "ego":
                from ego_planner.msg import Bspline
                rospy.Subscriber("/planning/bspline", Bspline, trajectory, queue_size=1)
            else:
                from quadrotor_msgs.msg import PolynomialTrajectory
                rospy.Subscriber("/p5/polynomial", PolynomialTrajectory, trajectory, queue_size=1)
            goal = None
            ticks = 0
            for line in sys.stdin:
                request = json.loads(line)
                response = {"sequence": request["sequence"]}
                if request["op"] == "start":
                    if planner is not None:
                        raise RuntimeError("One worker can own only one episode")
                    goal = request["goal"]
                    launch = launch_file(args.method, request["calibration"], goal, Path(temporary))
                    pubs["clock"].publish(Clock(rospy.Time.from_sec(1.0)))
                    planner = subprocess.Popen(["roslaunch", launch], stdout=sys.stderr,
                                               stderr=sys.stderr, start_new_session=True)
                    deadline = time.monotonic() + 20
                    sensor_pub = pubs["depth" if args.method == "ego" else "cloud"]
                    while time.monotonic() < deadline:
                        goal_ready = args.method == "ego" or pubs["goal"].get_num_connections()
                        if sensor_pub.get_num_connections() and pubs["odom"].get_num_connections() and goal_ready:
                            break
                        if planner.poll() is not None:
                            raise RuntimeError("Native planner launch failed")
                        time.sleep(0.05)
                    else:
                        raise RuntimeError("Native sensor subscribers not ready")
                    response["ready"] = True
                    response["launch_xml"] = Path(launch).read_text()
                    config_path = Path(temporary) / "super.yaml"
                    response["planner_yaml"] = config_path.read_text() if config_path.exists() else None
                elif request["op"] == "step":
                    if planner.poll() is not None:
                        raise RuntimeError("Native planner process died")
                    stamp = rospy.Time.from_sec(1.0 + request["time"])
                    odom = Odometry()
                    odom.header.stamp, odom.header.frame_id = stamp, "world"
                    odom.child_frame_id = "body"
                    for axis, value in zip("xyz", request["position"]):
                        setattr(odom.pose.pose.position, axis, value)
                    for axis, value in zip("xyzw", request["quaternion"]):
                        setattr(odom.pose.pose.orientation, axis, value)
                    for axis, value in zip("xyz", request["velocity"]):
                        setattr(odom.twist.twist.linear, axis, value)
                    pubs["odom"].publish(odom)
                    if "depth" in request:
                        pose = PoseStamped()
                        pose.header = odom.header
                        for axis, value in zip("xyz", request["camera_position"]):
                            setattr(pose.pose.position, axis, value)
                        for axis, value in zip("xyzw", request["camera_quaternion"]):
                            setattr(pose.pose.orientation, axis, value)
                        depth = Image()
                        depth.header.stamp, depth.header.frame_id = stamp, "camera_optical"
                        depth.height, depth.width = request["height"], request["width"]
                        depth.encoding, depth.step = "32FC1", depth.width * 4
                        depth.data = base64.b64decode(request["depth"])
                        pubs["pose"].publish(pose)
                        pubs["depth"].publish(depth)
                    if "points" in request:
                        cloud = PointCloud2()
                        cloud.header = odom.header
                        cloud.height, cloud.width = 1, request["point_count"]
                        cloud.fields = [PointField(n, i * 4, PointField.FLOAT32, 1)
                                        for i, n in enumerate(["x", "y", "z", "intensity"])]
                        cloud.point_step, cloud.row_step = 16, 16 * cloud.width
                        cloud.is_dense = True
                        cloud.data = base64.b64decode(request["points"])
                        pubs["cloud"].publish(cloud)
                    pubs["clock"].publish(Clock(stamp))
                    if ticks == 10:
                        target = PoseStamped()
                        target.header = odom.header
                        target.pose.orientation.w = 1.0
                        for axis, value in zip("xyz", goal):
                            setattr(target.pose.position, axis, value)
                        if args.method == "ego":
                            path = PathMsg()
                            path.header, path.poses = odom.header, [target]
                            pubs["waypoint"].publish(path)
                        else:
                            pubs["goal"].publish(target)
                    ticks += 1
                    # Wall time for asynchronous native callbacks; simulation time is /clock.
                    time.sleep(0.01)
                    response.update(latest)
                elif request["op"] == "close":
                    response.update(latest)
                else:
                    raise ValueError("Unknown bridge request")
                PROTOCOL.write(json.dumps(response, allow_nan=False) + "\n")
                PROTOCOL.flush()
                if request["op"] == "close":
                    break
        finally:
            stop(planner)
            stop(core)

if __name__ == "__main__":
    main()
