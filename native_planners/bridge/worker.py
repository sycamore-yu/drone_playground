#!/usr/bin/env python3
"""ROS1 adapter for the shared native gRPC service; owns its planner and map."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from google.protobuf.json_format import MessageToDict

from drone_playground.native.geometry import PlannerGeometry
from drone_playground.native.proto import algorithm_pb2 as pb
from drone_playground.native.ros_trajectory import ego_trajectory, super_trajectory
from drone_playground.native.ros_visualization import corridor_from_markers
from drone_playground.native.server import serve
from drone_playground.native.waypoints import OrderedWaypointGoals
from drone_playground.native.wire import decode_output, encode_geometry, encode_output, vec3


def stop(process):
    if process is None or process.poll() is not None:
        return
    # roslaunch starts children in separate process groups. Killing only its own
    # group can orphan a busy planner; retain each descendant's PID/start identity.
    records = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                records[int(entry.name)] = (int(fields[1]), fields[19])
            except (OSError, IndexError, ValueError):
                pass
    owned = {process.pid}
    while True:
        found = {pid for pid, (parent, _) in records.items() if parent in owned}
        if found <= owned:
            break
        owned |= found

    def send(sig):
        for pid in sorted(owned, reverse=True):
            try:
                fields = Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()
                if pid in records and fields[19] == records[pid][1]:
                    os.kill(pid, sig)
            except (ProcessLookupError, FileNotFoundError):
                pass

    send(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    send(signal.SIGKILL)
    process.wait()


def native_runtime_identity(method):
    root = Path("/opt/drone_playground/planners")
    if method == "ego":
        paths = [
            root / "ego/devel/lib/ego_planner/ego_planner_node",
            root / "ego/src/ego-planner/src/planner/plan_manage/src/ego_replan_fsm.cpp",
        ]
    else:
        paths = [
            root / "super/devel/lib/super_planner/fsm_node",
            root / "super/src/SUPER/super_planner/src/traj_opt/exp_traj_optimizer_s4.cpp",
            root / "super/src/SUPER/super_planner/src/super_core/super_planner.cpp",
        ]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def launch_file(method, calibration, goal, folder, limits=None, task_adapter=None, parameters=None):
    inflation = (parameters or {}).get('occupancy_inflation_m')
    if inflation is not None and (method != 'ego' or isinstance(inflation, bool)
                                  or not isinstance(inflation, (int, float))
                                  or not math.isfinite(inflation) or inflation <= 0):
        raise ValueError('occupancy_inflation_m must be finite and positive, for EGO only')
    limits = limits or {
        "max_velocity_mps": 20.0,
        "max_acceleration_mps2": 3.0,
        "planning_horizon_m": 7.5,
    }
    root = ET.Element("launch")
    interactive = bool((task_adapter or {}).get("interactive_goals", False))
    bounds = task_adapter or {}
    low, high = bounds.get("world_low"), bounds.get("world_high")
    ceiling = float(high[2]) - 0.1 if high else 4.9
    if method == "ego":
        source = "/opt/drone_playground/planners/ego/src/ego-planner/src/planner/plan_manage/launch/advanced_param.xml"
        node = ET.parse(source).getroot().find("node")
        k = calibration["intrinsics"]
        args = dict(
            # EGO's map is fixed at (-size_x/2, -size_y/2, ground), unlike
            # SUPER's rolling map. Cover the declared world in that native frame.
            map_size_x_=2 * max(abs(low[0]), abs(high[0])) + 2 if low and high else 40,
            map_size_y_=2 * max(abs(low[1]), abs(high[1])) + 2 if low and high else 20,
            map_size_z_=max(6.0, high[2] + 0.5) if high else 6,
            odometry_topic="/p5/odom",
            camera_pose_topic="/p5/camera_pose",
            depth_topic="/p5/depth",
            cloud_topic="/p5/unused_cloud",
            cx=k["cx_px"],
            cy=k["cy_px"],
            fx=k["fx_px"],
            fy=k["fy_px"],
            max_vel=limits["max_velocity_mps"],
            max_acc=limits["max_acceleration_mps2"],
            planning_horizon=limits["planning_horizon_m"],
            flight_type=1 if interactive else 2,
            point_num=1,
        )
        for i in range(5):
            for j, axis in enumerate("xyz"):
                args["point%d_%s" % (i, axis)] = goal[j]
        for child in node:
            for attr, value in list(child.attrib.items()):
                for key, replacement in args.items():
                    value = value.replace("$(arg %s)" % key, str(replacement))
                child.set(attr, value)
        overrides = {
            "grid_map/pose_type": "1",
            "grid_map/virtual_ceil_height": str(ceiling),
            "grid_map/depth_filter_maxdist": "10.0",
            "grid_map/max_ray_length": "10.0",
        }
        if inflation is not None:
            overrides['grid_map/obstacles_inflation'] = str(inflation)
        for child in node.findall("param"):
            if child.get("name") in overrides:
                child.set("value", overrides[child.get("name")])
        root.append(node)
        server = ET.SubElement(
            root, "node", pkg="ego_planner", name="traj_server", type="traj_server", output="screen"
        )
        ET.SubElement(server, "remap", **{"from": "/position_cmd", "to": "/p5/command"})
        ET.SubElement(server, "remap", **{"from": "/odom_world", "to": "/p5/odom"})
        ET.SubElement(server, "param", name="traj_server/time_forward", value="1.0")
    else:
        import yaml

        source = "/opt/drone_playground/planners/super/src/SUPER/super_planner/config/click_smooth_ros1.yaml"
        with open(source) as handle:
            cfg = yaml.safe_load(handle)
        cfg["fsm"].update(
            click_goal_topic="/p5/goal",
            click_height=-10.0 if interactive else float(goal[2]),
            cmd_topic="/p5/command",
            mpc_cmd_topic="/p5/polynomial",
        )
        cfg["super_planner"]["visualization_en"] = bool(bounds.get("record_planner_visualization", False))
        cfg["traj_opt"]["boundary"].update(
            max_vel=limits["max_velocity_mps"], max_acc=limits["max_acceleration_mps2"]
        )
        cfg["rog_map"]["ros_callback"].update(cloud_topic="/p5/cloud", odom_topic="/p5/odom")
        cfg["rog_map"]["visualization"]["enable"] = False
        cfg["rog_map"]["virtual_ceil_height"] = ceiling
        cfg["rog_map"]["map_size"] = [40, 20, 6]
        target = folder / "super.yaml"
        with target.open("w") as handle:
            yaml.safe_dump(cfg, handle)
        node = ET.SubElement(
            root, "node", pkg="super_planner", name="fsm_node", type="fsm_node", output="screen"
        )
        ET.SubElement(
            node, "env", name="DRONE_PLAYGROUND_CONTROL_TRANSFER", value="1" if interactive else "0"
        )
        ET.SubElement(node, "param", name="config_path", value=str(target))
    path = folder / "planner.launch"
    ET.ElementTree(root).write(str(path))
    return str(path)


def check_master_port(port):
    # ROS uses SO_REUSEADDR too. A preceding, fully closed episode can leave
    # accepted TCP connections in TIME_WAIT; those are not another live owner.
    # Binding still fails when an active master is listening on this address.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


class RosAlgorithm:
    def __init__(self, method, port):
        self.method = method
        self.core = self.planner = None
        self.temporary = tempfile.TemporaryDirectory(prefix="drone-ros-")
        self.folder = Path(self.temporary.name)
        self.lock = threading.Lock()
        self.subscribers = []
        self.epoch = 0
        os.environ.update(
            ROS_MASTER_URI="http://127.0.0.1:%d" % port,
            ROS_IP="127.0.0.1",
            ROS_LOG_DIR=str(self.folder),
        )
        os.environ.pop("ROS_HOSTNAME", None)
        os.environ.pop("ROS_NAMESPACE", None)
        check_master_port(port)
        try:
            self.core = subprocess.Popen(
                ["roscore", "-p", str(port)],
                stdout=sys.stderr,
                stderr=sys.stderr,
                start_new_session=True,
            )
            import rosgraph

            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    rosgraph.Master("/drone_probe").getPid()
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                raise RuntimeError("Private ROS master failed to start")
            import rospy

            rospy.set_param("/use_sim_time", True)
            rospy.init_node("drone_bridge", anonymous=False, disable_signals=True)
        except BaseException:
            self.close()
            raise

    def initialize(self, request):
        if request.algorithm != self.method:
            raise ValueError("This adapter does not implement the requested algorithm")
        self.parameters = MessageToDict(request.parameters)
        self.upstream_waypoints = self.parameters.get("use_upstream_waypoints", False)
        if not isinstance(self.upstream_waypoints, bool):
            raise ValueError("use_upstream_waypoints must be a boolean")
        return pb.InitializeResponse(
            capabilities=pb.Capabilities(
                algorithm=self.method,
                required_inputs=["state", "upstream"] if self.upstream_waypoints else ["state"],
                accepted_upstream=["waypoint"] if self.upstream_waypoints else [],
                outputs=["trajectory"],
                derivatives="none",
            ),
            provenance={
                "runtime_sha256": json.dumps(native_runtime_identity(self.method)),
                "adapter": "ros1-grpc-v1",
                "clock_origin": "ROS = simulation + 1s",
                "goal_source": "ordered upstream waypoints" if self.upstream_waypoints else "task goal",
            },
        )

    def reset(self, request):
        import rospy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from nav_msgs.msg import Path as PathMsg
        from quadrotor_msgs.msg import PositionCommand
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image, PointCloud2

        stop(self.planner)
        self.planner = None
        self.epoch += 1
        epoch = self.epoch
        for subscriber in self.subscribers:
            subscriber.unregister()
        for pub in getattr(self, "pubs", {}).values():
            pub.unregister()
        self.pubs = {
            name: rospy.Publisher(topic, cls, queue_size=1, latch=latch)
            for name, topic, cls, latch in (
                ("clock", "/clock", Clock, True),
                ("odom", "/p5/odom", Odometry, False),
                ("pose", "/p5/camera_pose", PoseStamped, False),
                ("depth", "/p5/depth", Image, False),
                ("cloud", "/p5/cloud", PointCloud2, False),
                ("goal", "/p5/goal", PoseStamped, True),
                ("waypoint", "/waypoint_generator/waypoints", PathMsg, False),
            )
        }
        self.latest = dict(commands=0, trajectories=0, reference=None, trajectory=None, error=None,
                           corridors={})
        self.ticks = 0
        if not request.goal.positions:
            raise ValueError("ROS navigation requires a goal")
        self.goal = request.goal.positions[-1]
        self.route = OrderedWaypointGoals()

        def command(msg):
            with self.lock:
                if self.epoch != epoch:
                    return
                self.latest["reference"] = pb.TrackingReference(
                    time=msg.header.stamp.to_sec() - 1.0,
                    position=vec3([msg.position.x, msg.position.y, msg.position.z]),
                    velocity=vec3([msg.velocity.x, msg.velocity.y, msg.velocity.z]),
                    acceleration=vec3([msg.acceleration.x, msg.acceleration.y, msg.acceleration.z]),
                    yaw=msg.yaw,
                )
                self.latest["commands"] += 1

        def trajectory(msg):
            with self.lock:
                if self.epoch != epoch:
                    return
                if self.method == "super" and msg.type & msg.EMER_STOP:
                    self.latest["trajectory"] = None
                    return
                if self.method == "super" and not msg.piece_num_pos:
                    return  # Heartbeat is not a new trajectory.
                try:
                    curve = ego_trajectory(msg) if self.method == "ego" else super_trajectory(msg)
                    self.latest.update(trajectory=curve, generated=rospy.Time.now().to_sec() - 1.0)
                    self.latest["trajectories"] += 1
                except Exception as exc:
                    self.latest["error"] = repr(exc)

        self.subscribers = [rospy.Subscriber("/p5/command", PositionCommand, command, queue_size=1)]
        if self.method == "ego":
            from ego_planner.msg import Bspline

            self.subscribers.append(
                rospy.Subscriber("/planning/bspline", Bspline, trajectory, queue_size=1)
            )
        else:
            from quadrotor_msgs.msg import PolynomialTrajectory

            self.subscribers.append(
                rospy.Subscriber("/p5/polynomial", PolynomialTrajectory, trajectory, queue_size=1)
            )
        task = MessageToDict(request.task)
        if self.method == "super" and task.get("record_planner_visualization", False):
            from visualization_msgs.msg import MarkerArray

            def corridor_callback(role):
                def callback(msg):
                    with self.lock:
                        if self.epoch != epoch:
                            return
                        try:
                            now = max(0., rospy.Time.now().to_sec() - 1.)
                            value = corridor_from_markers(msg.markers, role)
                            self.latest['corridors'][role] = (now, value)
                        except Exception as exc:
                            self.latest['error'] = repr(exc)
                return callback

            for topic, role in [('exp_sfc','candidate'),('backup_sfc','backup')]:
                self.subscribers.append(rospy.Subscriber(
                    '/fsm_node/visualization/'+topic, MarkerArray, corridor_callback(role), queue_size=2))
        if self.upstream_waypoints:
            task["interactive_goals"] = True
        launch = launch_file(
            self.method,
            MessageToDict(request.calibration),
            [self.goal.x, self.goal.y, self.goal.z],
            self.folder,
            self.parameters.get("limits"),
            task,
            self.parameters,
        )
        self.pubs["clock"].publish(Clock(rospy.Time.from_sec(1.0 + request.header.simulation_time)))
        self.planner = subprocess.Popen(
            ["roslaunch", launch], stdout=sys.stderr, stderr=sys.stderr, start_new_session=True
        )
        deadline = time.monotonic() + 90
        sensor = self.pubs["depth" if self.method == "ego" else "cloud"]
        while time.monotonic() < deadline:
            if (
                sensor.get_num_connections()
                and self.pubs["odom"].get_num_connections()
                and (self.method == "ego" or self.pubs["goal"].get_num_connections())
            ):
                break
            if self.planner.poll() is not None:
                raise RuntimeError("Native planner launch failed")
            time.sleep(0.05)
        else:
            raise RuntimeError("Native sensor subscribers not ready")
        cfg = self.folder / "super.yaml"
        return dict(
            launch_xml=Path(launch).read_text(),
            planner_yaml=cfg.read_text() if cfg.exists() else "",
            runtime_sha256=json.dumps(native_runtime_identity(self.method)),
        )

    def step(self, request):
        import rospy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from nav_msgs.msg import Path as PathMsg
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image, PointCloud2, PointField

        if self.planner is None or self.planner.poll() is not None:
            raise RuntimeError("Native planner is not running")
        stamp = rospy.Time.from_sec(1.0 + request.header.simulation_time)
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = stamp, "world"
        odom.child_frame_id = "body"
        for axis in "xyz":
            setattr(odom.pose.pose.position, axis, getattr(request.state.position, axis))
            setattr(odom.twist.twist.linear, axis, getattr(request.state.velocity, axis))
        for axis, value in zip("xyzw", request.state.quaternion_xyzw):
            setattr(odom.pose.pose.orientation, axis, value)
        self.pubs["odom"].publish(odom)
        measurement = request.measurement
        kind = measurement.WhichOneof("data")
        if kind == "depth":
            pose = PoseStamped()
            pose.header = odom.header
            pose.header.stamp = rospy.Time.from_sec(1.0 + measurement.time)
            d = measurement.depth
            for axis in "xyz":
                setattr(pose.pose.position, axis, getattr(d.camera_position, axis))
            for axis, value in zip("xyzw", d.camera_quaternion_xyzw):
                setattr(pose.pose.orientation, axis, value)
            depth = Image()
            depth.header.stamp, depth.header.frame_id = pose.header.stamp, "camera_optical"
            depth.height, depth.width = d.height, d.width
            depth.encoding, depth.step, depth.data = "32FC1", d.width * 4, d.float32_le
            self.pubs["pose"].publish(pose)
            self.pubs["depth"].publish(depth)
        elif kind == "point_cloud":
            if measurement.frame != "world":
                raise ValueError("SUPER ROS adapter expects world point cloud")
            c = measurement.point_cloud
            cloud = PointCloud2()
            cloud.header = odom.header
            cloud.header.stamp = rospy.Time.from_sec(1.0 + measurement.time)
            cloud.height, cloud.width = 1, c.count
            cloud.fields = [
                PointField(n, i * 4, PointField.FLOAT32, 1)
                for i, n in enumerate(["x", "y", "z", "intensity"][: c.channels])
            ]
            cloud.point_step, cloud.row_step = 4 * c.channels, 4 * c.channels * c.count
            cloud.is_dense, cloud.data = True, c.float32_le
            self.pubs["cloud"].publish(cloud)
        self.pubs["clock"].publish(Clock(stamp))
        if self.upstream_waypoints:
            if request.WhichOneof("upstream") != "waypoints":
                raise ValueError("ROS waypoint input mode requires upstream Waypoint")
            target, goal_changed = self.route.update(
                decode_output(request.waypoints),
                [request.state.position.x, request.state.position.y, request.state.position.z],
            )
            self.goal = vec3(target)
        elif request.HasField("goal"):
            self.goal = request.goal.positions[-1]
            goal_changed = True
        else:
            goal_changed = False
        if self.ticks == 10 or (self.ticks > 10 and goal_changed):
            target = PoseStamped()
            target.header, target.pose.orientation.w = odom.header, 1.0
            for axis in "xyz":
                setattr(target.pose.position, axis, getattr(self.goal, axis))
            if self.method == "ego":
                path = PathMsg()
                path.header, path.poses = odom.header, [target]
                self.pubs["waypoint"].publish(path)
            else:
                self.pubs["goal"].publish(target)
        self.ticks += 1
        time.sleep(min(0.01, request.solve_budget_seconds))
        with self.lock:
            latest = self.latest.copy()
            corridors = list(self.latest['corridors'].values())
        if latest["error"]:
            raise ValueError(latest["error"])
        response = pb.StepResponse(
            diagnostics={key: latest[key] for key in ("commands", "trajectories")}
        )
        if self.upstream_waypoints:
            response.diagnostics["waypoint_index"] = self.route.index
        curve, now = latest["trajectory"], request.header.simulation_time
        # Marker topics inspect asynchronous candidate/backup work, not a proof
        # that this corridor belongs to the committed executable polynomial.
        current = [(stamp, value) for stamp, value in corridors if stamp <= now <= stamp + .5]
        if current:
            geometry = PlannerGeometry(max(stamp for stamp, _ in current),
                min(stamp+.5 for stamp, _ in current), tuple(value for _, value in current))
            response.planner_geometry.CopyFrom(encode_geometry(geometry))
        if curve is not None and curve.start_time <= now <= curve.end_time:
            response.decision.CopyFrom(
                pb.Decision(
                    status=pb.VALID,
                    plan_id=str(latest["trajectories"]),
                    generated_at=latest["generated"],
                    valid_until=curve.end_time,
                    trajectory=encode_output(curve),
                )
            )
        else:
            response.decision.status = pb.NO_PLAN
            response.decision.explanation = "No trajectory covering the current simulation time"
        if latest["reference"] is not None:
            response.sampled_reference.CopyFrom(latest["reference"])
        return response

    def close(self):
        stop(self.planner)
        stop(self.core)
        self.planner = self.core = None
        self.temporary.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["ego", "super"], required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--address", required=True)
    args = parser.parse_args()
    serve(RosAlgorithm(args.method, args.port), args.address)


if __name__ == "__main__":
    main()
