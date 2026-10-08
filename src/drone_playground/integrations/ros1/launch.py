"""Translate pinned planner settings into run-owned ROS1 launch files."""

import math
import xml.etree.ElementTree as ET


def launch_file(
    method,
    calibration,
    goal,
    folder,
    limits=None,
    task_adapter=None,
    parameters=None,
):
    """Render the ROS1 launch configuration for an external planning method."""
    inflation = (parameters or {}).get("occupancy_inflation_m")
    if inflation is not None and (
        method != "ego"
        or isinstance(inflation, bool)
        or not isinstance(inflation, (int, float))
        or not math.isfinite(inflation)
        or inflation <= 0
    ):
        raise ValueError("occupancy_inflation_m must be finite and positive, for EGO only")
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
        source = (
            "/opt/drone_playground/planners/ego/src/ego-planner/"
            "src/planner/plan_manage/launch/advanced_param.xml"
        )
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
                args[f"point{i}_{axis}"] = goal[j]
        for child in node:
            for attr, value in list(child.attrib.items()):
                for key, replacement in args.items():
                    value = value.replace(f"$(arg {key})", str(replacement))
                child.set(attr, value)
        overrides = {
            "grid_map/pose_type": "1",
            "grid_map/virtual_ceil_height": str(ceiling),
            "grid_map/depth_filter_maxdist": "10.0",
            "grid_map/max_ray_length": "10.0",
        }
        if inflation is not None:
            overrides["grid_map/obstacles_inflation"] = str(inflation)
        for child in node.findall("param"):
            if child.get("name") in overrides:
                child.set("value", overrides[child.get("name")])
        root.append(node)
        server = ET.SubElement(
            root,
            "node",
            pkg="ego_planner",
            name="traj_server",
            type="traj_server",
            output="screen",
        )
        ET.SubElement(server, "remap", **{"from": "/position_cmd", "to": "/p5/command"})
        ET.SubElement(server, "remap", **{"from": "/odom_world", "to": "/p5/odom"})
        ET.SubElement(server, "param", name="traj_server/time_forward", value="1.0")
    else:
        import yaml

        source = (
            "/opt/drone_playground/planners/super/src/SUPER/"
            "super_planner/config/click_smooth_ros1.yaml"
        )
        with open(source) as handle:
            cfg = yaml.safe_load(handle)
        cfg["fsm"].update(
            click_goal_topic="/p5/goal",
            click_height=-10.0 if interactive else float(goal[2]),
            cmd_topic="/p5/command",
            mpc_cmd_topic="/p5/polynomial",
        )
        cfg["super_planner"]["visualization_en"] = bool(
            bounds.get("record_planner_visualization", False)
        )
        cfg["traj_opt"]["boundary"].update(
            max_vel=limits["max_velocity_mps"],
            max_acc=limits["max_acceleration_mps2"],
        )
        cfg["rog_map"]["ros_callback"].update(cloud_topic="/p5/cloud", odom_topic="/p5/odom")
        cfg["rog_map"]["visualization"]["enable"] = False
        cfg["rog_map"]["virtual_ceil_height"] = ceiling
        cfg["rog_map"]["map_size"] = [40, 20, 6]
        target = folder / "super.yaml"
        with target.open("w") as handle:
            yaml.safe_dump(cfg, handle)
        node = ET.SubElement(
            root,
            "node",
            pkg="super_planner",
            name="fsm_node",
            type="fsm_node",
            output="screen",
        )
        ET.SubElement(
            node,
            "env",
            name="DRONE_PLAYGROUND_CONTROL_TRANSFER",
            value="1" if interactive else "0",
        )
        ET.SubElement(node, "param", name="config_path", value=str(target))
    path = folder / "planner.launch"
    ET.ElementTree(root).write(str(path))
    return str(path)
