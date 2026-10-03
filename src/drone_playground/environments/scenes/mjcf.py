"""Compile scene assets into runtime geometry; never keep a second geometry source."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


def custom_text(path, name):
    """Read nongeometric metadata from the MJCF custom section."""
    node = ET.parse(path).find(f"custom/text[@name='{name}']")
    if node is None:
        raise ValueError(f"Scene asset is missing metadata {name}: {path}")
    return node.get("data")


def compiled_obstacles(path):
    """Read primitive sizes, origins and motion parameters from a compiled model."""
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    obstacles = []
    for index in range(model.ngeom):
        geom = model.geom(index)
        if not geom.name.startswith("obstacle_geom_"):
            continue
        body = model.body(int(model.geom_bodyid[index]))
        shape = int(model.geom_type[index])
        if shape == mujoco.mjtGeom.mjGEOM_CYLINDER:
            shape_name = "cylinder"
            size = [model.geom_size[index, 0], 2 * model.geom_size[index, 1], 0.0]
        elif shape == mujoco.mjtGeom.mjGEOM_BOX:
            shape_name = "box"
            size = model.geom_size[index].tolist()
        else:
            raise ValueError(f"Unsupported fixed navigation geometry: {geom.name}")
        motion = int(model.body_user[body.id, 0])
        if motion not in (0, 1, 2):
            raise ValueError(f"Unknown obstacle motion code: {motion}")
        obstacles.append(
            dict(
                shape=shape_name,
                size=list(map(float, size)),
                origin=data.geom_xpos[index].tolist(),
                motion=("static", "trefoil", "linear_bounce")[motion],
                params=model.body_user[body.id, 1:6].tolist(),
            )
        )
    return obstacles


def load_catalog(path):
    """Load fixed scenes; only descriptive metadata is stored outside model fields."""
    path = Path(path)
    if path.suffix != ".xml":
        raise ValueError("Fixed scene assets must be MJCF/XML")
    metadata = json.loads(custom_text(path, "catalog_metadata"))
    model = mujoco.MjModel.from_xml_path(str(path))
    low, high = model.numeric("world_low").data, model.numeric("world_high").data
    world = dict(
        metadata.pop("world_metadata"),
        length_m=float(high[0] - low[0]),
        width_m=float(high[1] - low[1]),
        z_min_m=float(low[2]),
        z_max_m=float(high[2]),
        start=model.site("start").pos.tolist(),
        goal=model.site("goal").pos.tolist(),
    )
    boundaries = compiled_obstacles(path.parent / "boundary.xml")
    for obstacle, extra in zip(boundaries, metadata.pop("boundary_metadata")):
        obstacle.update(extra)
    scenes = []
    for filename in custom_text(path, "scene_files").split():
        source = path.parent / filename
        scene = json.loads(custom_text(source, "scene_metadata"))
        obstacles = compiled_obstacles(source)[len(boundaries) :]
        for obstacle, extra in zip(obstacles, scene.pop("obstacle_metadata")):
            obstacle.update(extra)
        scene.update(obstacles=obstacles, asset_path=str(source))
        scenes.append(scene)
    return dict(metadata, world=world, boundary_obstacles=boundaries, scenes=scenes)


def instance_spec(bank, scenario_id):
    """Return the authoritative fixed asset or serialize sampled runtime parameters.

    Fixed assets are read without reconstructing their geometry. Procedural banks
    are runtime instances: their actual sampled sizes and poses are serialized by
    this scene adapter, shared by visualization and external integrations.
    """
    source = (
        bank.asset_paths[int(np.asarray(bank.subtype[scenario_id]))] if bank.asset_paths else None
    )
    if source:
        return mujoco.MjSpec.from_file(source)
    root = ET.Element("mujoco", model="sampled-scene")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    from scipy.spatial.transform import Rotation

    from drone_playground.environments.scenes.geometry import (
        KIND_BOX,
        KIND_CAPSULE,
        KIND_CYLINDER,
        KIND_SPHERE,
    )

    for output_index, index in enumerate(np.flatnonzero(np.asarray(bank.active[scenario_id]))):
        kind = int(np.asarray(bank.kind[scenario_id, index]))
        size = np.asarray(bank.size[scenario_id, index], float)
        origin = np.asarray(bank.origin[scenario_id, index], float)
        shape = {
            KIND_BOX: "box",
            KIND_CAPSULE: "capsule",
            KIND_CYLINDER: "cylinder",
            KIND_SPHERE: "sphere",
        }[kind]
        native_size = (
            size
            if kind == KIND_BOX
            else size[:1]
            if kind == KIND_SPHERE
            else [size[0], size[1] / 2]
        )
        body = ET.SubElement(
            world,
            "body",
            name=f"obstacle_{output_index}",
            mocap="true",
            pos=" ".join(map(str, origin)),
        )
        attributes = dict(
            name=f"obstacle_geom_{output_index}",
            type=shape,
            size=" ".join(map(str, native_size)),
            rgba="0.35 0.42 0.5 1"
            if int(bank.motion[scenario_id, index]) == 0
            else "0.95 0.45 0.12 1",
        )
        if bank.rotations is not None:
            quaternion = Rotation.from_matrix(
                np.asarray(bank.rotations[scenario_id, index])
            ).as_quat(scalar_first=True)
            attributes["quat"] = " ".join(map(str, quaternion))
        ET.SubElement(body, "geom", **attributes)
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def write_catalog(catalog, directory):
    """Write candidate scene assets to a new directory without replacing a benchmark."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)

    def numbers(values):
        return " ".join(format(float(value), ".17g") for value in values)

    def write(filename, tree):
        ET.indent(tree, space="  ")
        path = directory / filename
        path.write_text(ET.tostring(tree, encoding="unicode") + "\n")
        mujoco.MjModel.from_xml_path(str(path))

    geometry_fields = {"shape", "origin", "size", "motion", "params"}

    def describe(obstacles):
        return [
            {key: value for key, value in item.items() if key not in geometry_fields}
            for item in obstacles
        ]

    def add_obstacle(world, obstacle, index):
        motion = ("static", "trefoil", "linear_bounce").index(obstacle.get("motion", "static"))
        body = ET.SubElement(
            world,
            "body",
            name=f"obstacle_{index}",
            mocap="true",
            pos=numbers(obstacle["origin"]),
            user=numbers([motion, *obstacle.get("params", [0.0] * 5)]),
        )
        size = obstacle["size"]
        if obstacle["shape"] == "cylinder":
            size = [size[0], size[1] / 2]
        ET.SubElement(
            body,
            "geom",
            name=f"obstacle_geom_{index}",
            type=obstacle["shape"],
            size=numbers(size),
            rgba="0.35 0.42 0.5 1" if motion == 0 else "0.95 0.45 0.12 1",
        )

    boundaries = catalog.get("boundary_obstacles", [])
    tree = ET.Element("mujoco", model="navigation-boundary")
    world = ET.SubElement(tree, "worldbody")
    for index, obstacle in enumerate(boundaries):
        add_obstacle(world, obstacle, index)
    write("boundary.xml", tree)
    for scene in catalog["scenes"]:
        tree = ET.Element("mujoco", model=scene["id"])
        ET.SubElement(tree, "compiler", angle="radian")
        ET.SubElement(tree, "include", file="boundary.xml")
        metadata = {
            key: value for key, value in scene.items() if key not in ("obstacles", "asset_path")
        }
        metadata["obstacle_metadata"] = describe(scene["obstacles"])
        ET.SubElement(
            ET.SubElement(tree, "custom"),
            "text",
            name="scene_metadata",
            data=json.dumps(metadata, ensure_ascii=False),
        )
        world = ET.SubElement(tree, "worldbody")
        for index, obstacle in enumerate(scene["obstacles"], start=len(boundaries)):
            add_obstacle(world, obstacle, index)
        write(scene["id"] + ".xml", tree)
    tree = ET.Element("mujoco", model="navigation-catalog")
    custom = ET.SubElement(tree, "custom")
    metadata = {
        key: value
        for key, value in catalog.items()
        if key not in ("scenes", "boundary_obstacles", "world")
    }
    metadata["world_metadata"] = {
        key: value
        for key, value in catalog["world"].items()
        if key not in ("length_m", "width_m", "z_min_m", "z_max_m", "start", "goal")
    }
    metadata["boundary_metadata"] = describe(boundaries)
    ET.SubElement(
        custom, "text", name="catalog_metadata", data=json.dumps(metadata, ensure_ascii=False)
    )
    ET.SubElement(
        custom,
        "text",
        name="scene_files",
        data=" ".join(scene["id"] + ".xml" for scene in catalog["scenes"]),
    )
    bounds = catalog["world"]
    ET.SubElement(
        custom,
        "numeric",
        name="world_low",
        data=numbers([0, -bounds["width_m"] / 2, bounds["z_min_m"]]),
    )
    ET.SubElement(
        custom,
        "numeric",
        name="world_high",
        data=numbers([bounds["length_m"], bounds["width_m"] / 2, bounds["z_max_m"]]),
    )
    world = ET.SubElement(tree, "worldbody")
    for name in ("start", "goal"):
        ET.SubElement(world, "site", name=name, pos=numbers(bounds[name]), size="0.001")
    write("catalog.xml", tree)
    return directory / "catalog.xml"
