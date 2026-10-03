"""Scene assets, analytic sensing and replay use the same physical geometry."""

import jax.numpy as jnp
import mujoco
import numpy as np
import pytest


@pytest.mark.parametrize("scene_id", ["S01", "D01", "S06", "D06"])
def test_fixed_scene_bank_matches_compiled_mjcf_and_runtime_pose(scene_id):
    from drone_playground.environments.scenes.catalog import NavigationCatalogScene
    from drone_playground.environments.scenes.geometry import obstacle_positions
    from drone_playground.resources import resource_path

    bank, _ = NavigationCatalogScene(scene_ids=(scene_id,)).build()
    model = mujoco.MjModel.from_xml_path(
        str(resource_path("assets/scenes/navigation/" + scene_id + ".xml"))
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for index in np.flatnonzero(np.asarray(bank.active[0])):
        geometry = model.geom("obstacle_geom_" + str(index))
        np.testing.assert_allclose(data.geom_xpos[geometry.id], bank.origin[0, index], atol=1e-5)
        size = geometry.size.copy()
        if geometry.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            size = np.array([size[0], 2 * size[1], 0.0])
        np.testing.assert_allclose(size, bank.size[0, index], atol=1e-6)
    positions = np.asarray(obstacle_positions(bank, jnp.int32(0), jnp.float32(2.5)))
    for index in np.flatnonzero(np.asarray(bank.active[0])):
        body = model.body("obstacle_" + str(index))
        data.mocap_pos[body.mocapid[0]] = positions[index]
    mujoco.mj_forward(model, data)
    for index in np.flatnonzero(np.asarray(bank.active[0])):
        np.testing.assert_allclose(
            data.geom_xpos[model.geom("obstacle_geom_" + str(index)).id],
            positions[index],
            atol=1e-5,
        )


def test_racing_pose_and_gate_order_come_from_mjcf():
    from drone_playground.environments.scenes.racing import load_lsy_config
    from drone_playground.resources import resource_path

    config = load_lsy_config()
    model = mujoco.MjModel.from_xml_path(str(resource_path("assets/scenes/racing/lsy_level0.xml")))
    for index, gate in enumerate(config.env.track.gates):
        np.testing.assert_allclose(model.body("gate:" + str(index)).pos, gate.pos)
    assert list(config.env.track.gate_order) == [1, 2, 3, 4, 2]


def test_candidate_mjcf_writer_preserves_geometry_and_never_overwrites(tmp_path):
    from drone_playground.environments.scenes.catalog import load_fixed_catalog
    from drone_playground.environments.scenes.mjcf import write_catalog

    original = load_fixed_catalog()
    original["scenes"] = [scene for scene in original["scenes"] if scene["id"] in ("S01", "D06")]
    path = write_catalog(original, tmp_path / "candidate")
    restored = load_fixed_catalog(path)
    assert restored["world"] == original["world"]
    assert restored["boundary_obstacles"] == original["boundary_obstacles"]
    for before, after in zip(original["scenes"], restored["scenes"], strict=True):
        assert before["obstacles"] == after["obstacles"]
    with pytest.raises(FileExistsError):
        write_catalog(original, path.parent)
