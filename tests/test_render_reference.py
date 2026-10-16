"""Compare ideal camera depth with MuJoCo's independent OpenGL renderer."""

import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from drone_playground.simulation.scene import Scene
from drone_playground.simulation.sensors import SensorConfig, measure


@pytest.mark.skipif(
    not os.environ.get("DISPLAY") and os.environ.get("MUJOCO_GL") not in {"egl", "osmesa"},
    reason="An OpenGL context is required for the independent renderer reference",
)
@pytest.mark.parametrize("wall_pitch", [0, 15])
def test_depth_matches_official_renderer(tmp_path, wall_pitch):
    """Pixel-center rays match metric axial depth for front-facing and slanted walls."""
    path = tmp_path / "camera.xml"
    path.write_text(
        '<mujoco><visual><map znear="0.001" zfar="10"/></visual><worldbody>'
        f'<geom type="box" pos="5.5 0 0" size=".5 20 20" euler="0 {wall_pitch} 0"/>'
        '<camera name="depth" xyaxes="0 -1 0 0 0 1" fovy="58"/>'
        "</worldbody></mujoco>"
    )
    output = tmp_path / "depth.npy"
    # A broken graphics driver must not crash the parent test process.
    script = """
import sys
import mujoco
import numpy as np
model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
with mujoco.Renderer(model, width=64, height=48) as renderer:
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="depth")
    np.save(sys.argv[2], renderer.render())
"""
    subprocess.run([sys.executable, "-c", script, str(path), str(output)], check=True, timeout=30)
    # This reference uses one fovy and the image aspect ratio, as MuJoCo does.
    hfov = float(np.rad2deg(2 * np.arctan(64 / 48 * np.tan(np.deg2rad(58) / 2))))
    config = SensorConfig.d435i(width=64, height=48, horizontal_fov_deg=hfov, max_range=10)
    actual = measure(Scene(path), config, jnp.zeros(3), jnp.array([0.0, 0, 0, 1]), 0.0)
    assert actual.mask.all()
    np.testing.assert_allclose(actual.values, np.load(output), atol=2e-5, rtol=2e-5)
