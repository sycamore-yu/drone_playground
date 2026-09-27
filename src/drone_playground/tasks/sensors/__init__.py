"""Virtual sensor models. Each sensor owns its geometry query, timing and framing."""

from drone_playground.tasks.sensors.depth import DepthCamera, DepthFrame, cast_depth

__all__ = ["DepthCamera", "DepthFrame", "cast_depth"]
