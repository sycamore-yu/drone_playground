"""Add generic visualization to a trusted local replay without reevaluation."""

import argparse
import json
from pathlib import Path

from drone_playground.artifacts.decisions import load_native_decisions
from drone_playground.visualization.layers import ReplayLayers, layers_from_decisions, sensor_view
from drone_playground.visualization.rscope_io import enhance_replay


def main():
    """Export an existing native rollout with optional sensor and decision layers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="Actual sensor calibration JSON")
    parser.add_argument("--decision-archive", type=Path)
    parser.add_argument("--lidar-display-range", type=float, default=10.0)
    parser.add_argument("--drone-mocap-id", type=int, default=0)
    args = parser.parse_args()
    sensor = (
        sensor_view(
            json.loads(args.calibration.read_text()),
            lidar_display_range_m=args.lidar_display_range,
        )
        if args.calibration
        else None
    )
    layers = (
        layers_from_decisions(load_native_decisions(args.decision_archive), sensor=sensor)
        if args.decision_archive
        else ReplayLayers(sensor=sensor)
    )
    print(enhance_replay(args.source, args.output, layers, drone_mocap_id=args.drone_mocap_id))


if __name__ == "__main__":
    main()
