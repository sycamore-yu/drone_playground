"""CLI to export a trusted native RScope recording for MuJoCo WASM playback."""

from __future__ import annotations

import argparse
from pathlib import Path

from drone_playground.simulation.browser_replay import export

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_directory", type=Path)
    parser.add_argument("--max-frames", type=int, default=350)
    args = parser.parse_args()
    print(export(args.episode_directory, max_frames=args.max_frames))
