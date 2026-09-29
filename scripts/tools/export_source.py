#!/usr/bin/env python3
"""Export a committed source snapshot without experiment artifacts or Git history."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from drone_playground.source_archive import export_source  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    print(json.dumps(export_source(ROOT, args.output, args.revision), indent=2))
