"""Evaluate fixed method parameters on the selected task protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drone_playground.app import script_main

if __name__ == "__main__":
    script_main("eval")
