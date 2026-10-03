"""Launch the installed implementation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from drone_playground.visualization.rscope_client import main

if __name__ == "__main__":
    main()
