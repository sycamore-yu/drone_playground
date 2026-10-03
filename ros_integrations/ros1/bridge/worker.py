"""Launch the installed implementation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from drone_playground.integrations.ros1_worker import main

if __name__ == "__main__":
    main()
