"""Copy a trusted local historical checkpoint into the current artifact schema."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from drone_playground.runs.migration import migrate_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="迁移可信本地检查点；源产物保持只读，目标应为新目录"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(migrate_checkpoint(args.source, args.destination))


if __name__ == "__main__":
    main()
