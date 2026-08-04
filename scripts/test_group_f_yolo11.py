"""Run the reproducible Group F YOLO11 smoke recipe through the public CLI.

Usage:
    uv run python scripts/test_group_f_yolo11.py
    uv run python scripts/test_group_f_yolo11.py configs/kitti-square-yolo11.json
"""

from __future__ import annotations

import sys

from src.cli import main

if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/kitti-square-yolo11.json"
    raise SystemExit(main(["generate-attack", "--config", config]))
