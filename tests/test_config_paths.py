from __future__ import annotations

from pathlib import Path

from src.config import Settings


def test_default_runs_root_is_anchored_to_the_checkout() -> None:
    """Launching the API from another working directory still finds local YOLO artefacts."""
    assert Path(Settings().runs_root) == Path(__file__).resolve().parents[1] / "runs"
