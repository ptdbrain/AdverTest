"""Explicit bootstrap for a runnable public demo catalog.

This deliberately supports only the server-maintained Ultralytics model ID.
It is not an upload or validation shortcut for user-supplied checkpoints.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def ensure_demo_checkpoint(*, enabled: bool, checkpoint_root: str, model_id: str) -> Path | None:
    if not enabled:
        return None
    target = Path(checkpoint_root).expanduser().resolve() / "surrogates" / f"{model_id}.pt"
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    # This identifier is a fixed allow-list entry, not caller controlled.
    from ultralytics import YOLO

    model = YOLO(f"{model_id}.pt")
    source = Path(str(model.ckpt_path)).expanduser().resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise RuntimeError("DEMO_CHECKPOINT_DOWNLOAD_FAILED")
    if source != target:
        shutil.copyfile(source, target)
    return target
