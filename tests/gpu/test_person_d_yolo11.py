import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu


def test_person_d_yolo11_owner_handoff() -> None:
    checkpoint = os.getenv("ADVERTEST_YOLO11_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).is_file():
        pytest.skip("WAITING_FOR_OWNER: set ADVERTEST_YOLO11_CHECKPOINT to a local B checkpoint")
    from src.adapters import get_adapter

    adapter = get_adapter("yolo11", weights=checkpoint, device="cpu")
    assert adapter.metadata().checkpoint_hash
