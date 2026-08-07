import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu


def test_person_d_sam2_owner_handoff() -> None:
    checkpoint = os.getenv("ADVERTEST_SAM2_CHECKPOINT")
    config = os.getenv("ADVERTEST_SAM2_CONFIG")
    if not checkpoint or not config or not Path(checkpoint).is_file():
        pytest.skip(
            "WAITING_FOR_OWNER: set ADVERTEST_SAM2_CHECKPOINT and ADVERTEST_SAM2_CONFIG"
        )
    from src.adapters import get_adapter

    adapter = get_adapter("sam2_surrogate", weights=checkpoint, config=config, device="cpu")
    assert adapter.metadata().checkpoint_hash
