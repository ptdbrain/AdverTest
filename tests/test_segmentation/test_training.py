from __future__ import annotations

import pytest

from src.training.sam2_trainer import Sam2Trainer, sam_b0_config, sam_r1_config, sam_r2_config


def test_b0_and_r1_configs_validate() -> None:
    Sam2Trainer().validate_config(sam_b0_config("manifest-b0"))
    config = sam_r1_config("manifest-r1", "sam-b0")
    Sam2Trainer().validate_config(config)
    assert Sam2Trainer().estimate(config, samples=10)["effective_batch_size"] == 1


def test_r2_requires_failure_cluster() -> None:
    with pytest.raises(ValueError, match="failure cluster"):
        sam_r2_config("manifest-r2", "sam-r1", ()).validate()
