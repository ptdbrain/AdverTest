from __future__ import annotations

import numpy as np
import pytest

from src.core.hashing import array_digest
from src.core.objectives import AttackObjective
from src.evaluation.contracts import MetricEnvelope
from src.training.hard_example_bank import HardExampleBank, HardExampleRecord


def _metric(name: str, value: float) -> MetricEnvelope:
    return MetricEnvelope(
        name=name,
        value=value,
        unit="ratio",
        percent_value=value * 100,
        version="1.0.0",
        higher_is_better=True,
    )


def _record(
    artifact: np.ndarray,
    *,
    artifact_id: str = "hard-1",
    attack_name: str = "pgd",
    allowed_uses: tuple[str, ...] = ("training", "review"),
    locked_test: bool = False,
    provenance: dict | None = None,
) -> HardExampleRecord:
    return HardExampleRecord(
        artifact_id=artifact_id,
        task="detection2d",
        source_sample_id="sample-1",
        source_hash="source-hash-1",
        model_id="yolo",
        model_version="yolo-v1",
        attack_name=attack_name,
        attack_version="1.0.0",
        attack_family="white_box",
        protocol_id="protocol-1",
        protocol_version="1.0.0",
        objective=AttackObjective(kind="untargeted"),
        parameters={"epsilon": 0.03},
        seeds=(195,),
        before_metrics=(_metric("ap", 0.9),),
        after_metrics=(_metric("ap", 0.2),),
        failure_reason="object_vanishing",
        affected_instances=("car-1",),
        class_label="Car",
        object_size_bucket="medium",
        severity=4,
        artifact_hash=array_digest(artifact, length=64),
        allowed_uses=allowed_uses,
        locked_test=locked_test,
        provenance=(
            {
                "dataset_version_id": "dataset-1",
                "recipe_hash": "recipe-1",
            }
            if provenance is None
            else provenance
        ),
    )


@pytest.mark.parametrize(
    "attack_name",
    ("pgd", "cw_l2", "square_attack", "dpatch", "sam2_pgd", "critical_scenario", "targeted_repair"),
)
def test_bank_stores_strong_and_targeted_hard_examples(tmp_path, attack_name: str) -> None:
    artifact = np.full((4, 5, 3), 0.25, dtype=np.float32)
    bank = HardExampleBank(tmp_path / "bank")
    record = _record(artifact, artifact_id=f"hard-{attack_name}", attack_name=attack_name)

    bank.put(record, artifact)
    restored, restored_artifact = bank.get(record.artifact_id, intended_use="training")

    assert restored == record
    assert np.array_equal(restored_artifact, artifact)


def test_bank_rejects_hash_mismatch_missing_provenance_and_locked_training(tmp_path) -> None:
    artifact = np.zeros((2, 2, 3), dtype=np.float32)
    bank = HardExampleBank(tmp_path / "bank")

    with pytest.raises(ValueError, match="hash"):
        bank.put(
            _record(artifact).model_copy(update={"artifact_hash": "wrong"}),
            artifact,
        )
    with pytest.raises(ValueError, match="provenance"):
        bank.put(_record(artifact, provenance={}), artifact)
    with pytest.raises(ValueError, match="locked"):
        bank.put(_record(artifact, locked_test=True), artifact)


def test_benchmark_only_artifact_cannot_be_read_for_training(tmp_path) -> None:
    artifact = np.ones((2, 2, 3), dtype=np.float32)
    bank = HardExampleBank(tmp_path / "bank")
    record = _record(
        artifact,
        allowed_uses=("benchmark", "review"),
    )
    bank.put(record, artifact)

    with pytest.raises(PermissionError, match="training"):
        bank.get(record.artifact_id, intended_use="training")


def test_query_is_deterministic_and_permission_filtered(tmp_path) -> None:
    bank = HardExampleBank(tmp_path / "bank")
    for index, attack in enumerate(("pgd", "cw_l2", "dpatch")):
        artifact = np.full((2, 2, 3), index, dtype=np.float32)
        bank.put(
            _record(
                artifact,
                artifact_id=f"hard-{index}",
                attack_name=attack,
                allowed_uses=("training",),
            ),
            artifact,
        )

    records = bank.query(
        intended_use="training",
        task="detection2d",
        attack_family="white_box",
    )

    assert [record.artifact_id for record in records] == [
        "hard-0",
        "hard-1",
        "hard-2",
    ]
