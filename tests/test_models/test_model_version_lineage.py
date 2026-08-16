"""Wave 0 — ModelVersion extended lineage tests."""

from __future__ import annotations

from src.models.versions import ModelVersion


class TestModelVersionLineage:
    """Verify the extended lineage fields work correctly."""

    def test_default_lineage_fields(self):
        v = ModelVersion(
            id="yolo-b0",
            model_name="yolo11s",
            task="detection2d",
            checkpoint_path=None,
            checkpoint_hash=None,
            parent_id=None,
            training_metadata={},
            runnable=False,
        )
        assert v.parent_lineage == ()
        assert v.source_training_run is None
        assert v.training_dataset_manifest is None
        assert v.checkpoint_validated is False
        assert v.gate_outcome is None
        assert v.evidence_tier is None

    def test_full_lineage(self):
        v = ModelVersion(
            id="yolo-r1",
            model_name="yolo11s",
            task="detection2d",
            checkpoint_path="/path/to/best.pt",
            checkpoint_hash="sha256abc",
            parent_id="yolo-b0",
            training_metadata={"role": "yolo_r1"},
            runnable=True,
            parent_lineage=("yolo-b0",),
            source_training_run="train-run-001",
            training_dataset_manifest="manifest-001",
            checkpoint_validated=True,
            gate_outcome="PASSED",
            evidence_tier="REAL_MODEL_VERIFIED",
        )
        assert v.parent_id == "yolo-b0"
        assert v.parent_lineage == ("yolo-b0",)
        assert v.source_training_run == "train-run-001"
        assert v.checkpoint_validated is True
        assert v.gate_outcome == "PASSED"

    def test_r2_lineage_chain(self):
        """R2.parent = R1, R1.parent = B0."""
        b0 = ModelVersion(
            id="yolo-b0", model_name="yolo11s", task="detection2d",
            checkpoint_path=None, checkpoint_hash=None,
            parent_id=None, training_metadata={}, runnable=True,
        )
        r1 = ModelVersion(
            id="yolo-r1", model_name="yolo11s", task="detection2d",
            checkpoint_path=None, checkpoint_hash=None,
            parent_id="yolo-b0", training_metadata={}, runnable=True,
            parent_lineage=("yolo-b0",),
        )
        r2 = ModelVersion(
            id="yolo-r2-fog", model_name="yolo11s", task="detection2d",
            checkpoint_path=None, checkpoint_hash=None,
            parent_id="yolo-r1", training_metadata={}, runnable=True,
            parent_lineage=("yolo-r1", "yolo-b0"),
        )
        assert r1.parent_id == b0.id
        assert r2.parent_id == r1.id
        assert r2.parent_lineage == ("yolo-r1", "yolo-b0")
        # No self-reference
        assert r2.id not in r2.parent_lineage

    def test_not_runnable_without_checkpoint(self):
        v = ModelVersion(
            id="yolo-r1", model_name="yolo11s", task="detection2d",
            checkpoint_path=None, checkpoint_hash=None,
            parent_id="yolo-b0", training_metadata={},
            runnable=False,
            blocked_reason="CHECKPOINT_MISSING",
        )
        assert not v.runnable
        assert v.blocked_reason == "CHECKPOINT_MISSING"

    def test_sam_waiting_for_artifacts(self):
        v = ModelVersion(
            id="sam2-base", model_name="sam2", task="segmentation",
            checkpoint_path="/path/sam2.pt", checkpoint_hash="sha256xyz",
            parent_id=None, training_metadata={},
            runnable=False,
            blocked_reason="WAITING_FOR_ARTIFACTS",
            evidence_tier=None,
        )
        assert v.blocked_reason == "WAITING_FOR_ARTIFACTS"
        assert v.evidence_tier is None
