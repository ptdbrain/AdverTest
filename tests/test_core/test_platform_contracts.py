from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.api.schemas.platform import ImportUltralyticsIn
from src.core.platform_contracts import (
    ArtifactKind,
    ArtifactState,
    ArtifactV1,
    AttackRecipeV1,
    BenchmarkRunStatus,
    BenchmarkRunV1,
    CheckpointArtifactV1,
    CheckpointSource,
    CheckpointStatus,
    ClassMappingStatus,
    ClassMappingV1,
    DatasetV1,
    DatasetVersionStatus,
    DatasetVersionV1,
    DefenceRunStatus,
    DefenceRunV1,
    JobStatus,
    JobV1,
    ModelComparisonStatus,
    ModelComparisonV1,
    ModelFamilyV1,
    ProjectMembershipV1,
    ProjectV1,
    TaskDefinitionV1,
    TaskId,
    TrainingRunStatus,
    TrainingRunV1,
    UserV1,
)

NOW = datetime(2026, 8, 19, tzinfo=UTC)
HASH = "a" * 64


def _artifact(**overrides: object) -> ArtifactV1:
    payload: dict[str, object] = {
        "id": uuid4(),
        "project_id": uuid4(),
        "created_by_user_id": uuid4(),
        "kind": ArtifactKind.CHECKPOINT,
        "state": ArtifactState.READY,
        "storage_key": "projects/project/artifacts/artifact/checkpoint.pt",
        "sha256": HASH,
        "size_bytes": 1024,
        "mime_type": "application/octet-stream",
        "original_filename": "checkpoint.pt",
        "created_at": NOW,
        "finalized_at": NOW,
    }
    payload.update(overrides)
    return ArtifactV1.model_validate(payload)


def test_finalized_artifact_requires_verified_integrity_metadata() -> None:
    with pytest.raises(ValidationError, match="finalized artifacts require"):
        _artifact(sha256=None)


def test_artifact_contract_is_frozen_and_rejects_unknown_fields() -> None:
    artifact = _artifact()
    with pytest.raises(ValidationError):
        artifact.state = ArtifactState.FAILED  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra"):
        ArtifactV1.model_validate({**artifact.model_dump(), "unexpected": True})


def test_ready_checkpoint_requires_native_class_metadata() -> None:
    with pytest.raises(ValidationError, match="complete native class metadata"):
        CheckpointArtifactV1(
            checkpoint_id=uuid4(),
            project_id=uuid4(),
            artifact_id=uuid4(),
            task_id=TaskId.DETECTION2D,
            model_family_id="yolo11",
            source=CheckpointSource.UPLOAD,
            status=CheckpointStatus.READY,
            created_at=NOW,
        )


def test_job_progress_and_terminal_timestamp_are_enforced() -> None:
    with pytest.raises(ValidationError, match="completed_units"):
        JobV1(
            id=uuid4(),
            project_id=uuid4(),
            owner_user_id=uuid4(),
            type="checkpoint_validation",
            status=JobStatus.RUNNING,
            stage="SMOKE_TESTING",
            completed_units=3,
            total_units=2,
            created_at=NOW,
        )
    with pytest.raises(ValidationError, match="terminal jobs require"):
        JobV1(
            id=uuid4(),
            project_id=uuid4(),
            owner_user_id=uuid4(),
            type="checkpoint_validation",
            status=JobStatus.COMPLETED,
            stage="COMPLETE",
            completed_units=2,
            total_units=2,
            created_at=NOW,
        )


def test_ultralytics_import_contract_only_accepts_whitelisted_model_ids() -> None:
    assert ImportUltralyticsIn(model_id="yolo11s").model_id == "yolo11s"
    with pytest.raises(ValidationError):
        ImportUltralyticsIn(model_id="https://untrusted.example/model.pt")


def test_task_model_family_user_project_and_dataset_contracts() -> None:
    task = TaskDefinitionV1(
        id=TaskId.DETECTION3D,
        display_name="3D Detection",
        required_modalities=("lidar",),
        prediction_contract="detection3d-prediction-v1",
        annotation_schema=("boxes3d", "class_labels"),
    )
    family = ModelFamilyV1(
        id="centerpoint3d",
        display_name="CenterPoint",
        supported_task_ids=(task.id,),
        checkpoint_extensions=(".pt", ".pth"),
        adapter_key="centerpoint3d",
    )
    user = UserV1(id=uuid4(), email="user@example.test", role="USER", created_at=NOW)
    project = ProjectV1(id=uuid4(), owner_user_id=user.id, name="Safety", created_at=NOW)
    membership = ProjectMembershipV1(
        project_id=project.id,
        user_id=user.id,
        role="OWNER",
        created_at=NOW,
    )
    dataset = DatasetV1(
        id=uuid4(),
        project_id=project.id,
        created_by_user_id=user.id,
        name="nuScenes validation",
        task_id=task.id,
        created_at=NOW,
    )
    version = DatasetVersionV1(
        id=uuid4(),
        dataset_id=dataset.id,
        project_id=project.id,
        manifest_artifact_id=uuid4(),
        version=1,
        status=DatasetVersionStatus.READY,
        schema_hash=HASH,
        sample_count=100,
        created_at=NOW,
    )

    assert family.supported_task_ids == (task.id,)
    assert membership.user_id == project.owner_user_id
    assert version.dataset_id == dataset.id


def test_attack_benchmark_training_defence_and_comparison_contracts() -> None:
    project_id = uuid4()
    user_id = uuid4()
    checkpoint_id = uuid4()
    dataset_version_id = uuid4()
    mapping = ClassMappingV1(
        id=uuid4(),
        project_id=project_id,
        checkpoint_id=checkpoint_id,
        dataset_version_id=dataset_version_id,
        native_class_id=0,
        canonical_class_id="Car",
        status=ClassMappingStatus.MAPPED,
        created_at=NOW,
    )
    recipe = AttackRecipeV1(
        id=uuid4(),
        project_id=project_id,
        created_by_user_id=user_id,
        task_id=TaskId.DETECTION2D,
        recipe_hash=HASH,
        catalog_version="1.0.0",
        implementation_version="1.0.0",
        created_at=NOW,
    )
    benchmark = BenchmarkRunV1(
        id=uuid4(),
        project_id=project_id,
        created_by_user_id=user_id,
        checkpoint_id=checkpoint_id,
        dataset_version_id=dataset_version_id,
        attack_recipe_id=recipe.id,
        class_mapping_id=mapping.id,
        job_id=uuid4(),
        protocol_id="locked-protocol",
        status=BenchmarkRunStatus.COMPLETED,
        completed_at=NOW,
        created_at=NOW,
    )
    training = TrainingRunV1(
        id=uuid4(),
        project_id=project_id,
        created_by_user_id=user_id,
        parent_checkpoint_id=checkpoint_id,
        dataset_version_id=dataset_version_id,
        defence_profile_id="balanced-v1",
        status=TrainingRunStatus.COMPLETED,
        seed=195,
        output_checkpoint_id=uuid4(),
        created_at=NOW,
        completed_at=NOW,
    )
    defence = DefenceRunV1(
        id=uuid4(),
        project_id=project_id,
        created_by_user_id=user_id,
        baseline_checkpoint_id=checkpoint_id,
        baseline_benchmark_run_id=benchmark.id,
        defence_profile_id="balanced-v1",
        training_run_id=training.id,
        defended_checkpoint_id=training.output_checkpoint_id,
        defended_benchmark_run_id=uuid4(),
        status=DefenceRunStatus.COMPLETED,
        created_at=NOW,
        completed_at=NOW,
    )
    comparison = ModelComparisonV1(
        id=uuid4(),
        project_id=project_id,
        created_by_user_id=user_id,
        baseline_benchmark_run_id=benchmark.id,
        candidate_benchmark_run_id=defence.defended_benchmark_run_id,
        status=ModelComparisonStatus.COMPLETED,
        paired=True,
        created_at=NOW,
        completed_at=NOW,
    )

    assert recipe.project_id == project_id
    assert defence.training_run_id == training.id
    assert comparison.paired is True


def test_scientific_benchmark_requires_class_mapping() -> None:
    with pytest.raises(ValidationError, match="require class_mapping_id"):
        BenchmarkRunV1(
            id=uuid4(),
            project_id=uuid4(),
            created_by_user_id=uuid4(),
            checkpoint_id=uuid4(),
            dataset_version_id=uuid4(),
            job_id=uuid4(),
            protocol_id="locked-protocol",
            status=BenchmarkRunStatus.QUEUED,
            created_at=NOW,
        )
