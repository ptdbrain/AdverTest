from __future__ import annotations

import time
from pathlib import Path

from src.api.training_service import TrainingJobService
from src.api.workflow_store import WorkflowJobStore
from src.core.hashing import file_digest
from src.training import (
    CheckpointMetadata,
    ExportedCheckpoint,
    MetricSnapshot,
    ModelTrainer,
    PreparedTrainingData,
    TrainerCallbacks,
    TrainerMetadata,
    TrainerRegistry,
    TrainingEstimate,
    TrainingReport,
    TrainingRunConfig,
    ValidationReport,
)


class _CpuTrainer(ModelTrainer):
    def __init__(self, root: Path) -> None:
        self.root = root

    def validate_config(self, config: TrainingRunConfig) -> ValidationReport:
        return ValidationReport(valid=True)

    def estimate(self, config: TrainingRunConfig) -> TrainingEstimate:
        return TrainingEstimate(gpu_hours=0.0, storage_bytes=1, wall_time_seconds=1)

    def prepare_data(self, config: TrainingRunConfig) -> PreparedTrainingData:
        return PreparedTrainingData(manifest_id="manifest-1", manifest_hash="a" * 64, lineage_valid=True)

    def train(self, config: TrainingRunConfig, callbacks: TrainerCallbacks) -> TrainingReport:
        checkpoint = self.root / f"{config.run_id}.bin"
        checkpoint.write_bytes(b"cpu-checkpoint")
        callbacks.on_epoch(1, {"loss": 0.25})
        return TrainingReport(
            run_id=config.run_id,
            state="TRAINING",
            checkpoint=CheckpointMetadata(
                path=str(checkpoint),
                sha256=file_digest(checkpoint, length=64),
                parent_model_version=config.model_version,
            ),
        )

    def evaluate_checkpoint(self, checkpoint: CheckpointMetadata) -> MetricSnapshot:
        return MetricSnapshot(metrics={"clean": 0.9})

    def export_checkpoint(self, checkpoint: CheckpointMetadata) -> ExportedCheckpoint:
        return ExportedCheckpoint(path=checkpoint.path, sha256=checkpoint.sha256, load_valid=True)

    def metadata(self) -> TrainerMetadata:
        return TrainerMetadata(name="cpu", task="detection2d", version="1.0.0")


def test_training_job_persists_events_checkpoint_and_child_lineage(tmp_path: Path) -> None:
    """Catch a queued job that loses worker events or checkpoint registration."""
    registry = TrainerRegistry()
    registry.register(_CpuTrainer(tmp_path))
    store = WorkflowJobStore(f"sqlite:///{tmp_path / 'jobs.db'}")
    service = TrainingJobService(store, registry)
    request = TrainingRunConfig(
        run_id="request-id",
        trainer_name="cpu",
        model_version="base-v1",
        dataset_version_id="dataset-v1",
        split_manifest_id="split-v1",
        defense_profile_id="defense-v1",
        seed=17,
        epochs=1,
        batch_size=1,
        learning_rate=0.001,
    )

    job_id = service.enqueue(request)
    for _ in range(100):
        job = service.get(job_id)
        if job and job["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
            break
        time.sleep(0.01)
    else:
        raise AssertionError("training job did not terminate")

    assert job["status"] == "COMPLETED", job
    assert job["result"]["model_version"]["parent_id"] == "base-v1"
    assert job["result"]["checkpoint"]["sha256"] == file_digest(tmp_path / f"{job_id}.bin", length=64)
    assert [event["state"] for event in store.events(job_id)][-1] == "COMPLETED"
