from pathlib import Path

import pytest

from src.core.hashing import file_digest
from src.training import (
    CheckpointMetadata,
    ComputeWorker,
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
    TrainingStateMachine,
    ValidationReport,
)


def config(name: str = "fake-yolo") -> TrainingRunConfig:
    return TrainingRunConfig(
        run_id="run-1",
        trainer_name=name,
        model_version="base-v1",
        dataset_version_id="dataset-v1",
        split_manifest_id="split-v1",
        defense_profile_id="defense-v1",
        seed=195,
        epochs=2,
        batch_size=1,
        learning_rate=0.001,
        max_gpu_hours=1.0,
        max_storage_bytes=10_000,
        max_wall_time_seconds=60,
    )


class FakeTrainer(ModelTrainer):
    def __init__(self, root: Path, name: str):
        self.root = root
        self.name = name

    def validate_config(self, config):
        return ValidationReport(valid=True)

    def estimate(self, config):
        return TrainingEstimate(gpu_hours=0.1, storage_bytes=100, wall_time_seconds=5)

    def prepare_data(self, config):
        return PreparedTrainingData(
            manifest_id="training-data-v1",
            manifest_hash="manifest-hash",
            lineage_valid=True,
        )

    def train(self, config, callbacks: TrainerCallbacks):
        checkpoint = self.root / f"{self.name}.bin"
        checkpoint.write_bytes(b"deterministic-checkpoint")
        callbacks.on_epoch(1, {"loss": 0.5})
        return TrainingReport(
            run_id=config.run_id,
            state="TRAINING",
            epoch_metrics=({"epoch": 1.0, "loss": 0.5},),
            checkpoint=CheckpointMetadata(
                path=str(checkpoint),
                sha256=file_digest(checkpoint, length=64),
                parent_model_version=config.model_version,
            ),
        )

    def evaluate_checkpoint(self, checkpoint):
        return MetricSnapshot(metrics={"clean": 0.9})

    def export_checkpoint(self, checkpoint):
        return ExportedCheckpoint(
            path=checkpoint.path,
            sha256=checkpoint.sha256,
            load_valid=True,
        )

    def metadata(self):
        return TrainerMetadata(name=self.name, task="detection2d", version="1.0.0")


def test_state_machine_accepts_only_declared_transitions() -> None:
    machine = TrainingStateMachine()
    for state in (
        "VALIDATING",
        "ESTIMATING",
        "QUEUED",
        "PREPARING_DATA",
        "TRAINING",
        "VALIDATING_CHECKPOINT",
        "EXPORTING",
        "REGISTERING_MODEL",
        "COMPLETED",
    ):
        machine.transition(state)
    assert machine.state == "COMPLETED"
    with pytest.raises(ValueError, match="illegal training transition"):
        machine.transition("FAILED")


@pytest.mark.parametrize("name", ["fake-yolo", "fake-sam"])
def test_worker_dispatches_cpu_fakes_and_emits_lineage(tmp_path: Path, name: str) -> None:
    registry = TrainerRegistry()
    registry.register(FakeTrainer(tmp_path, name))
    events = []

    result = ComputeWorker(registry).run(config(name), callbacks=events.append)

    assert result.state == "COMPLETED"
    assert result.registration is not None
    assert result.registration["parent_model_version"] == "base-v1"
    assert [event.sequence for event in events] == list(range(len(events)))
    assert events[-1].state == "COMPLETED"


def test_worker_stops_before_queue_on_budget_or_cancellation(tmp_path: Path) -> None:
    registry = TrainerRegistry()
    registry.register(FakeTrainer(tmp_path, "fake-yolo"))
    too_small = config().model_copy(update={"max_gpu_hours": 0.01})

    budget = ComputeWorker(registry).run(too_small)
    cancelled = ComputeWorker(registry, cancel_check=lambda: True).run(config())

    assert budget.state == "BUDGET_EXCEEDED"
    assert cancelled.state == "CANCELLED"
