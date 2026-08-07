"""Versioned SAM2 training configuration and preflight checks.

Execution is deliberately deferred until the shared ``ModelTrainer`` worker is
landed by Người D; this module owns all SAM-specific decisions so that worker
integration is mechanical rather than a second design exercise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


Stage = Literal["b0", "r1", "r2"]


@dataclass(frozen=True, slots=True)
class Sam2TrainingConfig:
    stage: Stage
    dataset_manifest_id: str
    parent_model_version: str | None = None
    epochs: int = 15
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation: int = 1
    amp: bool = True
    freeze_image_encoder: bool = True
    unfreeze_final_encoder_blocks: int = 0
    early_stopping_patience: int = 5
    seed: int = 20260807
    clean_ratio: float = 1.0
    weather_noise_ratio: float = 0.0
    blur_compression_ratio: float = 0.0
    occlusion_ratio: float = 0.0
    adversarial_ratio: float = 0.0
    general_robust_replay_ratio: float = 0.0
    targeted_replay_ratio: float = 0.0
    target_failure_cluster_ids: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.dataset_manifest_id:
            raise ValueError("SAM2 training requires a leakage-validated dataset manifest")
        if self.stage == "b0" and self.parent_model_version is not None:
            raise ValueError("SAM-B0 is initialized from the official pretrained checkpoint, not a parent version")
        if self.stage in {"r1", "r2"} and not self.parent_model_version:
            raise ValueError(f"SAM-{self.stage.upper()} requires its parent model version")
        if self.stage == "r2" and not self.target_failure_cluster_ids:
            raise ValueError("SAM-R2 requires at least one stable failure cluster")
        if self.epochs < 1 or self.batch_size < 1 or self.gradient_accumulation < 1:
            raise ValueError("epochs, batch_size, and gradient_accumulation must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        ratios = (self.clean_ratio, self.weather_noise_ratio, self.blur_compression_ratio, self.occlusion_ratio, self.adversarial_ratio, self.general_robust_replay_ratio, self.targeted_replay_ratio)
        if any(value < 0 for value in ratios):
            raise ValueError("SAM training data ratios must be non-negative")
        if self.stage == "r2":
            if any((self.weather_noise_ratio, self.blur_compression_ratio, self.occlusion_ratio, self.adversarial_ratio)) or abs(self.clean_ratio + self.general_robust_replay_ratio + self.targeted_replay_ratio - 1.0) > 1e-6:
                raise ValueError("SAM-R2 uses clean, general robust, and targeted replay ratios only")
        elif abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("SAM training data ratios must sum to 1.0")
        if self.clean_ratio < (0.45 if self.stage == "r1" else 0.15 if self.stage == "r2" else 1.0):
            raise ValueError("clean replay ratio violates the SAM safety floor")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def sam_b0_config(dataset_manifest_id: str, **overrides: object) -> Sam2TrainingConfig:
    return Sam2TrainingConfig(stage="b0", dataset_manifest_id=dataset_manifest_id, epochs=15, **overrides)


def sam_r1_config(dataset_manifest_id: str, parent_model_version: str, **overrides: object) -> Sam2TrainingConfig:
    return Sam2TrainingConfig(
        stage="r1", dataset_manifest_id=dataset_manifest_id, parent_model_version=parent_model_version,
        epochs=20, clean_ratio=0.45, weather_noise_ratio=0.25, blur_compression_ratio=0.10,
        occlusion_ratio=0.15, adversarial_ratio=0.05, **overrides,
    )


def sam_r2_config(dataset_manifest_id: str, parent_model_version: str, failure_cluster_ids: tuple[str, ...], **overrides: object) -> Sam2TrainingConfig:
    return Sam2TrainingConfig(
        stage="r2", dataset_manifest_id=dataset_manifest_id, parent_model_version=parent_model_version,
        clean_ratio=0.15, general_robust_replay_ratio=0.60, targeted_replay_ratio=0.25,
        target_failure_cluster_ids=failure_cluster_ids, **overrides,
    )


class Sam2Trainer:
    """SAM-specific configuration preflight for the shared trainer worker."""

    version = "sam2-trainer-contract-v1"

    def validate_config(self, config: Sam2TrainingConfig) -> None:
        config.validate()

    def estimate(self, config: Sam2TrainingConfig, *, samples: int = 0) -> dict[str, int | bool | str]:
        config.validate()
        return {
            "trainer_version": self.version,
            "epochs": config.epochs,
            "effective_batch_size": config.batch_size * config.gradient_accumulation,
            "estimated_sample_updates": samples * config.epochs,
            "amp": config.amp,
            "freeze_image_encoder": config.freeze_image_encoder,
        }
