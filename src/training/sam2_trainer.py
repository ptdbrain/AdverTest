"""Versioned SAM2 training configuration and preflight checks.

Execution is deliberately deferred until the shared ``ModelTrainer`` worker is
landed by Người D; this module owns all SAM-specific decisions so that worker
integration is mechanical rather than a second design exercise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.adapters.sam2 import Sam2Adapter
from src.core.hashing import file_digest
from src.datasets.cityscapes import CityscapesSegmentationDataset
from src.evaluation.segmentation_metrics import boundary_iou
from src.segmentation.protocol import fixed_box_prompts
from src.training.base import ModelTrainer, TrainerCallbacks
from src.training.contracts import TrainingRunConfig
from src.training.report import (
    CheckpointMetadata,
    ExportedCheckpoint,
    MetricSnapshot,
    PreparedTrainingData,
    TrainerMetadata,
    TrainingEstimate,
    TrainingReport,
    ValidationReport,
)

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
        ratios = (
            self.clean_ratio,
            self.weather_noise_ratio,
            self.blur_compression_ratio,
            self.occlusion_ratio,
            self.adversarial_ratio,
            self.general_robust_replay_ratio,
            self.targeted_replay_ratio,
        )
        if any(value < 0 for value in ratios):
            raise ValueError("SAM training data ratios must be non-negative")
        if self.stage == "r2":
            if (
                any(
                    (
                        self.weather_noise_ratio,
                        self.blur_compression_ratio,
                        self.occlusion_ratio,
                        self.adversarial_ratio,
                    )
                )
                or abs(self.clean_ratio + self.general_robust_replay_ratio + self.targeted_replay_ratio - 1.0) > 1e-6
            ):
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
        stage="r1",
        dataset_manifest_id=dataset_manifest_id,
        parent_model_version=parent_model_version,
        epochs=20,
        clean_ratio=0.45,
        weather_noise_ratio=0.25,
        blur_compression_ratio=0.10,
        occlusion_ratio=0.15,
        adversarial_ratio=0.05,
        **overrides,
    )


def sam_r2_config(
    dataset_manifest_id: str, parent_model_version: str, failure_cluster_ids: tuple[str, ...], **overrides: object
) -> Sam2TrainingConfig:
    return Sam2TrainingConfig(
        stage="r2",
        dataset_manifest_id=dataset_manifest_id,
        parent_model_version=parent_model_version,
        clean_ratio=0.15,
        general_robust_replay_ratio=0.60,
        targeted_replay_ratio=0.25,
        target_failure_cluster_ids=failure_cluster_ids,
        **overrides,
    )


class Sam2Trainer(ModelTrainer):
    """Fine-tune official SAM2.1 Hiera Small with persisted GT-box prompts."""

    version = "sam2-trainer-contract-v1"

    def validate_config(self, config: TrainingRunConfig) -> ValidationReport:
        errors: list[str] = []
        if config.trainer_name != "sam2":
            errors.append("Sam2Trainer requires trainer_name='sam2'")
        stage = config.metadata.get("sam_stage")
        if stage not in {"b0", "r1", "r2"}:
            errors.append("training metadata must set sam_stage to b0, r1, or r2")
        if stage in {"r1", "r2"} and not config.model_version:
            errors.append("SAM robust stages require parent model_version")
        if stage == "r2" and not config.metadata.get("failure_cluster_ids"):
            errors.append("SAM-R2 requires stable failure_cluster_ids metadata")
        return ValidationReport(valid=not errors, errors=tuple(errors))

    def estimate(self, config: TrainingRunConfig) -> TrainingEstimate:
        validation = self.validate_config(config)
        if not validation.valid:
            raise ValueError("invalid SAM2 training config: " + "; ".join(validation.errors))
        accumulation = int(config.metadata.get("gradient_accumulation", 1))
        # Conservative estimate is intentionally explicit until the runtime
        # telemetry calibration is available on the provisioned GPU worker.
        gpu_hours = config.epochs * config.batch_size * accumulation * 0.02
        return TrainingEstimate(
            gpu_hours=gpu_hours,
            storage_bytes=int(config.metadata.get("estimated_storage_bytes", 0)),
            wall_time_seconds=int(gpu_hours * 3600),
        )

    def prepare_data(self, config: TrainingRunConfig) -> PreparedTrainingData:
        manifest_hash = str(config.metadata.get("training_manifest_hash", ""))
        if not manifest_hash:
            raise ValueError("SAM2 training requires training_manifest_hash metadata from TrainingDatasetBuilder")
        return PreparedTrainingData(
            manifest_id=str(config.metadata.get("training_manifest_id", config.split_manifest_id)),
            manifest_hash=manifest_hash,
            lineage_valid=bool(config.metadata.get("lineage_valid", False)),
            leakage_report_id=str(config.metadata.get("leakage_report_id", "")) or None,
        )

    def train(self, config: TrainingRunConfig, callbacks: TrainerCallbacks) -> TrainingReport:
        runtime = _runtime_config(config)
        train_source = _cityscapes_source(runtime, runtime["train_split"])
        validation_source = _cityscapes_source(runtime, runtime["validation_split"])
        try:
            import torch
            import torch.nn.functional as functional
            from sam2.utils.transforms import SAM2Transforms
        except ImportError as exc:  # pragma: no cover - optional runtime guard
            raise RuntimeError("SAM2 training requires the models-gpu or models-cpu extra") from exc

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        adapter = Sam2Adapter(
            weights=runtime["checkpoint"],
            config=runtime["sam_config"],
            device=runtime["device"],
        )
        model = adapter._load_model()
        _configure_trainable_modules(model, freeze_image_encoder=bool(runtime["freeze_image_encoder"]))
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=config.learning_rate,
        )
        cuda = str(runtime["device"]).startswith("cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=cuda and bool(runtime["amp"]))
        transforms = SAM2Transforms(resolution=model.image_size, mask_threshold=0.0)
        best_score, best_metrics, stale = -float("inf"), {}, 0
        output = Path(runtime["output_dir"]).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        for epoch in range(1, config.epochs + 1):
            if callbacks.is_cancelled():
                break
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses: list[float] = []
            sample_count = 0
            for index, sample in enumerate(train_source.iter_samples(), start=1):
                with torch.amp.autocast("cuda", enabled=cuda and bool(runtime["amp"])):
                    logits, targets = _prompt_logits(model, transforms, sample, runtime["device"])
                    bce = functional.binary_cross_entropy_with_logits(logits, targets)
                    probabilities = logits.sigmoid()
                    dice = 1 - (2 * (probabilities * targets).sum(dim=(1, 2, 3)) + 1) / (
                        probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1
                    )
                    loss = (bce + dice.mean()) / int(runtime["gradient_accumulation"])
                scaler.scale(loss).backward()
                sample_count = index
                if index % int(runtime["gradient_accumulation"]) == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()) * int(runtime["gradient_accumulation"]))
                # Release full-resolution activations before decoding the next image.
                del logits, targets, probabilities, dice, loss
                if cuda:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            if sample_count % int(runtime["gradient_accumulation"]):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if cuda:
                    torch.cuda.synchronize()
            metrics = _validation_metrics(model, transforms, validation_source.iter_samples(), runtime["device"])
            metrics["train_loss"] = float(np.mean(losses))
            callbacks.on_epoch(epoch, metrics)
            score = metrics["miou"] + metrics["boundary_iou"]
            if score > best_score:
                best_score, best_metrics, stale = score, metrics, 0
                torch.save(
                    {"model": model.state_dict(), "metadata": _checkpoint_payload(config, runtime, metrics)}, output
                )
            else:
                stale += 1
                if stale >= int(runtime["early_stopping_patience"]):
                    break
        if not output.is_file():
            raise RuntimeError("SAM2 training did not produce a checkpoint")
        checkpoint = CheckpointMetadata(
            path=str(output),
            sha256=file_digest(output, length=64),
            parent_model_version=config.model_version,
            metadata=_checkpoint_payload(config, runtime, best_metrics),
        )
        return TrainingReport(
            run_id=config.run_id, state="TRAINING", checkpoint=checkpoint, resume_metadata=checkpoint.metadata
        )

    def evaluate_checkpoint(self, checkpoint: CheckpointMetadata) -> MetricSnapshot:
        metrics = checkpoint.metadata.get("validation_metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError("SAM2 checkpoint has no validation metrics")
        return MetricSnapshot(metrics={str(name): float(value) for name, value in metrics.items()})

    def export_checkpoint(self, checkpoint: CheckpointMetadata) -> ExportedCheckpoint:
        try:
            import torch

            payload = torch.load(checkpoint.path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise RuntimeError(f"SAM2 checkpoint reload failed: {exc}") from exc
        return ExportedCheckpoint(
            path=checkpoint.path,
            sha256=file_digest(checkpoint.path, length=64),
            load_valid=isinstance(payload, dict) and "model" in payload,
        )

    def metadata(self) -> TrainerMetadata:
        return TrainerMetadata(name="sam2", task="segmentation", version=self.version)


def _runtime_config(config: TrainingRunConfig) -> dict[str, Any]:
    metadata = config.metadata
    required = ("dataset_root", "anonymization_manifest", "sam_checkpoint", "sam_config")
    missing = [name for name in required if not metadata.get(name)]
    if missing:
        raise ValueError(f"SAM2 runtime metadata missing: {', '.join(missing)}")
    return {
        "dataset_root": str(metadata["dataset_root"]),
        "anonymization_manifest": str(metadata["anonymization_manifest"]),
        "checkpoint": str(metadata["sam_checkpoint"]),
        "sam_config": str(metadata["sam_config"]),
        "device": str(metadata.get("device", "cuda")),
        "train_split": str(metadata.get("train_split", "train")),
        "validation_split": str(metadata.get("validation_split", "val")),
        "output_dir": str(metadata.get("output_checkpoint", f"checkpoints/training/{config.run_id}.pt")),
        "gradient_accumulation": int(metadata.get("gradient_accumulation", 1)),
        "amp": bool(metadata.get("amp", True)),
        "freeze_image_encoder": bool(metadata.get("freeze_image_encoder", True)),
        "early_stopping_patience": int(metadata.get("early_stopping_patience", 5)),
        "max_samples": int(metadata["max_samples"]) if metadata.get("max_samples") else None,
    }


def _cityscapes_source(runtime: dict[str, Any], split: str):
    return CityscapesSegmentationDataset(
        root=runtime["dataset_root"],
        split=split,
        anonymization_manifest=runtime["anonymization_manifest"],
        max_samples=runtime["max_samples"],
    )


def _configure_trainable_modules(model: Any, *, freeze_image_encoder: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (model.sam_prompt_encoder, model.sam_mask_decoder):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    if not freeze_image_encoder:
        for parameter in model.image_encoder.parameters():
            parameter.requires_grad_(True)


def _prompt_logits(model: Any, transforms: Any, sample: Any, device: str):
    import torch
    import torch.nn.functional as functional

    pixels = np.ascontiguousarray(np.rint(np.clip(sample.image, 0, 1) * 255).astype(np.uint8))
    input_image = transforms(pixels)[None].to(device)
    backbone = model.forward_image(input_image)
    _, vision_feats, _, _ = model._prepare_backbone_features(backbone)
    sizes = [(256, 256), (128, 128), (64, 64)]
    features = [feature.permute(1, 2, 0).view(1, -1, *size) for feature, size in zip(vision_feats[::-1], sizes[::-1])][
        ::-1
    ]
    prompts = fixed_box_prompts(sample)
    boxes = torch.tensor([prompt.coordinates for prompt in prompts], dtype=torch.float32, device=device)
    boxes = transforms.transform_boxes(boxes, normalize=True, orig_hw=sample.image.shape[:2])
    labels = torch.tensor([[2, 3]], dtype=torch.int, device=device).repeat(boxes.size(0), 1)
    sparse, dense = model.sam_prompt_encoder(points=(boxes, labels), boxes=None, masks=None)
    logits, _, _, _ = model.sam_mask_decoder(
        image_embeddings=features[-1],
        image_pe=model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
        repeat_image=True,
        high_res_features=[feature[0].unsqueeze(0) for feature in features[:-1]],
    )
    targets = torch.from_numpy(
        np.stack([(sample.mask == prompt.object_id) for prompt in prompts]).astype(np.float32)
    ).to(device)[:, None]
    targets = functional.interpolate(targets, size=logits.shape[-2:], mode="nearest")
    return logits, targets


def _validation_metrics(model: Any, transforms: Any, samples: Any, device: str) -> dict[str, float]:
    import torch

    values: list[float] = []
    boundaries: list[float] = []
    model.eval()
    with torch.no_grad():
        for sample in samples:
            logits, targets = _prompt_logits(model, transforms, sample, device)
            predicted = logits.sigmoid() >= 0.5
            for prediction, target in zip(predicted, targets, strict=True):
                prediction_np, target_np = prediction[0].cpu().numpy(), target[0].cpu().numpy().astype(bool)
                union = np.logical_or(prediction_np, target_np).sum()
                values.append(float(np.logical_and(prediction_np, target_np).sum() / union) if union else 1.0)
                boundaries.append(boundary_iou(prediction_np, target_np))
    return {
        "miou": float(np.mean(values)) if values else 0.0,
        "boundary_iou": float(np.mean(boundaries)) if boundaries else 0.0,
    }


def _checkpoint_payload(
    config: TrainingRunConfig, runtime: dict[str, Any], metrics: dict[str, float]
) -> dict[str, Any]:
    return {
        "stage": config.metadata.get("sam_stage"),
        "seed": config.seed,
        "dataset_manifest_hash": config.metadata.get("training_manifest_hash"),
        "source_checkpoint_hash": file_digest(runtime["checkpoint"], length=64),
        "config_hash": file_digest(runtime["sam_config"], length=64),
        "validation_metrics": metrics,
    }
