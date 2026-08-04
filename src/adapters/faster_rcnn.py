"""Optional torchvision Faster R-CNN surrogate for DAG generation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

import numpy as np

from src.adapters import MODELS
from src.adapters.base import ModelAdapter
from src.core.hashing import file_digest
from src.core.objectives import AttackObjective, SurrogateCapability
from src.core.types import Box, ModelInfo, Prediction, Sample

COCO_MAP = {1: "Pedestrian", 2: "Cyclist", 3: "Car"}
COCO_REVERSE = {"Pedestrian": 1, "Cyclist": 2, "Car": 3}


@MODELS.register
class FasterRcnnAdapter(ModelAdapter):
    """Torchvision Faster R-CNN R50-FPN surrogate with dense proposal gradients."""

    name: ClassVar[str] = "faster_rcnn"
    version: ClassVar[str] = "torchvision-fasterrcnn-r50-fpn"
    supports_gradients: ClassVar[bool] = True
    capabilities: ClassVar[frozenset[SurrogateCapability]] = frozenset(
        {"input_gradient", "detection_loss", "dense_proposals"}
    )
    owner: ClassVar[str] = "group-d-e"

    def __init__(
        self,
        *,
        weights: str,
        device: str = "cpu",
        score_threshold: float = 0.25,
        max_detections: int = 100,
    ) -> None:
        super().__init__(
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
        self.weights = weights
        self.device = device
        self._backend: Any | None = None

    def metadata(self) -> ModelInfo:
        checkpoint = Path(self.weights).expanduser()
        return ModelInfo(
            name=self.name,
            task="detection2d",
            version=f"{self.version}:{self.weights}",
            supports_gradients=True,
            capabilities=self.capabilities,
            checkpoint_hash=file_digest(checkpoint) if checkpoint.is_file() else None,
            preprocessing_version="torchvision-default-transform-v1",
        )

    def predict(self, samples: Sequence[Sample]) -> list[Prediction]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Faster R-CNN adapter requires torch") from exc
        model = self._load().eval()
        predictions: list[Prediction] = []
        with torch.no_grad():
            for sample in samples:
                started = perf_counter()
                tensor = torch.from_numpy(sample.image).permute(2, 0, 1).to(self.device)
                raw = model([tensor])[0]
                boxes: list[Box] = []
                for coordinates, label_id, score in zip(
                    raw["boxes"].cpu().numpy(),
                    raw["labels"].cpu().numpy(),
                    raw["scores"].cpu().numpy(),
                    strict=True,
                ):
                    label = COCO_MAP.get(int(label_id))
                    if label is not None:
                        boxes.append(Box(*map(float, coordinates), label, float(score)))
                predictions.append(
                    Prediction(
                        sample.sample_id,
                        self.postprocess(boxes),
                        (perf_counter() - started) * 1000.0,
                    )
                )
        return predictions

    def loss_for_attack(
        self,
        sample: Sample,
        target: AttackObjective | Any | None = None,
    ) -> float:
        _, loss = self._dense_loss(sample, requires_grad=False)
        return float(loss.detach().cpu())

    def input_gradient(
        self,
        sample: Sample,
        target: AttackObjective | Any | None = None,
    ) -> np.ndarray:
        tensor, loss = self._dense_loss(sample, requires_grad=True)
        loss.backward()
        if tensor.grad is None:
            raise RuntimeError("Faster R-CNN surrogate produced no input gradient")
        return tensor.grad.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    def _dense_loss(self, sample: Sample, *, requires_grad: bool) -> tuple[Any, Any]:
        try:
            import torch
            import torch.nn.functional as functional
            from torchvision.ops import box_iou
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Faster R-CNN adapter requires torch") from exc
        if not sample.boxes:
            raise ValueError("DAG requires ground-truth boxes")
        model = self._load().eval()
        tensor = torch.from_numpy(sample.image).permute(2, 0, 1).to(self.device)
        tensor.requires_grad_(requires_grad)
        targets = [
            {
                "boxes": torch.tensor(
                    [box.as_tuple() for box in sample.boxes],
                    dtype=torch.float32,
                    device=self.device,
                ),
                "labels": torch.tensor(
                    [COCO_REVERSE[box.label] for box in sample.boxes],
                    dtype=torch.int64,
                    device=self.device,
                ),
            }
        ]
        images, transformed_targets = model.transform([tensor], targets)
        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        proposals, _ = model.rpn(images, features, transformed_targets)
        pooled = model.roi_heads.box_roi_pool(
            features,
            proposals,
            images.image_sizes,
        )
        representation = model.roi_heads.box_head(pooled)
        class_logits, _ = model.roi_heads.box_predictor(representation)
        ground_truth = transformed_targets[0]
        overlap = box_iou(proposals[0], ground_truth["boxes"])
        best_overlap, assignment = overlap.max(dim=1)
        active = best_overlap > 0.1
        if not bool(active.any()):
            raise RuntimeError("Faster R-CNN produced no proposals overlapping ground truth")
        true_labels = ground_truth["labels"][assignment[active]]
        loss = functional.cross_entropy(class_logits[active], true_labels)
        return tensor, loss

    def _load(self) -> Any:
        if self._backend is None:
            checkpoint = Path(self.weights).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Faster R-CNN checkpoint does not exist; automatic download is disabled: {checkpoint}"
                )
            try:
                import torch
                from torchvision.models.detection import (
                    fasterrcnn_resnet50_fpn,
                )
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("adapter 'faster_rcnn' requires torch and torchvision") from exc
            self._backend = fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone=None,
            )
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self._backend.load_state_dict(state)
            self._backend.to(self.device)
        return self._backend
