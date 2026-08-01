"""Optional YOLO11 surrogate with a differentiable raw-output objective."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

import numpy as np

from src.adapters import MODELS
from src.adapters.base import ModelAdapter
from src.core.objectives import AttackObjective, SurrogateCapability
from src.core.types import Box, ModelInfo, Prediction, Sample

COCO_MAP = {0: "Pedestrian", 1: "Cyclist", 2: "Car"}
COCO_REVERSE = {label: class_id for class_id, label in COCO_MAP.items()}


@MODELS.register
class Yolo11Adapter(ModelAdapter):
    """Ultralytics YOLO11 detector loaded lazily for attack generation."""

    name: ClassVar[str] = "yolo11"
    version: ClassVar[str] = "ultralytics-yolo11-adversarial-v4"
    supports_gradients: ClassVar[bool] = True
    capabilities: ClassVar[frozenset[SurrogateCapability]] = frozenset(
        {
            "input_gradient",
            "detection_loss",
            "objectness",
            "class_logits",
            "class_margin",
        }
    )
    owner: ClassVar[str] = "group-d-e"

    def __init__(
        self,
        *,
        weights: str,
        device: str = "cpu",
        score_threshold: float = 0.25,
        max_detections: int = 100,
        image_size: int = 640,
    ) -> None:
        super().__init__(
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
        self.weights = weights
        self.device = device
        self.image_size = image_size
        self._backend: Any | None = None

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            task="detection2d",
            version=f"{self.version}:{self.weights}",
            supports_gradients=True,
        )

    def predict(self, samples: Sequence[Sample]) -> list[Prediction]:
        backend = self._load()
        predictions: list[Prediction] = []
        for sample in samples:
            started = perf_counter()
            results = backend.predict(
                source=np.rint(sample.image * 255.0).astype(np.uint8),
                device=self.device,
                imgsz=self.image_size,
                conf=min(self.score_threshold, 0.001),
                max_det=self.max_detections,
                verbose=False,
            )
            converted: list[Box] = []
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confidence = boxes.conf.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                for coordinates, score, class_id in zip(xyxy, confidence, classes, strict=True):
                    label = COCO_MAP.get(int(class_id))
                    if label is not None:
                        converted.append(
                            Box(
                                float(coordinates[0]),
                                float(coordinates[1]),
                                float(coordinates[2]),
                                float(coordinates[3]),
                                label,
                                float(score),
                            )
                        )
            predictions.append(
                Prediction(
                    sample.sample_id,
                    self.postprocess(converted),
                    (perf_counter() - started) * 1000.0,
                )
            )
        return predictions

    def loss_for_attack(
        self,
        sample: Sample,
        target: AttackObjective | Any | None = None,
    ) -> float:
        _, objective = self._raw_objective(sample, target, requires_grad=False)
        return float(objective.detach().cpu())

    def input_gradient(
        self,
        sample: Sample,
        target: AttackObjective | Any | None = None,
    ) -> np.ndarray:
        tensor, objective = self._raw_objective(sample, target, requires_grad=True)
        objective.backward()
        if tensor.grad is None:
            raise RuntimeError("YOLO surrogate produced no input gradient")
        return (
            tensor.grad[0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def _raw_objective(
        self,
        sample: Sample,
        target: AttackObjective | Any | None,
        *,
        requires_grad: bool,
    ) -> tuple[Any, Any]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YOLO attack generation requires torch") from exc
        backend = self._load()
        tensor = (
            torch.from_numpy(sample.image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        tensor.requires_grad_(requires_grad)
        model_input = _letterbox_tensor(tensor, self.image_size)
        raw = backend.model(model_input)
        prediction, class_logits = _prediction_and_logits(raw)
        class_scores = prediction[:, 4:, :]
        if class_scores.shape[1] < 3:
            raise RuntimeError(
                f"YOLO prediction has no class-score channels: {tuple(prediction.shape)}"
            )
        if class_logits is None:
            class_logits = _probabilities_to_logits(class_scores)
        objectness_logits = class_logits.max(dim=1).values
        kind = target.kind if isinstance(target, AttackObjective) else "untargeted"
        if kind == "fabrication":
            objective = objectness_logits.mean()
        elif kind in {"targeted", "mislabeling", "cw_margin"}:
            selected_target = (
                target if isinstance(target, AttackObjective) else AttackObjective()
            )
            objective = _classification_margin(
                prediction,
                class_logits,
                sample,
                selected_target,
                self.image_size,
            )
        else:
            selected_target = (
                target if isinstance(target, AttackObjective) else AttackObjective()
            )
            focused = _selected_true_scores(
                prediction,
                class_logits,
                sample,
                selected_target,
                self.image_size,
            )
            objective = (
                -focused.mean()
                if focused is not None
                else -objectness_logits.mean()
            )
        return tensor, objective

    def _load(self) -> Any:
        if self._backend is None:
            checkpoint = Path(self.weights).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"YOLO checkpoint does not exist; automatic download is disabled: "
                    f"{checkpoint}"
                )
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "adapter 'yolo11' requires optional dependency: pip install ultralytics"
                ) from exc
            self._backend = YOLO(str(checkpoint))
            self._backend.model.eval()
            for parameter in self._backend.model.parameters():
                parameter.requires_grad_(False)
        return self._backend


def _collect_tensors(value: Any) -> list[Any]:
    if hasattr(value, "requires_grad") and hasattr(value, "float"):
        return [value]
    if isinstance(value, dict):
        return [
            tensor
            for nested in value.values()
            for tensor in _collect_tensors(nested)
        ]
    if isinstance(value, (list, tuple)):
        return [tensor for nested in value for tensor in _collect_tensors(nested)]
    return []


def _prediction_and_logits(raw: Any) -> tuple[Any, Any | None]:
    inference = raw[0] if isinstance(raw, tuple) and raw else raw
    prediction = next(
        (value for value in _collect_tensors(inference) if value.ndim == 3),
        None,
    )
    if prediction is None:
        raise RuntimeError("YOLO backend returned no differentiable prediction tensor")
    if prediction.shape[1] > prediction.shape[2]:
        prediction = prediction.transpose(1, 2)

    class_logits = _find_score_logits(raw[1] if isinstance(raw, tuple) else None)
    if class_logits is not None:
        if class_logits.shape[1] > class_logits.shape[2]:
            class_logits = class_logits.transpose(1, 2)
        if class_logits.shape[2] != prediction.shape[2]:
            class_logits = None
    return prediction, class_logits


def _find_score_logits(value: Any) -> Any | None:
    if isinstance(value, dict):
        scores = value.get("scores")
        if scores is not None and getattr(scores, "ndim", None) == 3:
            return scores
        for nested in value.values():
            found = _find_score_logits(nested)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for nested in value:
            found = _find_score_logits(nested)
            if found is not None:
                return found
    return None


def _letterbox_tensor(tensor: Any, image_size: int) -> Any:
    """Resize and pad without breaking the gradient to the original image."""
    import torch.nn.functional as functional

    height, width = tensor.shape[-2:]
    resized_height, resized_width, top, bottom, left, right, _ = _letterbox_geometry(
        height,
        width,
        image_size,
    )
    resized = functional.interpolate(
        tensor,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    return functional.pad(
        resized,
        (left, right, top, bottom),
        mode="constant",
        value=114 / 255,
    )


def _classification_margin(
    prediction: Any,
    class_logits: Any,
    sample: Sample,
    target: AttackObjective,
    image_size: int,
) -> Any:
    """Signed detector margin; positive means the requested misclassification won."""
    import torch

    pairs = _proposal_pairs(prediction, sample, target, image_size)
    if not pairs:
        top_two = class_logits.topk(k=2, dim=1).values
        return (top_two[:, 1] - top_two[:, 0]).mean()
    requested_index = COCO_REVERSE.get(target.target_label or "")
    margins = []
    for proposal_index, true_index in pairs:
        scores = class_logits[0, :, proposal_index]
        if requested_index is None:
            alternatives = scores.clone()
            alternatives[true_index] = float("-inf")
            margins.append(alternatives.max() - scores[true_index])
        else:
            alternatives = scores.clone()
            alternatives[requested_index] = float("-inf")
            margins.append(scores[requested_index] - alternatives.max())
    return margins[0] if len(margins) == 1 else torch.stack(margins).mean()


def _selected_true_scores(
    prediction: Any,
    class_logits: Any,
    sample: Sample,
    target: AttackObjective,
    image_size: int,
) -> Any | None:
    import torch

    pairs = _proposal_pairs(
        prediction,
        sample,
        target,
        image_size,
        label_filter=target.target_label,
    )
    if not pairs:
        return None
    values = [
        class_logits[0, class_index, proposal_index]
        for proposal_index, class_index in pairs
    ]
    return values[0].reshape(1) if len(values) == 1 else torch.stack(values)


def _proposal_pairs(
    prediction: Any,
    sample: Sample,
    target: AttackObjective,
    image_size: int,
    *,
    label_filter: str | None = None,
) -> list[tuple[int, int]]:
    import torch

    candidates = [
        (index, box)
        for index, box in enumerate(sample.boxes)
        if box.label in COCO_REVERSE
        and (label_filter is None or box.label == label_filter)
        and (target.target_box_index is None or index == target.target_box_index)
    ]
    if not candidates:
        return []
    height, width = sample.image.shape[:2]
    _, _, top, _, left, _, scale = _letterbox_geometry(height, width, image_size)
    predicted_xywh = prediction[0, :4, :].transpose(0, 1)
    predicted_xyxy = torch.stack(
        (
            predicted_xywh[:, 0] - predicted_xywh[:, 2] / 2,
            predicted_xywh[:, 1] - predicted_xywh[:, 3] / 2,
            predicted_xywh[:, 0] + predicted_xywh[:, 2] / 2,
            predicted_xywh[:, 1] + predicted_xywh[:, 3] / 2,
        ),
        dim=1,
    )
    ground_truth = predicted_xyxy.new_tensor(
        [
            (
                box.x1 * scale + left,
                box.y1 * scale + top,
                box.x2 * scale + left,
                box.y2 * scale + top,
            )
            for _, box in candidates
        ]
    )
    overlaps = _box_iou(ground_truth, predicted_xyxy)
    proposal_indices = overlaps.argmax(dim=1)
    return [
        (int(proposal_index), COCO_REVERSE[box.label])
        for proposal_index, (_, box) in zip(
            proposal_indices.detach().cpu().tolist(),
            candidates,
            strict=True,
        )
    ]


def _box_iou(left: Any, right: Any) -> Any:
    import torch

    top_left = torch.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    left_area = (left[:, 2] - left[:, 0]).clamp(min=0) * (
        left[:, 3] - left[:, 1]
    ).clamp(min=0)
    right_area = (right[:, 2] - right[:, 0]).clamp(min=0) * (
        right[:, 3] - right[:, 1]
    ).clamp(min=0)
    return intersection_area / (
        left_area[:, None] + right_area[None, :] - intersection_area
    ).clamp(min=1e-12)


def _probabilities_to_logits(class_scores: Any) -> Any:
    import torch

    epsilon = torch.finfo(class_scores.dtype).tiny
    clipped = class_scores.clamp(min=epsilon, max=1.0 - 1e-6)
    return torch.log(clipped) - torch.log1p(-clipped)


def _letterbox_geometry(
    height: int,
    width: int,
    image_size: int,
) -> tuple[int, int, int, int, int, int, float]:
    scale = min(image_size / height, image_size / width)
    resized_height = max(1, int(round(height * scale)))
    resized_width = max(1, int(round(width * scale)))
    vertical = image_size - resized_height
    horizontal = image_size - resized_width
    left = horizontal // 2
    right = horizontal - left
    top = vertical // 2
    bottom = vertical - top
    return resized_height, resized_width, top, bottom, left, right, scale
