"""Runnable promptable SAM2.1 Hiera Small inference adapter.

The official ``SAM2ImagePredictor`` is used for inference.  White-box input
gradients additionally require the explicitly versioned AdverTest
``forward_image_with_box`` bridge; standard upstream SAM2 installs remain valid
for benchmark inference but cannot silently claim SAM-PGD support.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

import numpy as np

from src.adapters import MODELS
from src.adapters.base import GradientsNotSupportedError, ModelAdapter
from src.core.hashing import file_digest
from src.core.objectives import AttackObjective, SurrogateCapability
from src.core.types import MaskPrediction, ModelInfo, ModelPrediction, Sample, SegmentationPrediction
from src.segmentation.protocol import GroundTruthBoxPrompt, fixed_box_prompts


@MODELS.register
class Sam2Adapter(ModelAdapter):
    """SAM2.1 Hiera Small with fixed box prompts at original image resolution."""

    name: ClassVar[str] = "sam2"
    task = "segmentation"
    version: ClassVar[str] = "sam2.1-hiera-small-v1"
    supports_gradients: ClassVar[bool] = True
    capabilities: ClassVar[frozenset[SurrogateCapability]] = frozenset({"segmentation_loss", "input_gradient"})
    owner: ClassVar[str] = "group-c"

    def __init__(
        self,
        *,
        weights: str,
        config: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        device: str = "cpu",
        mask_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.config = config
        self.device = device
        self.mask_threshold = mask_threshold
        self._model: Any | None = None
        self._predictor: Any | None = None

    def metadata(self) -> ModelInfo:
        checkpoint = Path(self.weights).expanduser()
        return ModelInfo(
            name=self.name,
            task="segmentation",
            version=f"{self.version}:{checkpoint.stem}",
            supports_gradients=True,
            capabilities=self.capabilities,
            checkpoint_hash=file_digest(checkpoint) if checkpoint.is_file() else None,
            preprocessing_version="sam2-rgb-uint8-original-resolution-box-prompt-v1",
            runnable=True,
        )

    def predict(self, samples: Sequence[Sample]) -> list[ModelPrediction]:
        """Run the generic benchmark path with manifest-persisted GT prompts."""
        return list(self.predict_masks(samples, [fixed_box_prompts(sample) for sample in samples]))

    def predict_masks(
        self,
        samples: Sequence[Sample],
        prompts: Sequence[Sequence[GroundTruthBoxPrompt]],
    ) -> list[SegmentationPrediction]:
        if len(samples) != len(prompts):
            raise ValueError("provide one prompt collection per sample")
        predictor = self._load_predictor()
        results: list[SegmentationPrediction] = []
        for sample, sample_prompts in zip(samples, prompts, strict=True):
            if not sample_prompts:
                raise ValueError("SAM2 requires at least one fixed benchmark prompt per sample")
            started = perf_counter()
            predictor.set_image(self.to_backend_image(sample.image))
            instances: list[MaskPrediction] = []
            for prompt in sample_prompts:
                output_masks, output_scores, _ = predictor.predict(
                    box=np.asarray(prompt.coordinates, dtype=np.float32), multimask_output=False
                )
                if len(output_masks) != 1 or len(output_scores) != 1:
                    raise RuntimeError("SAM2 predictor returned an unexpected number of masks")
                mask = np.asarray(output_masks[0])
                if mask.shape != sample.image.shape[:2]:
                    raise RuntimeError("SAM2 did not return a mask at original sample resolution")
                instances.append(
                    MaskPrediction(
                        instance_id=str(prompt.object_id),
                        mask=(mask > self.mask_threshold).astype(np.bool_),
                        score=float(output_scores[0]),
                    )
                )
            results.append(
                SegmentationPrediction(
                    sample_id=sample.sample_id,
                    instances=tuple(instances),
                    prompt_id=_prompt_set_id(sample_prompts),
                    latency_ms=(perf_counter() - started) * 1000.0,
                    metadata={
                        "model_version_id": self.metadata().version,
                        "preprocessing_version": self.metadata().preprocessing_version,
                        "prompt_coordinates": {prompt.prompt_id: prompt.coordinates for prompt in sample_prompts},
                    },
                )
            )
        return results

    @staticmethod
    def to_backend_image(image: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray((np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8))

    def loss_for_attack(self, sample: Sample, target: AttackObjective | Any | None = None) -> float:
        _, loss = self._gradient_loss(sample, requires_grad=False)
        return float(loss.detach().cpu())

    def input_gradient(self, sample: Sample, target: AttackObjective | Any | None = None) -> np.ndarray:
        tensor, loss = self._gradient_loss(sample, requires_grad=True)
        loss.backward()
        if tensor.grad is None:
            raise RuntimeError("SAM2 differentiable bridge produced no input gradient")
        return tensor.grad[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    def _gradient_loss(self, sample: Sample, *, requires_grad: bool) -> tuple[Any, Any]:
        if sample.mask is None or not sample.boxes:
            raise ValueError("SAM-PGD requires a ground-truth mask and a ground-truth box prompt")
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("SAM-PGD requires torch") from exc
        model = self._load_model()
        if not hasattr(model, "forward_image_with_box"):
            raise GradientsNotSupportedError(
                "installed SAM2 supports inference but lacks the AdverTest differentiable forward_image_with_box bridge"
            )
        tensor = torch.from_numpy(sample.image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        tensor.requires_grad_(requires_grad)
        box = torch.tensor([sample.boxes[0].as_tuple()], dtype=torch.float32, device=self.device)
        logits = model.forward_image_with_box(tensor, box)
        truth = torch.from_numpy((sample.mask > 0).astype(np.float32)).to(self.device)[None, None]
        if logits.shape[-2:] != truth.shape[-2:]:
            truth = functional.interpolate(truth, size=logits.shape[-2:], mode="nearest")
        return tensor, functional.binary_cross_entropy_with_logits(logits, truth)

    def _load_model(self) -> Any:
        if self._model is None:
            checkpoint = Path(self.weights).expanduser().resolve()
            config = Path(self.config).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(f"SAM2 checkpoint does not exist: {checkpoint}")
            if not config.is_file():
                raise FileNotFoundError(f"SAM2 config does not exist: {config}")
            try:
                from sam2.build_sam import build_sam2
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("adapter 'sam2' requires the official SAM2 package") from exc
            self._model = build_sam2(_official_config_name(config), str(checkpoint), device=self.device)
            self._model.eval()
        return self._model

    def _load_predictor(self) -> Any:
        if self._predictor is None:
            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("adapter 'sam2' requires SAM2ImagePredictor from the official SAM2 package") from exc
            self._predictor = SAM2ImagePredictor(self._load_model())
        return self._predictor


def _prompt_set_id(prompts: Sequence[GroundTruthBoxPrompt]) -> str:
    return "|".join(prompt.prompt_id for prompt in prompts)


def _official_config_name(config: Path) -> str:
    """Map a hashable local SAM2.1 config file to the official package resource."""
    names = {
        "sam2.1_hiera_t.yaml": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2.1_hiera_s.yaml": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_b+.yaml": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_l.yaml": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
    try:
        return names[config.name]
    except KeyError as exc:
        raise ValueError(
            f"official SAM2 runtime only supports a matching SAM2.1 Hiera config filename, got {config.name}"
        ) from exc
