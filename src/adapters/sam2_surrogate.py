"""Optional SAM2 surrogate boundary for segmentation PGD."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from src.adapters import MODELS
from src.adapters.base import ModelAdapter
from src.core.objectives import AttackObjective, SurrogateCapability
from src.core.types import ModelInfo, Prediction, Sample


@MODELS.register
class Sam2SurrogateAdapter(ModelAdapter):
    """SAM2.1 Hiera-small adapter requiring a differentiable backend factory."""

    name: ClassVar[str] = "sam2_surrogate"
    task = "segmentation"
    version: ClassVar[str] = "sam2.1-hiera-small"
    supports_gradients: ClassVar[bool] = True
    capabilities: ClassVar[frozenset[SurrogateCapability]] = frozenset(
        {"input_gradient", "segmentation_loss"}
    )
    owner: ClassVar[str] = "group-d-e"

    def __init__(
        self,
        *,
        weights: str,
        config: str = "configs/sam2.1/sam2.1_hiera_s.yaml",
        device: str = "cpu",
        score_threshold: float = 0.25,
        max_detections: int = 100,
    ) -> None:
        super().__init__(
            score_threshold=score_threshold,
            max_detections=max_detections,
        )
        self.weights = weights
        self.config = config
        self.device = device
        self._backend: Any | None = None

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            task="segmentation",
            version=f"{self.version}:{self.weights}",
            supports_gradients=True,
        )

    def predict(self, samples: Sequence[Sample]) -> list[Prediction]:
        raise NotImplementedError(
            "SAM2 surrogate is generation-only; use input_gradient for sam2_pgd"
        )

    def loss_for_attack(
        self,
        sample: Sample,
        target: AttackObjective | Any | None = None,
    ) -> float:
        _, loss = self._loss(sample, requires_grad=False)
        return float(loss.detach().cpu())

    def input_gradient(
        self,
        sample: Sample,
        target: AttackObjective | Any | None = None,
    ) -> np.ndarray:
        tensor, loss = self._loss(sample, requires_grad=True)
        loss.backward()
        if tensor.grad is None:
            raise RuntimeError("SAM2 surrogate produced no input gradient")
        return tensor.grad[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)

    def _loss(self, sample: Sample, *, requires_grad: bool) -> tuple[Any, Any]:
        if sample.mask is None or not sample.boxes:
            raise ValueError("SAM2 PGD requires a ground-truth mask and box prompt")
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("SAM2 surrogate requires torch") from exc
        backend = self._load()
        tensor = (
            torch.from_numpy(sample.image)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        tensor.requires_grad_(requires_grad)
        box = torch.tensor(
            [sample.boxes[0].as_tuple()],
            dtype=torch.float32,
            device=self.device,
        )
        logits = backend.forward_image_with_box(tensor, box)
        target = torch.from_numpy((sample.mask > 0).astype(np.float32)).to(self.device)
        target = target.unsqueeze(0).unsqueeze(0)
        if logits.shape[-2:] != target.shape[-2:]:
            target = functional.interpolate(target, size=logits.shape[-2:], mode="nearest")
        return tensor, functional.binary_cross_entropy_with_logits(logits, target)

    def _load(self) -> Any:
        if self._backend is None:
            checkpoint = Path(self.weights).expanduser().resolve()
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"SAM2 checkpoint does not exist; automatic download is disabled: "
                    f"{checkpoint}"
                )
            try:
                from sam2.build_sam import build_sam2
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "adapter 'sam2_surrogate' requires the official SAM2 package"
                ) from exc
            model = build_sam2(self.config, str(checkpoint), device=self.device)
            if not hasattr(model, "forward_image_with_box"):
                raise RuntimeError(
                    "installed SAM2 backend lacks differentiable "
                    "forward_image_with_box(image, box); install the AdverTest bridge"
                )
            self._backend = model
        return self._backend
