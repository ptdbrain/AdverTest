"""COPY ME — template for a model adapter (M1–M6 of plan §1.2).

Skipped by auto-discovery (leading underscore), so it never appears in the
catalog. To add a model:

1. ``cp src/adapters/_template.py src/adapters/<your_model>.py``
2. Rename the class, set ``name`` / ``version`` / ``owner``.
3. Keep the heavy import inside ``_load`` — importing the module must stay free
   so the catalog, the API, and CI work without weights or a GPU.
4. Map the model's labels onto :data:`src.core.types.CLASSES`.
5. ``supports_gradients = True`` only once ``loss_for_attack`` and
   ``input_gradient`` are implemented; group D/E attacks skip adapters that
   report ``False``.

Sketch for a torch detector::

    def input_gradient(self, sample, target=None):
        image = torch.tensor(sample.image).permute(2, 0, 1)[None].requires_grad_(True)
        loss = self._detection_loss(self._model()(image), target)   # higher = worse
        loss.backward()
        return image.grad[0].permute(1, 2, 0).cpu().numpy().astype("float32")
"""

from __future__ import annotations

from collections.abc import Sequence
from time import perf_counter
from typing import Any, ClassVar

import numpy as np

from src.adapters import MODELS
from src.adapters.base import ModelAdapter
from src.core.types import Box, ModelInfo, Prediction, Sample, Task

#: Map the checkpoint's label space onto the normalised classes (plan §1.1).
LABEL_MAP: dict[str, str] = {"car": "Car", "person": "Pedestrian", "bicycle": "Cyclist"}


@MODELS.register
class TemplateAdapter(ModelAdapter):
    """One-line description shown in the model catalog."""

    name: ClassVar[str] = "template_model"
    task: ClassVar[Task] = "detection2d"
    version: ClassVar[str] = "template-0.1.0"
    supports_gradients: ClassVar[bool] = False
    owner: ClassVar[str] = "your-name"

    def __init__(
        self,
        *,
        weights: str = "weights.pt",
        score_threshold: float = 0.25,
        max_detections: int = 100,
    ) -> None:
        super().__init__(score_threshold=score_threshold, max_detections=max_detections)
        self.weights = weights
        self._backend: Any | None = None

    def metadata(self) -> ModelInfo:
        """Must work without loading weights — the catalog calls it on every request."""
        return ModelInfo(
            name=self.name,
            task=self.task,
            version=f"{self.version}:{self.weights}",
            supports_gradients=self.supports_gradients,
        )

    def predict(self, samples: Sequence[Sample]) -> list[Prediction]:
        backend = self._load()
        predictions: list[Prediction] = []
        for sample in samples:
            started = perf_counter()
            raw = backend.infer((sample.image * 255.0).round().astype(np.uint8))
            boxes = self.postprocess(self._convert(raw))
            predictions.append(Prediction(sample.sample_id, boxes, (perf_counter() - started) * 1000.0))
        return predictions

    @staticmethod
    def _convert(raw: Any) -> list[Box]:
        """Backend output -> :class:`~src.core.types.Box` in the normalised label space."""
        boxes: list[Box] = []
        for x1, y1, x2, y2, label, score in raw:
            mapped = LABEL_MAP.get(str(label).lower())
            if mapped is not None:
                boxes.append(Box(float(x1), float(y1), float(x2), float(y2), mapped, float(score)))
        return boxes

    def _load(self) -> Any:
        """Lazy, cached load of the heavy dependency."""
        if self._backend is None:
            try:
                import your_framework  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    f"adapter {self.name!r} needs an optional extra: pip install your-framework"
                ) from exc
            self._backend = your_framework.load(self.weights)
        return self._backend
