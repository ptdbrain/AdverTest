"""Reference adapter: a real (if tiny) detector that needs no weights.

Why it exists: every layer of AdverTest — attacks, evaluator, pipeline, agent,
API — must be testable in CI on a laptop, with no GPU, no checkpoints, and no
network. ``blob_detector`` is an honest threshold + connected-components
detector: it finds bright regions, classifies them by aspect ratio, and scores
them by contrast margin and fill ratio. On :class:`~src.datasets.synthetic.SyntheticShapes`
it reaches high clean AP, and it genuinely degrades under noise, weather,
occlusion, and gradient attacks — which is exactly what the plan's sanity checks
(§3) need to be meaningful.

It is **not** a model under test in the product sense; M1–M6 of plan §1.2 are.
Use it as the template for writing those adapters, and as the fast fixture for
unit tests.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from time import perf_counter
from typing import Any, ClassVar

import numpy as np

from src.adapters import MODELS
from src.adapters.base import ModelAdapter
from src.core.image_ops import box_blur, box_slice, luminance, spread_to_channels
from src.core.types import Box, ModelInfo, Prediction, Sample, Task

#: Aspect-ratio cut points separating the three normalised classes.
CAR_MIN_ASPECT = 1.25
PEDESTRIAN_MAX_ASPECT = 0.78


@MODELS.register
class BlobDetector(ModelAdapter):
    """Threshold + connected-components detector used as the CI reference model."""

    name: ClassVar[str] = "blob_detector"
    task: ClassVar[Task] = "detection2d"
    version: ClassVar[str] = "blob-1.0.0"
    supports_gradients: ClassVar[bool] = True
    owner: ClassVar[str] = "core"

    def __init__(
        self,
        *,
        score_threshold: float = 0.25,
        max_detections: int = 100,
        brightness_threshold: float = 0.45,
        min_pixels: int = 24,
        softness: float = 0.05,
        texture_penalty: float = 4.0,
    ) -> None:
        super().__init__(score_threshold=score_threshold, max_detections=max_detections)
        self.brightness_threshold = brightness_threshold
        self.min_pixels = min_pixels
        self.softness = softness
        self.texture_penalty = texture_penalty

    # ------------------------------------------------------------------ predict

    def predict(self, samples: Sequence[Sample]) -> list[Prediction]:
        predictions: list[Prediction] = []
        for sample in samples:
            started = perf_counter()
            boxes = self.postprocess(self._detect(sample.image))
            elapsed_ms = (perf_counter() - started) * 1000.0
            predictions.append(Prediction(sample.sample_id, boxes, elapsed_ms))
        return predictions

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            task="detection2d",
            version=f"{self.version}:thr{self.brightness_threshold:.2f}",
            supports_gradients=True,
        )

    def _detect(self, image: np.ndarray) -> list[Box]:
        gray = luminance(image)
        # Local roughness: how far each pixel deviates from its 3x3 neighbourhood.
        roughness = np.abs(gray - box_blur(gray[..., None], 1)[..., 0])
        mask = gray > self.brightness_threshold
        return [self._to_box(gray, roughness, component) for component in self._components(mask)]

    def _to_box(
        self,
        gray: np.ndarray,
        roughness: np.ndarray,
        component: tuple[np.ndarray, np.ndarray],
    ) -> Box:
        """Turn one pixel component into a scored, labelled box.

        Confidence multiplies three terms, all measured over the *bounding box*:

        * ``margin``     — how far the region sits above the detection threshold,
        * ``fill``       — how solid the region is (mask pixels / box area),
        * ``smoothness`` — how little high-frequency texture the region carries.

        Two deliberate choices, both needed for the degradation numbers to mean
        something:

        * the mean is taken over the whole box, not over the pixels that passed
          the threshold — that would be a selection bias which *rises* with
          noise and would mask the very effect we measure;
        * the smoothness term is what makes pixel-level corruptions move the
          metric at all. A pure box mean is mathematically insensitive to
          zero-mean noise, whereas a real detector loses confidence because its
          features are corrupted. This is a stub standing in for that effect,
          not a claim about any real architecture.
        """
        rows, cols = component
        y1, y2 = int(rows.min()), int(rows.max() + 1)
        x1, x2 = int(cols.min()), int(cols.max() + 1)
        fill = len(rows) / max(1.0, (y2 - y1) * (x2 - x1))
        region_mean = float(gray[y1:y2, x1:x2].mean())
        margin = (region_mean - self.brightness_threshold) / (1.0 - self.brightness_threshold)
        texture = float(roughness[y1:y2, x1:x2].mean())
        smoothness = 1.0 - self.texture_penalty * texture
        score = float(
            np.clip(margin, 0.0, 1.0) * np.clip(fill, 0.0, 1.0) * np.clip(smoothness, 0.0, 1.0)
        )
        return Box(float(x1), float(y1), float(x2), float(y2), self._label(x2 - x1, y2 - y1), score)

    @staticmethod
    def _label(width: float, height: float) -> str:
        aspect = width / max(height, 1e-6)
        if aspect >= CAR_MIN_ASPECT:
            return "Car"
        if aspect <= PEDESTRIAN_MAX_ASPECT:
            return "Pedestrian"
        return "Cyclist"

    def _components(self, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """4-connected components of a boolean mask, filtered by pixel count."""
        height, width = mask.shape
        unvisited = mask.copy()
        found: list[tuple[np.ndarray, np.ndarray]] = []
        for start_y, start_x in np.argwhere(mask):
            if not unvisited[start_y, start_x]:
                continue
            pixels = self._flood(unvisited, int(start_y), int(start_x), height, width)
            if len(pixels) < self.min_pixels:
                continue
            coordinates = np.array(pixels, dtype=np.int64)
            found.append((coordinates[:, 0], coordinates[:, 1]))
        return found

    @staticmethod
    def _flood(unvisited: np.ndarray, start_y: int, start_x: int, height: int, width: int) -> list[tuple[int, int]]:
        """Breadth-first flood fill; clears visited pixels from ``unvisited``."""
        queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
        unvisited[start_y, start_x] = False
        pixels: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            pixels.append((y, x))
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= next_y < height and 0 <= next_x < width and unvisited[next_y, next_x]:
                    unvisited[next_y, next_x] = False
                    queue.append((next_y, next_x))
        return pixels

    # ------------------------------------------------------- white-box interface

    def loss_for_attack(self, sample: Sample, target: Any | None = None) -> float:
        """Object-vanishing objective: how far the target regions sit below threshold.

        ``L = mean over target pixels of (1 - sigmoid((luma - thr) / softness))``.
        Ascending ``L`` darkens the objects until the detector stops firing —
        the differentiable surrogate of "make the blob disappear".

        Note for sanity check #3: this surrogate is separable and monotone per
        pixel, so a single FGSM step already lands on the L-inf corner and PGD
        ties with it on this reference model. On real detection heads PGD is
        expected to win outright, so the check asserts "PGD is not weaker".
        """
        probability, weights = self._vanishing_terms(sample)
        if weights.sum() == 0.0:
            return 0.0
        return float(((1.0 - probability) * weights).sum() / weights.sum())

    def input_gradient(self, sample: Sample, target: Any | None = None) -> np.ndarray:
        """Analytic ``d loss_for_attack / d image`` (same shape as the image)."""
        probability, weights = self._vanishing_terms(sample)
        total = weights.sum()
        if total == 0.0:
            return np.zeros_like(sample.image, dtype=np.float32)
        # d/dluma [1 - sigmoid(z)] = -sigmoid'(z)/softness, sigmoid'(z) = p(1-p)
        gray_gradient = -(probability * (1.0 - probability)) / self.softness * weights / total
        return spread_to_channels(gray_gradient.astype(np.float32))

    def _vanishing_terms(self, sample: Sample) -> tuple[np.ndarray, np.ndarray]:
        """Soft "is bright" map plus the 0/1 mask of pixels the attack targets."""
        gray = luminance(sample.image)
        exponent = np.clip((gray - self.brightness_threshold) / self.softness, -60.0, 60.0)
        probability = 1.0 / (1.0 + np.exp(-exponent))
        return probability.astype(np.float32), self._target_mask(sample)

    @staticmethod
    def _target_mask(sample: Sample) -> np.ndarray:
        """Ground-truth boxes when available, otherwise the whole frame."""
        height, width = sample.image.shape[:2]
        if not sample.boxes:
            return np.ones((height, width), dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.float32)
        for box in sample.boxes:
            region = box_slice(box, height, width)
            if region is not None:
                mask[region] = 1.0
        return mask if mask.sum() > 0 else np.ones((height, width), dtype=np.float32)
