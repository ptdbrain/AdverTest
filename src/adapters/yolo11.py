"""M1: YOLO11 via Ultralytics — the 1-stage detection baseline of plan §1.2.

Importing this module is free: ``torch`` and ``ultralytics`` are only touched
inside :meth:`Yolo11Adapter._load`, so the catalog, the API and CI keep working
with no GPU, no weights and no network.

Two things that quietly cost AP if you get them wrong:

* **Colour order.** Ultralytics interprets a raw ``HWC`` ndarray source as
  **BGR** (it is an OpenCV-shaped API), while :class:`~src.core.types.Sample`
  carries RGB float32 in ``[0, 1]``. :meth:`to_backend_image` does the swap and
  the ``uint8`` conversion in one place. Feeding RGB straight in does not raise —
  it just makes the model a few points worse and the robustness numbers wrong.
* **Label space.** COCO weights do not speak KITTI. ``car -> Car`` and
  ``person -> Pedestrian`` are clean, but KITTI's ``Cyclist`` is a *single* box
  around rider and bike, whereas COCO emits a separate ``person`` and
  ``bicycle``. Nothing maps that cleanly, so ``AP_clean`` is depressed on the
  Cyclist class until the model is fine-tuned (``scripts/train_yolo11_kitti.py``).
  Degradation ``D(c, s)`` is a ratio against this model's own clean AP, so group
  C results stay meaningful either way — absolute AP does not.

``version`` carries the weights, image size and confidence threshold, so a
fine-tuned checkpoint automatically gets its own namespace in the content-addressed
prediction cache instead of silently reusing the COCO run's predictions.

Gradients (``loss_for_attack`` / ``input_gradient``) are **not** implemented yet:
the adapter reports ``supports_gradients = False``, so the runner records a skip
reason for group D/E attacks rather than failing. Wiring them to
``ultralytics.utils.loss.v8DetectionLoss`` is the next slot on this file.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

import numpy as np

from src.adapters import MODELS
from src.adapters.base import ModelAdapter
from src.core.types import Box, ModelInfo, Prediction, Sample, Task

#: COCO class name -> normalised class (plan §1.1). Missing names are dropped.
COCO_LABEL_MAP: dict[str, str] = {
    "car": "Car",
    "person": "Pedestrian",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
}
#: Enabled by ``map_truck_bus_to_car``: KITTI keeps Truck separate, COCO does not.
LARGE_VEHICLE_ALIASES: dict[str, str] = {"truck": "Car", "bus": "Car"}

_MISSING_ULTRALYTICS = (
    "adapter 'yolo11' needs optional extras: pip install ultralytics torch "
    "(kept out of requirements.txt so CI stays numpy-only)"
)


@MODELS.register
class Yolo11Adapter(ModelAdapter):
    """YOLO11 (Ultralytics) 2D detector, mapped onto the normalised classes."""

    name: ClassVar[str] = "yolo11"
    task: ClassVar[Task] = "detection2d"
    version: ClassVar[str] = "yolo11-1.0.0"
    supports_gradients: ClassVar[bool] = False
    owner: ClassVar[str] = "phong"

    def __init__(
        self,
        *,
        weights: str = "yolo11s.pt",
        imgsz: int = 640,
        score_threshold: float = 0.25,
        max_detections: int = 100,
        nms_iou: float = 0.7,
        batch_size: int = 8,
        device: str | None = None,
        #: Opt-in FP16. Off by default — read :meth:`_use_half` before turning it on.
        half: bool = False,
        map_truck_bus_to_car: bool = False,
    ) -> None:
        super().__init__(score_threshold=score_threshold, max_detections=max_detections)
        self.weights = weights
        self.imgsz = imgsz
        self.nms_iou = nms_iou
        self.batch_size = max(1, batch_size)
        self.device = device
        self.half = half
        self.map_truck_bus_to_car = map_truck_bus_to_car
        self._model: Any | None = None

    # ------------------------------------------------------------------ catalog

    def metadata(self) -> ModelInfo:
        """Must work with no weights on disk: the catalog calls it per request."""
        stem = Path(self.weights).stem
        return ModelInfo(
            name=self.name,
            task=self.task,
            version=f"{self.version}:{stem}:imgsz{self.imgsz}:conf{self.score_threshold:.2f}",
            supports_gradients=self.supports_gradients,
        )

    # ---------------------------------------------------------------- inference

    def predict(self, samples: Sequence[Sample]) -> list[Prediction]:
        model = self._load()
        predictions: list[Prediction] = []
        for chunk in self._chunks(samples):
            started = perf_counter()
            results = model.predict(
                [self.to_backend_image(sample.image) for sample in chunk],
                imgsz=self.imgsz,
                conf=self.score_threshold,
                iou=self.nms_iou,
                device=self.device,
                verbose=False,
                **self._precision_kwargs(),
            )
            elapsed_ms = (perf_counter() - started) * 1000.0 / max(1, len(chunk))
            for sample, result in zip(chunk, results, strict=True):
                boxes = self.postprocess(self.convert(self._rows(result)))
                predictions.append(Prediction(sample.sample_id, boxes, elapsed_ms))
        return predictions

    def _chunks(self, samples: Sequence[Sample]) -> Iterable[Sequence[Sample]]:
        """Fixed-size batches — 6 GB of VRAM does not fit a 500-image call."""
        for start in range(0, len(samples), self.batch_size):
            yield samples[start : start + self.batch_size]

    @staticmethod
    def _rows(result: Any) -> list[tuple[float, float, float, float, str, float]]:
        """Ultralytics ``Results`` -> plain rows, so :meth:`convert` stays testable."""
        names = result.names
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        return [
            (float(x1), float(y1), float(x2), float(y2), str(names[int(cls)]), float(score))
            for (x1, y1, x2, y2), cls, score in zip(xyxy, classes, confidences, strict=True)
        ]

    # ------------------------------------------------------------- pure helpers

    @staticmethod
    def to_backend_image(image: np.ndarray) -> np.ndarray:
        """RGB float32 ``[0, 1]`` -> BGR ``uint8``, the layout Ultralytics expects."""
        scaled = np.clip(image, 0.0, 1.0) * 255.0
        return np.ascontiguousarray(scaled.round().astype(np.uint8)[..., ::-1])

    def map_label(self, raw_label: str) -> str | None:
        """COCO class name -> normalised class, or ``None`` when it is dropped."""
        key = raw_label.lower()
        if key in COCO_LABEL_MAP:
            return COCO_LABEL_MAP[key]
        if self.map_truck_bus_to_car and key in LARGE_VEHICLE_ALIASES:
            return LARGE_VEHICLE_ALIASES[key]
        return None

    def convert(self, rows: Iterable[tuple[float, float, float, float, str, float]]) -> list[Box]:
        """Detector rows -> :class:`Box` in the normalised label space."""
        boxes: list[Box] = []
        for x1, y1, x2, y2, raw_label, score in rows:
            label = self.map_label(raw_label)
            if label is not None and x2 > x1 and y2 > y1:
                boxes.append(Box(float(x1), float(y1), float(x2), float(y2), label, float(score)))
        return boxes

    # -------------------------------------------------------------- lazy loading

    def _use_half(self) -> bool:
        """FP16 is **opt-in**, not the default. See :attr:`half` and the note below.

        Plan §5 lists FP16 as a throughput win, and it is — but on this project's
        reference GPU (GTX 1660 Ti, Turing) with torch 2.13+cu130 and ultralytics
        8.4.113, half-precision inference returns *wrong results* for most batch
        sizes rather than failing loudly. Measured on one image repeated N times,
        FP32 giving 5 detections every time:

            batch 1 -> 5    batch 2 -> 0    batch 3 -> 5
            batch 4 -> 0    batch 5 -> 9    batch 8 -> 0

        Zero detections would zero the AP; nine would inflate it. Either way every
        robustness number downstream is corrupt, and nothing in the pipeline would
        flag it. Correctness beats throughput here, so the default is FP32.
        Re-enable with ``half=True`` only after checking clean AP against FP32 on
        the same batch size.
        """
        return bool(self.half) and self._cuda_available()

    def _precision_kwargs(self) -> dict[str, Any]:
        """Spell the precision the way the installed Ultralytics expects.

        8.4 replaced ``half=True`` with ``quantize=16`` and warns on every call
        for the old spelling; 8.3 does not know ``quantize`` at all.
        """
        use_half = self._use_half()
        try:
            from ultralytics.cfg import DEFAULT_CFG

            if hasattr(DEFAULT_CFG, "quantize"):
                return {"quantize": 16 if use_half else None}
        except ImportError:  # pragma: no cover - optional extra
            pass
        return {"half": use_half}

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
        except ImportError:  # pragma: no cover - optional extra
            return False
        return bool(torch.cuda.is_available())

    def _load(self) -> Any:
        """Lazy, cached model load; the heavy imports live here and nowhere else."""
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(_MISSING_ULTRALYTICS) from exc
            self._model = YOLO(self.weights)
        return self._model
