"""KITTI 2D object detection (plan §1.1) — the baseline dataset for benchmarking.

Layout expected under ``root`` (produced by ``scripts/fetch_kitti.sh``)::

    data/kitti/
    ├── ImageSets/{train,val}.txt      # optional: Chen split ids, one per line
    └── training/
        ├── image_2/000000.png ...     # left colour camera
        └── label_2/000000.txt ...     # KITTI label rows

Three things here are decisions, not defaults, and they all move the AP number:

**Label space.** Plan §1.1 normalises everything to ``Car / Pedestrian / Cyclist``
so KITTI and nuScenes can be compared. ``Van``, ``Truck``, ``Tram``, ``Misc``,
``Person_sitting`` and ``DontCare`` are therefore dropped rather than folded in
(``merge_van_truck`` folds the two vehicle classes back into ``Car`` if you want
the looser definition). The per-frame counts of what was dropped land in
``Sample.meta`` so the choice stays auditable instead of invisible.

**Difficulty.** KITTI's own benchmark reports easy / moderate / hard, defined by
box height, occlusion and truncation. Evaluating against *all* boxes — including
25-pixel-tall, mostly-occluded ones — produces an AP that is not comparable to
any published number, so the default is ``moderate``.

**Anonymisation.** KITTI ships un-anonymised. ``DatasetSource.require_anonymized``
is called by the runner *before* :meth:`load` (``src/pipeline/runner.py``), so the
flag is set in the constructor as a promise, and :meth:`load` keeps it by running
:mod:`src.datasets.kitti_anonymize` over every frame it returns. That module is a
**placeholder** — read its warning before trusting the word "anonymised".
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from src.core.hashing import stable_digest
from src.core.types import Box, Modality, Sample
from src.datasets import DATASETS
from src.datasets.base import DatasetInfo, DatasetParams, DatasetSource
from src.datasets.kitti_anonymize import anonymize_placeholder, write_manifest

#: KITTI class -> normalised class (plan §1.1). Missing keys are dropped.
LABEL_MAP: dict[str, str] = {
    "Car": "Car",
    "Pedestrian": "Pedestrian",
    "Cyclist": "Cyclist",
}
#: Extra mapping enabled by ``merge_van_truck``.
VEHICLE_ALIASES: dict[str, str] = {"Van": "Car", "Truck": "Car"}

Difficulty = Literal["all", "easy", "moderate", "hard"]

#: Official KITTI difficulty bands: (min box height px, max occlusion, max truncation).
DIFFICULTY_LIMITS: dict[Difficulty, tuple[float, int, float]] = {
    "easy": (40.0, 0, 0.15),
    "moderate": (25.0, 1, 0.30),
    "hard": (25.0, 2, 0.50),
    "all": (0.0, 3, 1.0),
}

ENV_ROOT = "ADVERTEST_KITTI_ROOT"
DEFAULT_ROOT = "data/kitti"

_MISSING_ROOT = (
    "KITTI root {root!r} not found (expected {expected}). Download it first:\n"
    "    bash scripts/fetch_kitti.sh --subset 500\n"
    "or point the loader at an existing copy with ADVERTEST_KITTI_ROOT=/path/to/kitti"
)
_MISSING_PILLOW = (
    "reading KITTI PNGs needs an optional extra: pip install pillow "
    "(kept out of requirements.txt so CI stays numpy-only)"
)


class KittiParams(DatasetParams):
    """Everything that changes which pixels and which boxes come out of :meth:`load`."""

    root: str = Field(default_factory=lambda: os.environ.get(ENV_ROOT, DEFAULT_ROOT))
    split: Literal["train", "val", "all"] = "val"
    difficulty: Difficulty = "moderate"
    #: ``"off"`` leaves the anonymisation gate closed, which blocks every test run.
    anonymize: Literal["placeholder", "off"] = "placeholder"
    merge_van_truck: bool = False
    #: Explicit id list, used by the benchmark to bootstrap over resampled subsets.
    sample_ids: tuple[str, ...] | None = None
    #: Where the anonymisation manifest is written; ``None`` = ``<root>/anonymization_manifest.json``.
    manifest_path: str | None = None


@DATASETS.register
class Kitti(DatasetSource):
    """KITTI 2D object detection, left colour camera, normalised to 3 classes."""

    name: ClassVar[str] = "kitti"
    #: Class default stays False: KITTI as distributed is *not* anonymised.
    anonymized: ClassVar[bool] = False
    modality: ClassVar[Modality] = "image"
    owner: ClassVar[str] = "phong"
    params_model: ClassVar[type[DatasetParams]] = KittiParams

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        settings: KittiParams = self.params  # type: ignore[assignment]
        # The runner calls require_anonymized() before load(), so the flag has to be
        # decided here; load() is what actually keeps the promise.
        self.anonymized = settings.anonymize != "off"
        self.root = Path(settings.root)
        self.image_dir = self.root / "training" / "image_2"
        self.label_dir = self.root / "training" / "label_2"

    # ------------------------------------------------------------------ catalog

    def info(self) -> DatasetInfo:
        settings: KittiParams = self.params  # type: ignore[assignment]
        note = (
            f"split={settings.split} difficulty={settings.difficulty} "
            f"anonymize={settings.anonymize}"
        )
        if settings.anonymize == "placeholder":
            note += " — PLACEHOLDER anonymiser, not a privacy guarantee (plan §6)"
        if not (self.root / "ImageSets").is_dir():
            note += " — no ImageSets/, using a deterministic 50/50 id split"
        return DatasetInfo(name=self.name, anonymized=self.anonymized, note=note)

    # --------------------------------------------------------------------- load

    def load(self, limit: int | None = None) -> list[Sample]:
        settings: KittiParams = self.params  # type: ignore[assignment]
        self._require_root()
        ids = list(self._ids())
        if limit is not None:
            ids = ids[:limit]
        samples = [self._load_sample(image_id) for image_id in ids]
        if settings.anonymize == "placeholder":
            samples = self._anonymize(samples)
        return samples

    def _require_root(self) -> None:
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(
                _MISSING_ROOT.format(root=str(self.root), expected=f"{self.image_dir}, {self.label_dir}")
            )

    def _ids(self) -> Iterable[str]:
        """Frame ids for the configured split, in a stable order."""
        settings: KittiParams = self.params  # type: ignore[assignment]
        if settings.sample_ids is not None:
            return list(settings.sample_ids)
        split_file = self.root / "ImageSets" / f"{settings.split}.txt"
        if split_file.is_file():
            return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        available = sorted(path.stem for path in self.image_dir.glob("*.png"))
        if settings.split == "all":
            return available
        half = len(available) // 2
        return available[:half] if settings.split == "train" else available[half:]

    def _load_sample(self, image_id: str) -> Sample:
        image = self._read_image(self.image_dir / f"{image_id}.png")
        boxes, dropped = self._read_labels(self.label_dir / f"{image_id}.txt", image.shape[:2])
        return Sample(
            sample_id=self._sample_id(image_id),
            image=image,
            boxes=boxes,
            anonymized=False,
            meta={
                "image_id": image_id,
                "dataset": self.name,
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "dropped_labels": dropped,
            },
        )

    def _sample_id(self, image_id: str) -> str:
        """Unique per *content*, not just per frame.

        The prediction cache is keyed on ``sample_id`` (``src/core/hashing.py``),
        and anonymisation, difficulty and label merging all change what a frame
        contains — so they belong in the id, or a second run would silently reuse
        the first run's predictions.
        """
        settings: KittiParams = self.params  # type: ignore[assignment]
        variant = stable_digest(
            {
                "anonymize": settings.anonymize,
                "difficulty": settings.difficulty,
                "merge_van_truck": settings.merge_van_truck,
            },
            length=6,
        )
        return f"kitti_{variant}_{image_id}"

    # ------------------------------------------------------------------- pixels

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        """PNG -> float32 RGB ``(H, W, 3)`` in ``[0, 1]``."""
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(_MISSING_PILLOW) from exc
        if not path.is_file():
            raise FileNotFoundError(f"KITTI image not found: {path}")
        with Image.open(path) as handle:
            pixels = np.asarray(handle.convert("RGB"), dtype=np.float32)
        return np.ascontiguousarray(pixels / 255.0, dtype=np.float32)

    # ------------------------------------------------------------------- labels

    def _read_labels(self, path: Path, shape: tuple[int, int]) -> tuple[tuple[Box, ...], dict[str, int]]:
        """Parse one ``label_2`` file into normalised boxes plus a drop tally."""
        if not path.is_file():
            raise FileNotFoundError(f"KITTI label file not found: {path}")
        height, width = shape
        boxes: list[Box] = []
        dropped: dict[str, int] = {}
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) < 8:
                continue
            box = self._parse_row(fields, height, width, dropped)
            if box is not None:
                boxes.append(box)
        return tuple(boxes), dropped

    def _parse_row(
        self,
        fields: Sequence[str],
        height: int,
        width: int,
        dropped: dict[str, int],
    ) -> Box | None:
        """``type truncated occluded alpha x1 y1 x2 y2 ...`` -> :class:`Box` or ``None``."""
        settings: KittiParams = self.params  # type: ignore[assignment]
        raw_label = fields[0]
        label = self._normalised_label(raw_label)
        if label is None:
            dropped[raw_label] = dropped.get(raw_label, 0) + 1
            return None
        truncated = float(fields[1])
        occluded = int(float(fields[2]))
        x1, y1, x2, y2 = (float(value) for value in fields[4:8])
        x1, x2 = np.clip([x1, x2], 0.0, float(width))
        y1, y2 = np.clip([y1, y2], 0.0, float(height))
        min_height, max_occlusion, max_truncation = DIFFICULTY_LIMITS[settings.difficulty]
        if x2 <= x1 or y2 <= y1:
            dropped["degenerate"] = dropped.get("degenerate", 0) + 1
            return None
        if (y2 - y1) < min_height or occluded > max_occlusion or truncated > max_truncation:
            dropped[f"{raw_label}:difficulty"] = dropped.get(f"{raw_label}:difficulty", 0) + 1
            return None
        return Box(float(x1), float(y1), float(x2), float(y2), label, 1.0)

    def _normalised_label(self, raw_label: str) -> str | None:
        settings: KittiParams = self.params  # type: ignore[assignment]
        if raw_label in LABEL_MAP:
            return LABEL_MAP[raw_label]
        if settings.merge_van_truck and raw_label in VEHICLE_ALIASES:
            return VEHICLE_ALIASES[raw_label]
        return None

    # ------------------------------------------------------------ anonymisation

    def _anonymize(self, samples: Sequence[Sample]) -> list[Sample]:
        """Run the placeholder anonymiser and write its manifest (plan §6)."""
        settings: KittiParams = self.params  # type: ignore[assignment]
        anonymized: list[Sample] = []
        entries: list[dict[str, Any]] = []
        for sample in samples:
            result = anonymize_placeholder(sample.image, sample.boxes)
            entries.append(result.manifest_entry(sample.sample_id))
            anonymized.append(
                Sample(
                    sample_id=sample.sample_id,
                    image=result.image,
                    boxes=sample.boxes,
                    depth=sample.depth,
                    lidar=sample.lidar,
                    anonymized=True,
                    meta={**sample.meta, "anonymized_regions": result.n_regions},
                )
            )
        manifest = settings.manifest_path or str(self.root / "anonymization_manifest.json")
        write_manifest(manifest, entries)
        return anonymized
