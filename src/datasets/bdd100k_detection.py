"""BDD100K detection loader with versioned class mapping.

Loads BDD100K detection split with 2D bounding boxes for external evaluation.
Class mapping is versioned and reports dropped/merged classes explicitly.

Important: BDD100K semantic-only data MUST NOT substitute instance-mask GT or
be used to claim instance mAP.  External split MUST NOT be used for checkpoint
selection or threshold tuning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field as PydanticField

from src.core.types import Box, Sample
from src.datasets import DATASETS
from src.datasets.base import DatasetInfo, DatasetParams, DatasetSource
from src.datasets.io import load_image

# ---- Versioned class mapping ----

@dataclass(frozen=True)
class ClassMapping:
    """Maps BDD100K class names to the canonical AdverTest class space."""

    version: str
    mapping: dict[str, str]
    dropped: tuple[str, ...] = ()
    merged: dict[str, str] = field(default_factory=dict)

    def translate(self, bdd_label: str) -> str | None:
        """Return canonical label or None if dropped."""
        if bdd_label in self.dropped:
            return None
        return self.mapping.get(bdd_label) or self.merged.get(bdd_label)


# Default: map BDD100K → AdverTest canonical classes
BDD100K_CLASS_MAPPING_V1 = ClassMapping(
    version="bdd100k-det-v1",
    mapping={
        "car": "Car",
        "pedestrian": "Pedestrian",
        "rider": "Cyclist",
        "bicycle": "Cyclist",
        "motorcycle": "Cyclist",
    },
    dropped=("bus", "truck", "train", "traffic light", "traffic sign"),
    merged={"rider": "Cyclist", "bicycle": "Cyclist", "motorcycle": "Cyclist"},
)


def class_mapping_report(mapping: ClassMapping) -> dict[str, Any]:
    """Generate a human-readable report of class mapping decisions."""
    return {
        "version": mapping.version,
        "canonical_classes": sorted(set(mapping.mapping.values())),
        "dropped_classes": list(mapping.dropped),
        "merged_classes": {k: v for k, v in mapping.merged.items()},
        "total_source_classes": len(mapping.mapping) + len(mapping.dropped),
    }


# ---- Dataset loader ----

class BDD100KDetectionParams(DatasetParams):
    root: str
    split: Literal["train", "val"] = "val"
    anonymization_manifest: str | None = None
    max_samples: int | None = PydanticField(default=None, ge=1)
    class_mapping_version: str = "bdd100k-det-v1"


@DATASETS.register
class BDD100KDetectionDataset(DatasetSource):
    """BDD100K 2D detection loader for external evaluation.

    External evaluation results MUST NOT participate in checkpoint selection.
    """

    name: ClassVar[str] = "bdd100k_detection"
    owner: ClassVar[str] = "group-d"
    loader_version: ClassVar[str] = "bdd100k-det-v1"
    params_model: ClassVar[type[DatasetParams]] = BDD100KDetectionParams

    _KNOWN_MAPPINGS: ClassVar[dict[str, ClassMapping]] = {
        "bdd100k-det-v1": BDD100K_CLASS_MAPPING_V1,
    }

    def __init__(self, **params: object) -> None:
        super().__init__(**params)
        self.root = Path(self.params.root).expanduser().resolve()  # type: ignore[attr-defined]
        mapping_version = self.params.class_mapping_version  # type: ignore[attr-defined]
        if mapping_version not in self._KNOWN_MAPPINGS:
            raise ValueError(
                f"unknown class mapping version {mapping_version!r}; "
                f"available: {sorted(self._KNOWN_MAPPINGS)}"
            )
        self.class_mapping = self._KNOWN_MAPPINGS[mapping_version]

    def info(self) -> DatasetInfo:
        report = class_mapping_report(self.class_mapping)
        return DatasetInfo(
            name=self.name,
            anonymized=False,
            classes=tuple(report["canonical_classes"]),
            note=(
                f"external detection; class mapping {self.class_mapping.version}; "
                f"dropped: {report['dropped_classes']}; "
                f"merged: {report['merged_classes']}"
            ),
        )

    def load(self, limit: int | None = None) -> list[Sample]:
        split = self.params.split  # type: ignore[attr-defined]
        image_dir = self.root / "images" / "100k" / split
        label_dir = self.root / "labels" / "det_20" / f"det_{split}.json"

        if not image_dir.is_dir():
            raise FileNotFoundError(f"BDD100K image directory not found: {image_dir}")

        # Load detection annotations
        annotations: dict[str, list[dict[str, Any]]] = {}
        if label_dir.is_file():
            raw = json.loads(label_dir.read_text(encoding="utf-8"))
            for frame in raw:
                name = frame.get("name", "")
                labels = frame.get("labels", [])
                boxes = []
                for label in labels:
                    if label.get("category") and label.get("box2d"):
                        boxes.append(label)
                if boxes:
                    annotations[name] = boxes

        image_paths = sorted(image_dir.glob("*.jpg"))
        if limit is None:
            limit = self.params.max_samples  # type: ignore[attr-defined]
        if limit is not None:
            image_paths = image_paths[:limit]

        samples: list[Sample] = []
        for image_path in image_paths:
            filename = image_path.name
            frame_boxes = annotations.get(filename, [])
            boxes: list[Box] = []
            for det in frame_boxes:
                bdd_label = det["category"]
                canonical = self.class_mapping.translate(bdd_label)
                if canonical is None:
                    continue  # dropped class
                b = det["box2d"]
                boxes.append(Box(
                    x1=float(b["x1"]),
                    y1=float(b["y1"]),
                    x2=float(b["x2"]),
                    y2=float(b["y2"]),
                    label=canonical,
                ))

            samples.append(Sample(
                sample_id=f"bdd100k/{split}/{image_path.stem}",
                image=load_image(image_path),
                boxes=tuple(boxes),
                meta={
                    "source": "bdd100k",
                    "split": split,
                    "class_mapping_version": self.class_mapping.version,
                    "evaluation_scope": "external",
                    "checkpoint_selection_prohibited": True,
                },
            ))

        return samples
