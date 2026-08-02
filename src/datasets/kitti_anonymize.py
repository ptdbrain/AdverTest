"""PLACEHOLDER anonymiser standing in for the plan §6 gate on KITTI.

.. warning::

   **This is a placeholder, not a privacy control.** It does not detect faces or
   licence plates. It blurs the region of each ground-truth box where a face or a
   plate *usually* sits, which is enough to unblock the pipeline and to make the
   ``ΔAP`` measurement of plan §6 meaningful, and nothing more. A manifest it
   writes records what it did — it is **not** evidence that a frame is anonymous.
   Do not publish or redistribute KITTI images processed only by this module.

The real thing (plan §6) is a separate slot: SCRFD/RetinaFace for faces plus a
plate detector, tuned to recall ≈ 0.98 and filtered by intersection with the
``person``/``car`` ground truth, exactly as nuScenes does it [Caesar et al. 2020].

Why it still blurs something instead of being a no-op: the gate in
:meth:`~src.datasets.base.DatasetSource.require_anonymized` exists so that
un-anonymised pixels never reach a model. A stub that flipped the flag without
touching pixels would turn that gate into a lie, and would also hide the fact
that blurring itself costs AP — the effect plan §6 asks us to report separately
from attack degradation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.core.hashing import array_digest
from src.core.image_ops import box_slice, clip01
from src.core.types import Box

#: Top fraction of a person-like box that plausibly contains a face.
HEAD_FRACTION = 0.35
#: Bottom fraction of a vehicle box that plausibly contains a licence plate.
PLATE_FRACTION = 0.40
#: Classes treated as person-like / vehicle-like in the normalised label space.
PERSON_CLASSES = ("Pedestrian", "Cyclist")
VEHICLE_CLASSES = ("Car",)

ANONYMIZER_ID = "placeholder-v1"
MANIFEST_WARNING = (
    "PLACEHOLDER anonymiser: ground-truth-box heuristic, no face/plate detector. "
    "Not a privacy guarantee — see src/datasets/kitti_anonymize.py and plan §6."
)


@dataclass(frozen=True, slots=True)
class AnonymizationResult:
    """Blurred pixels plus the audit trail for one frame."""

    image: np.ndarray
    n_regions: int

    def manifest_entry(self, sample_id: str) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "n_regions": self.n_regions,
            "digest": array_digest(self.image),
            "anonymizer": ANONYMIZER_ID,
        }


def anonymize_placeholder(
    image: np.ndarray,
    boxes: Sequence[Box],
    *,
    head_fraction: float = HEAD_FRACTION,
    plate_fraction: float = PLATE_FRACTION,
    blur_scale: float = 0.25,
) -> AnonymizationResult:
    """Blur the head band of person boxes and the plate band of vehicle boxes."""
    height, width = image.shape[:2]
    output = image.copy()
    regions = _regions(boxes, height, width, head_fraction, plate_fraction)
    for rows, columns in regions:
        patch = output[rows, columns]
        radius = max(1, int(round(min(patch.shape[0], patch.shape[1]) * blur_scale)))
        output[rows, columns] = _mosaic(patch, radius)
    return AnonymizationResult(clip01(output), len(regions))


def _regions(
    boxes: Sequence[Box],
    height: int,
    width: int,
    head_fraction: float,
    plate_fraction: float,
) -> list[tuple[slice, slice]]:
    """Candidate identity regions derived from the ground truth."""
    regions: list[tuple[slice, slice]] = []
    for box in boxes:
        clipped = box_slice(box, height, width)
        if clipped is None:
            continue
        rows, columns = clipped
        span = rows.stop - rows.start
        if box.label in PERSON_CLASSES:
            band = max(1, int(round(span * head_fraction)))
            regions.append((slice(rows.start, rows.start + band), columns))
        elif box.label in VEHICLE_CLASSES:
            band = max(1, int(round(span * plate_fraction)))
            regions.append((slice(rows.stop - band, rows.stop), columns))
    return regions


def _mosaic(patch: np.ndarray, block: int) -> np.ndarray:
    """Block-average the patch — irreversible, unlike a small Gaussian blur."""
    height, width = patch.shape[:2]
    output = np.empty_like(patch)
    for top in range(0, height, block):
        for left in range(0, width, block):
            cell = patch[top : top + block, left : left + block]
            output[top : top + block, left : left + block] = cell.mean(axis=(0, 1))
    return output


def write_manifest(path: str | Path, entries: Sequence[dict[str, Any]]) -> Path:
    """Persist the audit trail next to the dataset; returns the written path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "anonymizer": ANONYMIZER_ID,
        "warning": MANIFEST_WARNING,
        "n_frames": len(entries),
        "entries": list(entries),
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
