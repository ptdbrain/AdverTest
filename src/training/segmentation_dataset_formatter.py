"""Package reviewed segmentation samples without converting masks to boxes."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_attacked_segmentation_zip(
    items: list[dict[str, Any]], *, run_id: str, dataset_name: str = "adversarial-dataset"
) -> tuple[bytes, dict[str, Any]]:
    included: list[tuple[dict[str, Any], Path, Path]] = []
    skipped: list[dict[str, Any]] = []
    class_map: dict[str, str] = {}
    for item in items:
        ground_truth = item.get("ground_truth") or {}
        image_value = item.get("attacked_image_path") or ""
        mask_value = ground_truth.get("mask_path") or item.get("ground_truth_mask_path") or item.get("mask_path") or ""
        image = Path(str(image_value)) if image_value else None
        mask = Path(str(mask_value)) if mask_value else None
        if ground_truth.get("type") not in {"mask", "segmentation"}:
            skipped.append({"sample_id": item.get("sample_id"), "reason": "no segmentation mask ground truth"})
            continue
        if not image or not image.is_file():
            skipped.append({"sample_id": item.get("sample_id"), "reason": f"attacked image missing: {image_value}"})
            continue
        if not mask or not mask.is_file():
            skipped.append({"sample_id": item.get("sample_id"), "reason": f"mask missing: {mask_value}"})
            continue
        if isinstance(ground_truth.get("class_map"), dict):
            class_map.update({str(key): str(value) for key, value in ground_truth["class_map"].items()})
        included.append((item, image, mask))

    payload = io.BytesIO()
    samples: list[dict[str, Any]] = []
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (item, image, mask) in enumerate(included):
            stem = f"{index:06d}_{Path(str(item['sample_id'])).name}"
            image_member = f"images/{stem}{image.suffix.lower() or '.png'}"
            mask_member = f"masks/{stem}{mask.suffix.lower() or '.png'}"
            archive.writestr(image_member, image.read_bytes())
            archive.writestr(mask_member, mask.read_bytes())
            samples.append(
                {
                    "sample_id": item["sample_id"],
                    "image": image_member,
                    "mask": mask_member,
                    "attack": item.get("attack"),
                    "severity": item.get("severity"),
                }
            )
        manifest = {
            "format": "segmentation-mask",
            "dataset_name": dataset_name,
            "run_id": run_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "sample_count": len(samples),
            "samples": samples,
            "skipped": skipped,
            "class_map": class_map,
        }
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, indent=2))
    return payload.getvalue(), manifest

