"""Package reviewed 3D samples in a real KITTI directory contract."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_attacked_kitti3d_zip(
    items: list[dict[str, Any]], *, run_id: str, dataset_name: str = "adversarial-dataset"
) -> tuple[bytes, dict[str, Any]]:
    included: list[tuple[dict[str, Any], Path, Path, Path]] = []
    skipped: list[dict[str, Any]] = []
    labels: set[str] = set()
    for item in items:
        ground_truth = item.get("ground_truth") or {}
        points_value = item.get("attacked_point_cloud_path") or item.get("point_cloud_path") or ""
        label_value = ground_truth.get("label_path") or item.get("label_3d_path") or ""
        calib_value = item.get("calibration_path") or ground_truth.get("calibration_path") or ""
        points = Path(str(points_value)) if points_value else None
        label = Path(str(label_value)) if label_value else None
        calib = Path(str(calib_value)) if calib_value else None
        missing = next(
            (
                (name, value)
                for name, value, path in (
                    ("point cloud", points_value, points),
                    ("label", label_value, label),
                    ("calibration", calib_value, calib),
                )
                if not path or not path.is_file()
            ),
            None,
        )
        if ground_truth.get("type") != "boxes3d":
            skipped.append({"sample_id": item.get("sample_id"), "reason": "no 3D box ground truth"})
            continue
        if missing:
            skipped.append({"sample_id": item.get("sample_id"), "reason": f"{missing[0]} missing: {missing[1]}"})
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
            name = line.split(maxsplit=1)[0] if line.strip() else ""
            if name and name != "DontCare":
                labels.add(name)
        included.append((item, points, label, calib))

    payload = io.BytesIO()
    samples: list[dict[str, Any]] = []
    split_ids: list[str] = []
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item, points, label, calib in included:
            stem = Path(str(item["sample_id"])).name
            split_ids.append(stem)
            point_member = f"velodyne/{stem}.bin"
            label_member = f"label_2/{stem}.txt"
            calib_member = f"calib/{stem}.txt"
            archive.writestr(point_member, points.read_bytes())
            archive.writestr(label_member, label.read_bytes())
            archive.writestr(calib_member, calib.read_bytes())
            samples.append(
                {
                    "sample_id": item["sample_id"],
                    "point_cloud": point_member,
                    "label": label_member,
                    "calibration": calib_member,
                    "attack": item.get("attack"),
                    "severity": item.get("severity"),
                }
            )
        archive.writestr("ImageSets/train.txt", "\n".join(split_ids) + ("\n" if split_ids else ""))
        manifest = {
            "format": "kitti3d",
            "dataset_name": dataset_name,
            "run_id": run_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "sample_count": len(samples),
            "samples": samples,
            "skipped": skipped,
            "class_map": {str(index): label for index, label in enumerate(sorted(labels))},
        }
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True, indent=2))
    return payload.getvalue(), manifest

