#!/usr/bin/env python3
"""Build the reviewed 200-sample KITTI 3D catalog without fetching a 27 GB archive.

The official Velodyne archive supports HTTP ranges.  ``remotezip`` reads only
the selected ``.bin`` members, while camera images and annotations come from
the already anonymised KITTI-200 bundle.  This keeps the deployed data plane
small and reproducible.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

VELODYNE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip"
CALIB_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-200", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", default=200, type=int)
    return parser.parse_args()


def build(*, kitti_200: Path, output: Path, count: int) -> None:
    if count < 1:
        raise ValueError("count must be positive")
    manifest = kitti_200 / "manifest.jsonl"
    descriptor = kitti_200 / "dataset.json"
    if not manifest.is_file() or not descriptor.is_file():
        raise FileNotFoundError("kitti-200 must contain dataset.json and manifest.jsonl")
    source_metadata = json.loads(descriptor.read_text(encoding="utf-8"))
    if not source_metadata.get("anonymized") or source_metadata.get("status") != "complete":
        raise ValueError("kitti-200 must be a completed anonymised bundle")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = sorted(str(record["sample_id"]) for record in records)[:count]
    if len(selected) != count:
        raise ValueError(f"kitti-200 only contains {len(selected)} samples")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    with tempfile.TemporaryDirectory(prefix="advertest-kitti3d-", dir=output.parent) as temporary:
        staged = Path(temporary) / "kitti3d-200"
        shutil.copytree(kitti_200 / "image_2", staged / "image_2")
        shutil.copytree(kitti_200 / "label_2", staged / "label_2")
        shutil.copy2(manifest, staged / "manifest.jsonl")
        (staged / "velodyne").mkdir(parents=True)
        (staged / "calib").mkdir(parents=True)
        (staged / "ImageSets").mkdir(parents=True)
        (staged / "ImageSets" / "all.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")

        _download_calibration(staged / "calib", selected)
        _download_velodyne(staged / "velodyne", selected)
        metadata = {
            "name": "kitti3d-200",
            "display_name": "KITTI 3D Test Set (200 Samples)",
            "task_id": "detection3d",
            "sample_count": count,
            "anonymized": True,
            "status": "complete",
            "license": "CC BY-NC-SA 3.0",
            "source_dataset": "KITTI Object Detection Evaluation 2012",
            "image_anonymization_manifest": "manifest.jsonl",
        }
        (staged / "dataset.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        staged.rename(output)


def _download_calibration(destination: Path, selected: list[str]) -> None:
    from zipfile import ZipFile

    with urllib.request.urlopen(CALIB_URL, timeout=60) as response:  # nosec B310 - fixed official KITTI endpoint
        archive = response.read()
    with tempfile.NamedTemporaryFile(suffix=".zip") as handle:
        handle.write(archive)
        handle.flush()
        with ZipFile(handle.name) as zipped:
            for sample_id in selected:
                source = f"training/calib/{sample_id}.txt"
                (destination / f"{sample_id}.txt").write_bytes(zipped.read(source))


def _download_velodyne(destination: Path, selected: list[str]) -> None:
    try:
        from remotezip import RemoteZip
    except ImportError as exc:  # pragma: no cover - operator dependency
        raise RuntimeError("install remotezip before building the KITTI 3D catalog") from exc
    with RemoteZip(VELODYNE_URL) as zipped:
        for sample_id in selected:
            source = f"training/velodyne/{sample_id}.bin"
            with zipped.open(source) as member:
                (destination / f"{sample_id}.bin").write_bytes(member.read())


if __name__ == "__main__":
    args = _parse_args()
    build(kitti_200=args.kitti_200.resolve(), output=args.output.resolve(), count=args.count)
