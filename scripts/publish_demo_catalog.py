#!/usr/bin/env python3
"""Build and publish the small, reviewed demo bundles used by the web catalog.

These are smoke-test assets, not redistributed copies of the source datasets.
Authentication uses the active ``gcloud`` account and uploads directly to GCS.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.hashing import array_digest, stable_digest  # noqa: E402
from src.datasets.io import load_image  # noqa: E402

DEFAULT_BUCKET = "advertest-prod-artifacts"
PREFIX = "catalog/datasets/demo-catalog/v1"


def copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def build(destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    anonymized = ROOT / "data" / "anonymized"
    copy_tree(anonymized / "bdd100k-smoke", destination / "bdd100k_semantic")
    copy_tree(anonymized / "cityscapes-smoke", destination / "cityscapes_segmentation")
    copy_tree(anonymized / "kitti-de", destination / "folder_dataset")

    # Detection needs its own official BDD100K layout and label index.
    detection = destination / "bdd100k_detection"
    image = next((anonymized / "bdd100k-smoke" / "10k" / "train").glob("*.jpg"))
    copied = detection / "images" / "100k" / "val" / image.name
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image, copied)
    write_json(
        detection / "labels" / "det_20" / "det_val.json",
        [{"name": image.name, "labels": [{"category": "car", "box2d": {"x1": 1, "y1": 1, "x2": 20, "y2": 20}}]}],
    )
    write_json(detection / "dataset.json", {"anonymized": True, "kind": "demo"})
    (detection / "manifest.jsonl").write_text("{}\\n", encoding="utf-8")

    # A minimal calibrated 3D sample exercises the true KITTI3D loader.
    kitti3d = destination / "kitti3d" / "training"
    for name in ("image_2", "velodyne", "calib", "label_2"):
        (kitti3d / name).mkdir(parents=True)
    source_image = next((anonymized / "kitti-de" / "image_2").glob("*.png"))
    sample_id = "000001"
    shutil.copy2(source_image, kitti3d / "image_2" / f"{sample_id}.png")
    np.asarray([[1.0, 2.0, 3.0, 0.5], [4.0, 5.0, 6.0, 0.25]], dtype=np.float32).tofile(kitti3d / "velodyne" / f"{sample_id}.bin")
    (kitti3d / "calib" / f"{sample_id}.txt").write_text(
        "P2: 1 0 0 0 0 1 0 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 0 -1 0 0 0 0 -1 0 1 0 0 0\n",
        encoding="utf-8",
    )
    (kitti3d / "label_2" / f"{sample_id}.txt").write_text("Car 0 0 0 0 0 50 50 2 2 4 0 0 10 0\n", encoding="utf-8")
    (destination / "kitti3d" / "manifest.jsonl").write_text("{}\n", encoding="utf-8")

    # A valid one-record generated dataset verifies the generated-data loader.
    generated = destination / "generated_dataset"
    generated.mkdir()
    generated_image = generated / "frame.png"
    shutil.copy2(source_image, generated_image)
    label = {"boxes": [{"x1": 1.0, "y1": 1.0, "x2": 20.0, "y2": 20.0, "label": "Car", "score": 1.0}]}
    write_json(generated / "frame.json", label)
    image_value = load_image(generated_image)
    record = {
        "variant_id": "demo-variant-001", "image_path": "frame.png", "label_path": "frame.json",
        "output_hash": array_digest(image_value, length=32),
        "label_hash": stable_digest(label, length=32), "mask_hash": None,
    }
    (generated / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    write_json(generated / "dataset.json", {
        "status": "complete", "format": "advertest-generated-v2", "anonymized": True,
        "n_variants": 1, "manifest_hash": stable_digest([record], length=32),
    })
    (destination / ".ready").write_text("demo-catalog-v1\n", encoding="utf-8")


def upload(root: Path, bucket: str) -> None:
    # CI and WSL sessions can provide a short-lived token without depending on
    # a platform-specific ``gcloud`` launcher.
    token = os.environ.get("GCS_ACCESS_TOKEN") or subprocess.check_output(
        ["gcloud", "auth", "print-access-token"], text=True
    ).strip()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        name = f"{PREFIX}/{path.relative_to(root).as_posix()}"
        url = "https://storage.googleapis.com/upload/storage/v1/b/" + bucket + "/o?uploadType=media&name=" + urllib.parse.quote(name, safe="")
        request = urllib.request.Request(url, data=path.read_bytes(), method="POST")
        request.add_header("Authorization", f"Bearer {token}")
        request.add_header("Content-Type", "application/octet-stream")
        with urllib.request.urlopen(request) as response:
            if response.status not in (200, 201):
                raise RuntimeError(f"upload failed for {name}: {response.status}")
        print(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / ".demo-catalog-build")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    build(args.output)
    if args.upload:
        upload(args.output, args.bucket)


if __name__ == "__main__":
    main()
