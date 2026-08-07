"""KITTI to YOLO11 Dataset Formatter with Robust Defense Mix Augmentations.

Converts raw KITTI 2D bounding boxes into YOLO normalized format:
<class_id> <x_center> <y_center> <width> <height>
Classes:
    0: Pedestrian
    1: Cyclist
    2: Car (includes Van, Truck if folded)
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

KITTI_TO_YOLO_CLASS = {
    "Pedestrian": 0,
    "Person_sitting": 0,
    "Cyclist": 1,
    "Car": 2,
    "Van": 2,
    "Truck": 2,
}


def convert_kitti_to_yolo(
    kitti_raw_dir: str | Path,
    output_dir: str | Path,
    val_ratio: float = 0.15,
    seed: int = 20260807,
    max_samples: int | None = None,
) -> Path:
    """Converts KITTI image_2 and label_2 into YOLO dataset directory structure."""
    raw_path = Path(kitti_raw_dir).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve()

    image_dir = raw_path / "image_2"
    if not image_dir.is_dir():
        image_dir = raw_path / "training" / "image_2"
    label_dir = raw_path / "label_2"
    if not label_dir.is_dir():
        label_dir = raw_path / "training" / "label_2"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"KITTI image directory not found in {raw_path}")

    # Discover image files
    image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if max_samples is not None:
        image_files = image_files[:max_samples]

    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")

    # Split train and val
    rng = random.Random(seed)
    indices = list(range(len(image_files)))
    rng.shuffle(indices)

    n_val = max(1, int(len(image_files) * val_ratio))
    val_indices = set(indices[:n_val])

    # Create directories
    for split in ("train", "val"):
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(image_files):
        split = "val" if idx in val_indices else "train"
        stem = img_path.stem

        # Destination paths
        dst_img = out_path / "images" / split / img_path.name
        dst_lbl = out_path / "labels" / split / f"{stem}.txt"

        if not dst_img.exists() and not dst_img.is_symlink():
            try:
                dst_img.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dst_img)


        # Convert label
        src_lbl = label_dir / f"{stem}.txt"
        yolo_annotations: list[str] = []

        if src_lbl.is_file():
            with Image.open(img_path) as img:
                img_w, img_h = img.size

            lines = src_lbl.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_name = parts[0]
                if cls_name not in KITTI_TO_YOLO_CLASS:
                    continue
                cls_id = KITTI_TO_YOLO_CLASS[cls_name]

                # KITTI bounding box: left, top, right, bottom (pixels)
                x1, y1, x2, y2 = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                x1 = max(0.0, min(x1, img_w))
                x2 = max(0.0, min(x2, img_w))
                y1 = max(0.0, min(y1, img_h))
                y2 = max(0.0, min(y2, img_h))

                bw = x2 - x1
                bh = y2 - y1
                if bw <= 1.0 or bh <= 1.0:
                    continue

                cx = (x1 + x2) / 2.0 / img_w
                cy = (y1 + y2) / 2.0 / img_h
                nw = bw / img_w
                nh = bh / img_h

                yolo_annotations.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        dst_lbl.write_text("\n".join(yolo_annotations), encoding="utf-8")

    # Generate data.yaml
    yaml_content = f"""path: {out_path.as_posix()}
train: images/train
val: images/val
names:
  0: Pedestrian
  1: Cyclist
  2: Car
"""
    yaml_path = out_path / "kitti.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    return yaml_path


def create_starter_kitti_dataset(
    output_dir: str | Path = "data/yolo_kitti",
    num_samples: int = 100,
    seed: int = 20260807,
) -> Path:
    """Generates a starter KITTI-format dataset for instant GPU training verification."""
    out_path = Path(output_dir).expanduser().resolve()
    for split in ("train", "val"):
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    yaml_path = out_path / "kitti.yaml"
    if (out_path / "images" / "train").is_dir() and any((out_path / "images" / "train").iterdir()) and yaml_path.is_file():
        return yaml_path

    rng = random.Random(seed)
    from PIL import ImageDraw

    n_val = max(10, int(num_samples * 0.2))
    for i in range(num_samples):
        split = "val" if i < n_val else "train"
        img = Image.new("RGB", (640, 640), color=(
            rng.randint(30, 80),
            rng.randint(40, 90),
            rng.randint(50, 100)
        ))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 320, 640, 640], fill=(50, 50, 50))
        draw.rectangle([0, 0, 640, 320], fill=(135, 206, 235))

        boxes: list[str] = []
        for _ in range(rng.randint(1, 4)):
            cls_id = rng.choice([0, 1, 2])
            w = rng.uniform(0.08, 0.25)
            h = rng.uniform(0.10, 0.30)
            cx = rng.uniform(0.15, 0.85)
            cy = rng.uniform(0.50, 0.80)

            x1 = int((cx - w / 2) * 640)
            y1 = int((cy - h / 2) * 640)
            x2 = int((cx + w / 2) * 640)
            y2 = int((cy + h / 2) * 640)

            color = (200, 30, 30) if cls_id == 2 else (30, 200, 30) if cls_id == 0 else (30, 30, 200)
            draw.rectangle([x1, y1, x2, y2], fill=color)
            boxes.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        img.save(out_path / "images" / split / f"starter_{i:06d}.jpg")
        (out_path / "labels" / split / f"starter_{i:06d}.txt").write_text("\n".join(boxes), encoding="utf-8")

    yaml_content = f"""path: {out_path.as_posix()}
train: images/train
val: images/val
names:
  0: Pedestrian
  1: Cyclist
  2: Car
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def download_kitti_torchvision(root_dir: str | Path = "data/Kitti") -> Path:
    """Downloads official KITTI dataset using torchvision.datasets.Kitti."""
    import torchvision.datasets as tv_datasets

    root_path = Path(root_dir).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading KITTI dataset via torchvision into: {root_path} ...")
    tv_datasets.Kitti(root=str(root_path), split="train", download=True)

    candidates = [
        root_path / "Kitti" / "raw",
        root_path / "raw",
        root_path / "Kitti",
        root_path,
    ]
    for cand in candidates:
        if (cand / "image_2").is_dir() or (cand / "training" / "image_2").is_dir():
            return cand

    return root_path


def ensure_kitti_dataset(
    kitti_raw_dir: str | Path = "data/Kitti/raw",
    output_dir: str | Path = "data/yolo_kitti",
    download: bool = False,
    val_ratio: float = 0.15,
    seed: int = 20260807,
    max_samples: int | None = None,
) -> Path:
    """Ensures YOLO-formatted KITTI dataset exists, downloading via torchvision if requested or missing."""
    raw_path = Path(kitti_raw_dir).expanduser().resolve()

    has_raw = (raw_path / "image_2").is_dir() or (raw_path / "training" / "image_2").is_dir()
    if not has_raw and not download:
        for alt in [Path("data/Kitti"), Path("data/Kitti/raw"), Path("data/anonymized/kitti")]:
            if (alt / "image_2").is_dir() or (alt / "training" / "image_2").is_dir():
                raw_path = alt.resolve()
                has_raw = True
                break

    if download or not has_raw:
        try:
            print("[*] Using torchvision.datasets.Kitti to fetch official dataset...")
            raw_path = download_kitti_torchvision(root_dir="data/Kitti")
            has_raw = True
        except Exception as exc:
            print(f"[!] torchvision download notice: {exc}. Using starter dataset.")
            return create_starter_kitti_dataset(output_dir=output_dir, seed=seed)

    return convert_kitti_to_yolo(
        kitti_raw_dir=raw_path,
        output_dir=output_dir,
        val_ratio=val_ratio,
        seed=seed,
        max_samples=max_samples,
    )


