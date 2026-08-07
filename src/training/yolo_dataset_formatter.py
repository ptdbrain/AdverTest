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

    n_val = max(1, int(num_samples * 0.15))
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
    try:
        tv_datasets.Kitti(root=str(root_path), train=True, download=True)
    except TypeError:
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


def build_robust_yolo_dataset(
    clean_yolo_dir: str | Path,
    output_dir: str | Path,
    mode: str = "r1",
    target_cluster: str | None = None,
    clean_ratio: float = 0.5,
    attack_severities: tuple[int, ...] = (1, 2, 3),
    seed: int = 20260807,
) -> Path:
    """Builds an augmented YOLO dataset mixing clean images with non-whitebox attacks for defense training (R1/R2).

    White-box attacks (Group D) are strictly excluded to ensure realistic black-box & environmental robustness
    and fast GPU execution.
    """
    import numpy as np
    from src.attacks import get_attack
    from src.attacks.base import AttackContext
    from src.core.types import Box, Sample

    clean_dir = Path(clean_yolo_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()

    clean_train_img = clean_dir / "images" / "train"
    clean_train_lbl = clean_dir / "labels" / "train"
    clean_val_img = clean_dir / "images" / "val"
    clean_val_lbl = clean_dir / "labels" / "val"

    if not clean_train_img.is_dir():
        raise FileNotFoundError(f"Clean training images directory not found at: {clean_train_img}")

    yaml_path = out_dir / "kitti.yaml"
    out_train_img = out_dir / "images" / "train"
    out_train_lbl = out_dir / "labels" / "train"
    out_val_img = out_dir / "images" / "val"
    out_val_lbl = out_dir / "labels" / "val"

    # Reuse existing generated dataset if valid
    if out_train_img.is_dir() and any(out_train_img.iterdir()) and yaml_path.is_file():
        print(f"[*] Reusing existing robust dataset at: {out_dir}")
        return yaml_path

    for d in (out_train_img, out_train_lbl, out_val_img, out_val_lbl):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Validation split is 100% clean benchmark
    val_images = sorted(clean_val_img.glob("*.png")) + sorted(clean_val_img.glob("*.jpg"))
    for v_img in val_images:
        dst_v_img = out_val_img / v_img.name
        dst_v_lbl = out_val_lbl / f"{v_img.stem}.txt"
        src_v_lbl = clean_val_lbl / f"{v_img.stem}.txt"
        if not dst_v_img.exists():
            try:
                dst_v_img.symlink_to(v_img.resolve())
            except OSError:
                shutil.copy2(v_img, dst_v_img)
        if src_v_lbl.is_file() and not dst_v_lbl.exists():
            shutil.copy2(src_v_lbl, dst_v_lbl)

    # 2. Configure Non-Whitebox Attack Pool (Strictly excluding Group D)
    normalized_mode = mode.lower()
    if normalized_mode in ("r1", "robust-mix"):
        attack_names = (
            "fog",
            "snow",
            "gaussian_noise",
            "motion_blur",
            "brightness",
            "contrast",
            "sensor_fault",
            "random_erasing",
            "object_occlusion",
            "camera_dropout",
        )
    else:
        # R2 Targeted Repair mode
        cluster = (target_cluster or "").lower()
        if "fog" in cluster or "weather" in cluster:
            attack_names = ("fog", "snow", "brightness", "motion_blur")
        elif "sensor" in cluster or "occlusion" in cluster:
            attack_names = ("sensor_fault", "random_erasing", "object_occlusion", "camera_dropout")
        elif "noise" in cluster:
            attack_names = ("gaussian_noise", "speckle_noise", "impulse_noise")
        else:
            attack_names = ("fog", "sensor_fault", "gaussian_noise", "random_erasing")

    # Instantiate attacks
    loaded_attacks: list[tuple[str, Any]] = []
    for name in attack_names:
        try:
            atk = get_attack(name)
            loaded_attacks.append((name, atk))
        except Exception as err:
            print(f"[!] Warning: Skipping attack {name}: {err}")

    if not loaded_attacks:
        raise RuntimeError("No non-whitebox attacks could be loaded for robust dataset generation.")

    # 3. Process Train Images
    train_images = sorted(clean_train_img.glob("*.png")) + sorted(clean_train_img.glob("*.jpg"))
    if not train_images:
        raise ValueError(f"No images found in clean train directory {clean_train_img}")

    rng = random.Random(seed)
    shuffled_indices = list(range(len(train_images)))
    rng.shuffle(shuffled_indices)

    num_clean = max(1, int(len(train_images) * clean_ratio))

    print(f"\n[*] Generating Robust Dataset for {normalized_mode.upper()}:")
    print(f"    - Base clean images  : {len(train_images)}")
    print(f"    - Clean images kept  : {num_clean} ({clean_ratio * 100:.0f}%)")
    print(f"    - Attacked images gen: {len(train_images) - num_clean} ({(1 - clean_ratio) * 100:.0f}%)")
    print(f"    - Attack pool        : {', '.join(attack_names)}")

    for idx, img_idx in enumerate(shuffled_indices):
        img_path = train_images[img_idx]
        stem = img_path.stem
        src_lbl = clean_train_lbl / f"{stem}.txt"

        if idx < num_clean:
            # Clean sample
            dst_img = out_train_img / img_path.name
            dst_lbl = out_train_lbl / f"{stem}.txt"
            if not dst_img.exists():
                try:
                    dst_img.symlink_to(img_path.resolve())
                except OSError:
                    shutil.copy2(img_path, dst_img)
            if src_lbl.is_file() and not dst_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)
        else:
            # Attacked / Corrupted sample
            atk_name, atk_inst = loaded_attacks[rng.randint(0, len(loaded_attacks) - 1)]
            sev = rng.choice(attack_severities)

            dst_img_name = f"aug_{atk_name}_s{sev}_{img_path.name}"
            dst_img = out_train_img / dst_img_name
            dst_lbl = out_train_lbl / f"aug_{atk_name}_s{sev}_{stem}.txt"

            if not dst_img.exists():
                with Image.open(img_path) as pil_img:
                    rgb_img = pil_img.convert("RGB")
                    img_np = np.asarray(rgb_img, dtype=np.float32) / 255.0

                h_px, w_px = img_np.shape[0], img_np.shape[1]
                boxes_list: list[Box] = []
                if src_lbl.is_file():
                    for line in src_lbl.read_text(encoding="utf-8").strip().splitlines():
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                c_id = parts[0]
                                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                                x1 = max(0.0, (cx - bw / 2.0) * w_px)
                                y1 = max(0.0, (cy - bh / 2.0) * h_px)
                                x2 = min(float(w_px), (cx + bw / 2.0) * w_px)
                                y2 = min(float(h_px), (cy + bh / 2.0) * h_px)
                                if x2 > x1 and y2 > y1:
                                    boxes_list.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, label=c_id))
                            except ValueError:
                                pass

                sample = Sample(sample_id=stem, image=img_np, boxes=tuple(boxes_list))
                ctx = AttackContext(rng=np.random.default_rng(seed + idx))
                try:
                    attacked_sample = atk_inst.run(sample, sev, ctx)
                except Exception:
                    # Fallback to image-level noise or fog if sample doesn't satisfy attack requirements
                    fallback_atk = get_attack("gaussian_noise")
                    attacked_sample = fallback_atk.run(sample, 2, ctx)

                out_img_np = np.clip(attacked_sample.image * 255.0, 0.0, 255.0).astype(np.uint8)
                Image.fromarray(out_img_np).save(dst_img, quality=95)

            if src_lbl.is_file() and not dst_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)

    yaml_content = f"""path: {out_dir.as_posix()}
train: images/train
val: images/val
names:
  0: Pedestrian
  1: Cyclist
  2: Car
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"      Robust dataset manifest written to: {yaml_path}")
    return yaml_path



