"""Fine-tune YOLO11s on KITTI (M1 of plan §1.2: "Ultralytics, 30 epochs on KITTI").

    python scripts/train_yolo11_kitti.py --epochs 30 --batch 8

Two stages, both resumable:

1. **Convert.** KITTI ``label_2`` rows become YOLO ``class cx cy w h`` (normalised)
   under ``--work-dir``, reusing the label parsing, class mapping and difficulty
   filter of :class:`src.datasets.kitti.Kitti` so the training label space is
   exactly the one the benchmark evaluates against. Images are symlinked, not
   copied — the dataset stays a single 12 GB copy on disk.
2. **Train.** Ultralytics does the rest.

Afterwards, point the benchmark at the new checkpoint:

    python scripts/benchmark_kitti_yolo11.py --weights runs/detect/train/weights/best.pt

The adapter's ``version`` string embeds the weights stem, so the fine-tuned run
gets its own prediction-cache namespace and both models can be compared in one
report rather than silently overwriting each other.

Sized for a 6 GB card: ``imgsz 640``, ``batch 8``, AMP on. Expect a few hours.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Runnable as `python scripts/train_yolo11_kitti.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.types import CLASSES  # noqa: E402
from src.datasets import get_dataset  # noqa: E402
from src.datasets.kitti import Kitti  # noqa: E402

CLASS_INDEX = {name: index for index, name in enumerate(CLASSES)}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset = get_dataset("kitti", root=args.root, difficulty=args.difficulty) if args.root else get_dataset("kitti", difficulty=args.difficulty)
    assert isinstance(dataset, Kitti)

    work_dir = Path(args.work_dir).resolve()
    counts = {split: _convert_split(dataset, split, work_dir) for split in ("train", "val")}
    print(f"converted: {counts}")
    config = _write_config(work_dir)
    print(f"dataset config: {config}")
    if args.convert_only:
        return 0
    return _train(args, config)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=None, help="KITTI root (default: $ADVERTEST_KITTI_ROOT)")
    parser.add_argument("--work-dir", default="data/kitti_yolo")
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--difficulty", default="moderate", choices=["all", "easy", "moderate", "hard"])
    parser.add_argument("--convert-only", action="store_true")
    return parser.parse_args(argv)


def _convert_split(dataset: Kitti, split: str, work_dir: Path) -> int:
    """KITTI labels -> YOLO txt, images symlinked. Returns the frame count."""
    image_dir = work_dir / "images" / split
    label_dir = work_dir / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for image_id in _split_ids(dataset, split):
        source = dataset.image_dir / f"{image_id}.png"
        if not source.is_file():
            continue
        width, height = _image_size(source)
        boxes, _ = dataset._read_labels(dataset.label_dir / f"{image_id}.txt", (height, width))
        rows = [_yolo_row(box, width, height) for box in boxes]
        (label_dir / f"{image_id}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        link = image_dir / f"{image_id}.png"
        if not link.exists():
            link.symlink_to(source.resolve())
        written += 1
    return written


def _split_ids(dataset: Kitti, split: str) -> list[str]:
    split_file = dataset.root / "ImageSets" / f"{split}.txt"
    if split_file.is_file():
        return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    available = sorted(path.stem for path in dataset.image_dir.glob("*.png"))
    half = len(available) // 2
    return available[:half] if split == "train" else available[half:]


def _image_size(path: Path) -> tuple[int, int]:
    """``(width, height)`` from the PNG header — no full decode."""
    from PIL import Image

    with Image.open(path) as handle:
        return handle.size


def _yolo_row(box, width: int, height: int) -> str:  # noqa: ANN001 - Box, kept local
    """``class cx cy w h``, all normalised to ``[0, 1]``."""
    centre_x = (box.x1 + box.x2) / 2.0 / width
    centre_y = (box.y1 + box.y2) / 2.0 / height
    return (
        f"{CLASS_INDEX[box.label]} {centre_x:.6f} {centre_y:.6f} "
        f"{box.width / width:.6f} {box.height / height:.6f}"
    )


def _write_config(work_dir: Path) -> Path:
    """Ultralytics dataset yaml — written by hand to avoid a PyYAML dependency."""
    lines = [f"path: {work_dir}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {index}: {name}" for name, index in CLASS_INDEX.items()]
    config = work_dir / "kitti.yaml"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config


def _train(args: argparse.Namespace, config: Path) -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("training needs the optional extras: pip install ultralytics torch")
        return 1
    YOLO(args.model).train(
        data=str(config),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        amp=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
