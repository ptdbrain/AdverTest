"""Run a fixed-prompt clean Cityscapes SAM2 benchmark without retaining all masks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.adapters.sam2 import Sam2Adapter
from src.datasets.cityscapes import CityscapesSegmentationDataset


def _iou(prediction: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / union) if union else 1.0


def _dice(prediction: np.ndarray, target: np.ndarray) -> float:
    total = prediction.sum() + target.sum()
    return float(2 * np.logical_and(prediction, target).sum() / total) if total else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    source = CityscapesSegmentationDataset(
        root=args.dataset_root,
        split=args.split,
        anonymization_manifest="manifest.jsonl",
        max_samples=args.max_samples,
    )
    adapter = Sam2Adapter(weights=args.checkpoint, config=args.config, device="cuda")
    by_metric: dict[str, list[float]] = defaultdict(list)
    failures = 0
    mask_count = 0
    sample_count = 0
    for sample in source.iter_samples():
        sample_count += 1
        prediction = adapter.predict([sample])[0]
        for instance in prediction.instances:
            target = sample.mask == int(instance.instance_id)
            predicted = np.asarray(instance.mask, dtype=bool)
            iou = _iou(predicted, target)
            mask_count += 1
            failures += int(iou < 0.5 or not predicted.any())
            by_metric["iou"].append(iou)
            by_metric["dice"].append(_dice(predicted, target))
    payload = {
        "comparison_scope": "cityscapes_locked_clean",
        "split": args.split,
        "sample_count": sample_count,
        "mask_count": mask_count,
        "miou": float(np.mean(by_metric["iou"])) if mask_count else 0.0,
        "dice": float(np.mean(by_metric["dice"])) if mask_count else 0.0,
        "mask_failure_rate": failures / mask_count if mask_count else 0.0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
