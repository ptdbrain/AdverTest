"""Run one real CPU inference per portable 2D model artifact."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

# Direct execution puts ``scripts/`` rather than the repository on sys.path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.adapters import get_adapter  # noqa: E402
from src.adapters.base import ModelAdapter  # noqa: E402
from src.core.memory import trim_memory  # noqa: E402
from src.core.types import DetectionPrediction, Sample, SegmentationPrediction  # noqa: E402
from src.datasets import get_dataset  # noqa: E402


@dataclass(frozen=True, slots=True)
class SmokeModel:
    model_id: str
    task: str
    role: str
    adapter_name: str
    checkpoint_path: Path
    config_path: Path | None = None


def build_smoke_plan(*, data_root: Path, checkpoint_root: Path, runs_root: Path) -> list[SmokeModel]:
    """Return the fixed clean/defence model matrix for the portable preview."""
    return [
        SmokeModel(
            model_id="yolo11s-base",
            task="detection2d",
            role="base",
            adapter_name="yolo11",
            checkpoint_path=checkpoint_root / "surrogates" / "yolo11s.pt",
        ),
        SmokeModel(
            model_id="yolo11s-robust-r1",
            task="detection2d",
            role="defence",
            adapter_name="yolo11",
            checkpoint_path=runs_root / "train" / "yolo_r1" / "run-yolo11s-robust-r1-1786164287" / "yolo11s-robust-r1_best.pt",
        ),
        SmokeModel(
            model_id="sam2-hiera-small-base",
            task="segmentation",
            role="base",
            adapter_name="sam2",
            checkpoint_path=checkpoint_root / "sam2" / "sam2.1_hiera_small.pt",
            config_path=Path("configs/sam2.1/sam2.1_hiera_s.yaml"),
        ),
        SmokeModel(
            model_id="sam21-robust-r1",
            task="segmentation",
            role="defence",
            adapter_name="sam2",
            checkpoint_path=runs_root / "train" / "sam2_r1" / "sam21-robust-r1_best.pt",
            config_path=Path("configs/sam2.1/sam2.1_hiera_s.yaml"),
        ),
    ]


def _load_samples(data_root: Path, task: str, limit: int) -> list[Sample]:
    if task == "detection2d":
        dataset = get_dataset(
            "kitti",
            root=str(data_root / "drive_export_100" / "detection2d" / "kitti2d_100"),
            split="all",
            difficulty="all",
            anonymize="required",
            manifest_path="manifest.jsonl",
        )
        samples = dataset.load(limit=max(limit, 10))
        with_boxes = [sample for sample in samples if sample.boxes]
        return with_boxes[:limit] or samples[:limit]

    dataset = get_dataset(
        "cityscapes_segmentation",
        root=str(data_root / "drive_export_100" / "segmentation" / "cityscapes_instance_100"),
        split="val",
        anonymization_manifest="manifest.jsonl",
    )
    return dataset.load(limit=limit)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_adapter_kwargs(model: SmokeModel, project_root: Path) -> dict[str, object]:
    """Build only the constructor arguments accepted by the selected adapter."""
    if model.adapter_name == "yolo11":
        return {
            "weights": str(model.checkpoint_path.resolve()),
            "device": "cpu",
            "batch_size": 1,
            "half": False,
            "score_threshold": 0.0,
        }
    if model.config_path is None:
        raise ValueError(f"SAM2 model {model.model_id} has no config path")
    return {
        "weights": str(model.checkpoint_path.resolve()),
        "device": "cpu",
        "config": str((project_root / model.config_path).resolve()),
        "mask_threshold": 0.0,
    }


def _run_one(model: SmokeModel, sample: Sample, project_root: Path) -> dict[str, object]:
    if not model.checkpoint_path.is_file():
        raise FileNotFoundError(f"required checkpoint is missing: {model.checkpoint_path}")
    if model.config_path is not None:
        config_path = project_root / model.config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"required SAM2 config is missing: {config_path}")
    else:
        config_path = None

    adapter: ModelAdapter | None = None
    started = perf_counter()
    try:
        adapter = get_adapter(model.adapter_name, **build_adapter_kwargs(model, project_root))
        metadata = adapter.metadata()
        predictions = adapter.predict([sample])
        if len(predictions) != 1:
            raise RuntimeError(f"{model.model_id} returned {len(predictions)} predictions for one sample")
        prediction = predictions[0]
        if model.task == "detection2d" and not isinstance(prediction, DetectionPrediction):
            raise TypeError(f"{model.model_id} returned {type(prediction).__name__}, expected DetectionPrediction")
        if model.task == "segmentation" and not isinstance(prediction, SegmentationPrediction):
            raise TypeError(f"{model.model_id} returned {type(prediction).__name__}, expected SegmentationPrediction")
        prediction_count = len(prediction.boxes) if isinstance(prediction, DetectionPrediction) else sum(
            len(instance.mask) > 0 for instance in prediction.instances
        )
        if prediction_count <= 0:
            raise RuntimeError(f"{model.model_id} returned no non-empty real predictions")
        return {
            "model_id": model.model_id,
            "task": model.task,
            "role": model.role,
            "checkpoint_path": str(model.checkpoint_path.resolve()),
            "checkpoint_sha256": metadata.checkpoint_hash,
            "prediction_type": type(prediction).__name__,
            "prediction_count": prediction_count,
            "sample_id": sample.sample_id,
            "device": "cpu",
            "latency_ms": (perf_counter() - started) * 1000.0,
            "real_model": True,
            "automatic_download": False,
        }
    finally:
        if adapter is not None and hasattr(adapter, "unload"):
            adapter.unload()
        del adapter
        gc.collect()
        trim_memory()


def run_smoke(*, data_root: Path, checkpoint_root: Path, runs_root: Path, limit: int) -> list[dict[str, object]]:
    project_root = Path(__file__).resolve().parents[1]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    samples = {
        task: _load_samples(data_root, task, limit)
        for task in {"detection2d", "segmentation"}
    }
    if any(not values for values in samples.values()):
        raise RuntimeError(f"portable dataset did not yield samples: {samples}")
    results = []
    for model in build_smoke_plan(data_root=data_root, checkpoint_root=checkpoint_root, runs_root=runs_root):
        results.append(_run_one(model, samples[model.task][0], project_root))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("data/checkpoints"))
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    results = run_smoke(
        data_root=args.data_root,
        checkpoint_root=args.checkpoint_root,
        runs_root=args.runs_root,
        limit=args.limit,
    )
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
