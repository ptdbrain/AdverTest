"""YOLO11 Robust Training Runner for Local GPU & Google Colab environments.

Usage:
    # 1. Train Baseline (YOLO-B0) on Clean KITTI split:
    python scripts/train_yolo.py --mode b0 --epochs 30 --batch-size 16 --output-dir runs/train/yolo_b0

    # 2. Train Robust Mix (YOLO-R1) from Baseline B0:
    python scripts/train_yolo.py --mode r1 --base-checkpoint runs/train/yolo_b0/yolo11s-clean-b0_best.pt \
        --epochs 20 --batch-size 16 --output-dir runs/train/yolo_r1

    # 3. Train Targeted Repair (YOLO-R2) for a specific failure cluster:
    python scripts/train_yolo.py --mode r2 --base-checkpoint runs/train/yolo_r1/yolo11s-robust-r1_best.pt \
        --epochs 15 --batch-size 16 --target-cluster "pedestrian_fog_occlusion" \
        --output-dir runs/train/yolo_r2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False

from src.training.base import TrainerCallbacks
from src.training.contracts import TrainingRunConfig
from src.training.yolo_dataset_formatter import (
    build_robust_yolo_dataset,
    ensure_kitti_dataset,
)
from src.training.yolo_trainer import YoloTrainer




def check_environment() -> dict[str, Any]:
    if HAS_TORCH:
        has_cuda = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"
        device_count = torch.cuda.device_count() if has_cuda else 0
        torch_ver = torch.__version__
    else:
        has_cuda = False
        device_name = "CPU (PyTorch not found)"
        device_count = 0
        torch_ver = "N/A"

    return {
        "cuda_available": has_cuda,
        "device_name": device_name,
        "device_count": device_count,
        "torch_version": torch_ver,
        "ultralytics_available": HAS_ULTRALYTICS,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train & Fine-tune YOLO11 models (B0, R1, R2) on Local / Colab GPU."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["b0", "r1", "r2", "baseline", "robust-mix", "targeted-repair"],
        default="b0",
        help="Training mode: b0 (clean baseline), r1 (robust mix), r2 (targeted repair)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/Kitti/raw",
        help="Path to KITTI raw dataset directory (contains image_2 and label_2)",
    )
    parser.add_argument(
        "--yolo-data-dir",
        type=str,
        default="data/yolo_kitti",
        help="Target directory for converted YOLO dataset",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default=None,
        help="Path to initial weights or parent checkpoint (.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs (default: 25)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260807,
        help="Random seed for reproducibility (default: 20260807)",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="kitti-v1",
        help="Dataset version ID",
    )
    parser.add_argument(
        "--split-id",
        type=str,
        default="kitti-train-split-v1",
        help="Split manifest ID (must not be locked test)",
    )
    parser.add_argument(
        "--target-cluster",
        type=str,
        default="general_robustness",
        help="Target failure cluster identifier (for R2 mode)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use for training: auto, cpu, cuda, 0, etc. (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/train/yolo11",
        help="Output directory for checkpoints and metrics",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable Automatic Mixed Precision (FP16 Tensor Cores, default: True)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_false",
        dest="amp",
        help="Disable Automatic Mixed Precision",
    )
    parser.add_argument(
        "--cache",
        type=str,
        choices=["ram", "disk", "none"],
        default="ram",
        help="Cache dataset images in RAM or disk to eliminate I/O bottleneck (default: ram)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of DataLoader worker threads for fast GPU feeding (default: 8)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download official KITTI dataset using torchvision.datasets.Kitti",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples for fast testing/debugging",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick dry-run without full epochs for sanity check",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = check_environment()

    print("=" * 75)
    print(" AdverTest — YOLO11 Robust Training Pipeline (Person B)")
    print("=" * 75)
    print(f"[*] PyTorch Version      : {env['torch_version']}")
    print(f"[*] Ultralytics Available : {env['ultralytics_available']}")
    print(f"[*] Compute Device       : {env['device_name']} (CUDA: {env['cuda_available']})")
    print(f"[*] Training Mode        : {args.mode.upper()}")
    print(f"[*] Target Epochs        : {args.epochs}")
    print(f"[*] Batch Size           : {args.batch_size}")
    print(f"[*] Learning Rate        : {args.lr}")
    print(f"[*] Random Seed          : {args.seed}")
    print(f"[*] Output Directory     : {args.output_dir}")
    print("=" * 75)

    # Normalize mode
    mode = args.mode.lower()
    if mode in ("b0", "baseline"):
        model_version = "yolo11s-clean-b0"
        defense_profile_id = "profile-clean-b0"
        clean_ratio = 1.0
        gen_ratio = 0.0
    elif mode in ("r1", "robust-mix"):
        model_version = "yolo11s-robust-r1"
        defense_profile_id = "profile-robust-mix-r1"
        clean_ratio = 0.5
        gen_ratio = 0.5
    else:
        model_version = f"yolo11s-repaired-r2-{args.target_cluster}"
        defense_profile_id = f"profile-repaired-r2-{args.target_cluster}"
        clean_ratio = 0.4
        gen_ratio = 0.6

    # Prepare KITTI dataset via torchvision or raw folder
    data_yaml_path: Path | None = None
    if not args.dry_run:
        print("\n[1/5] Preparing KITTI dataset (torchvision / local conversion)...")
        clean_yaml_path = ensure_kitti_dataset(
            kitti_raw_dir=args.data_root,
            output_dir=args.yolo_data_dir,
            download=args.download,
            val_ratio=0.15,
            seed=args.seed,
            max_samples=args.max_samples,
        )
        if mode in ("b0", "baseline"):
            data_yaml_path = clean_yaml_path
        else:
            robust_out_dir = Path(args.yolo_data_dir).parent / f"{Path(args.yolo_data_dir).name}_{mode}"
            data_yaml_path = build_robust_yolo_dataset(
                clean_yolo_dir=args.yolo_data_dir,
                output_dir=robust_out_dir,
                mode=mode,
                target_cluster=args.target_cluster,
                clean_ratio=clean_ratio,
                seed=args.seed,
            )
        print(f"      Dataset ready at: {data_yaml_path}")
    else:
        print("\n[1/5] Skipping KITTI auto-conversion (dry-run mode).")

    epochs = 2 if args.dry_run else args.epochs
    run_id = f"run-{model_version}-{int(time.time())}"

    config = TrainingRunConfig(
        run_id=run_id,
        trainer_name="yolo11",
        model_version=model_version,
        dataset_version_id=args.dataset_id,
        split_manifest_id=args.split_id,
        defense_profile_id=defense_profile_id,
        seed=args.seed,
        epochs=epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        metadata={
            "mode": mode,
            "base_checkpoint": args.base_checkpoint,
            "target_cluster": args.target_cluster,
            "clean_ratio": clean_ratio,
            "generated_ratio": gen_ratio,
            "cuda": env["cuda_available"],
            "device": args.device,
            "amp": args.amp,
            "cache": args.cache,
            "workers": args.workers,
            "data_yaml": str(data_yaml_path) if data_yaml_path else None,
        },
    )

    trainer = YoloTrainer(checkpoints_dir=args.output_dir)

    print("\n[2/5] Validating Configuration & Resource Estimation...")
    val_report = trainer.validate_config(config)
    if not val_report.valid:
        print(f"[!] Validation Failed: {val_report.errors}")
        return 1
    for warning in val_report.warnings:
        print(f"[?] Warning: {warning}")

    estimate = trainer.estimate(config)
    print(f"      Estimated GPU Hours : {estimate.gpu_hours:.3f} hrs")
    print(f"      Estimated Disk Space: {estimate.storage_bytes / (1024 * 1024):.1f} MB")
    print(f"      Estimated Wall Time : {estimate.wall_time_seconds} s")

    print("\n[3/5] Preparing Dataset & Checking Anti-Leakage...")
    try:
        data_prep = trainer.prepare_data(config)
        print(f"      Manifest ID: {data_prep.manifest_id}")
        print(f"      Lineage OK : {data_prep.lineage_valid}")
    except ValueError as exc:
        print(f"[!] Leakage / Data Error: {exc}")
        return 1

    print("\n[4/5] Running Training Loop...")
    if env["ultralytics_available"] and data_yaml_path:
        print("      >>> Executing Real Ultralytics PyTorch Training Engine on GPU/CPU <<<")
    else:
        print("      >>> Notice: Ultralytics not installed in local env. Executing simulation/validation mode. <<<")
        print("      >>> On Google Colab with `uv pip install -e \".[models-gpu]\"`, this will run full GPU training. <<<")

    start_time = time.perf_counter()

    def on_epoch(epoch: int, metrics: dict[str, float]) -> None:
        loss = metrics.get("loss", 0.0)
        precision = metrics.get("precision", 0.0)
        recall = metrics.get("recall", 0.0)
        map50 = metrics.get("map50", 0.0)
        map50_95 = metrics.get("map50_95", metrics.get("clean_map50_95", 0.0))
        print(
            f"  [Epoch {epoch:02d}/{epochs:02d}] "
            f"Loss: {loss:.4f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"mAP50: {map50:.4f} | "
            f"mAP50-95: {map50_95:.4f}"
        )

    callbacks = TrainerCallbacks(
        on_epoch=on_epoch,
        is_cancelled=lambda: False,
    )

    report = trainer.train(config, callbacks)
    duration = time.perf_counter() - start_time

    print(f"\n[5/5] Training Finished in {duration:.2f}s with State: {report.state}")
    if report.state != "COMPLETED":
        print(f"[!] Errors: {report.errors}")
        return 1

    if report.checkpoint:
        print(f"      Checkpoint Saved : {report.checkpoint.path}")
        print(f"      SHA256 Digest   : {report.checkpoint.sha256}")

    # Acceptance Gate Evaluation (for R1/R2)
    if mode in ("r1", "r2", "robust-mix", "targeted-repair") and report.epoch_metrics:
        final_metrics = report.epoch_metrics[-1]
        baseline_metrics = {"clean_map50_95": 0.685, "robust_score": 62.0}
        gate_result = YoloTrainer.evaluate_acceptance_gate(baseline_metrics, final_metrics)
        print("\n" + "=" * 75)
        print(" CHECKPOINT ACCEPTANCE GATE EVALUATION")
        print("=" * 75)
        print(f"[*] Gate Passed          : {'PASSED (ACCEPT)' if gate_result['passed'] else 'REJECTED'}")
        print(f"[*] Clean Delta          : {gate_result['clean_delta']:+.4f} (Max drop allowed: -0.0200)")
        print(f"[*] RobustScore Delta    : {gate_result['robust_score_delta']:+.2f} (Min gain required: +8.0)")
        print("=" * 75)

    # Save summary report JSON
    summary_path = Path(args.output_dir) / run_id / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    print(f"[✓] Full Report Exported to: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
