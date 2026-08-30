#!/usr/bin/env python3
"""AdversAI Lab - Standalone Local Defence & Adversarial Training Script.

Run a registered local trainer on a workstation to perform robust fine-tuning
or adversarial training without overloading the central evaluation server.
This script intentionally does not fabricate checkpoints or scientific metrics:
use ``--demo`` only for UI/demo plumbing and upload only real trainer output.

Example Usage:
    python scripts/train_defence.py \\
        --model weights/yolo11s.pt \\
        --dataset data/anonymized/kitti-de/data.yaml \\
        --recipe fgsm,depth_fog \\
        --strategy adversarial_training \\
        --epochs 10 \\
        --batch-size 16 \\
        --lr 0.001 \\
        --output-dir weights/defended
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AdversAI Lab Local Defence & Robust Fine-tuning Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to baseline model checkpoint (.pt, .pth, or .onnx)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/anonymized/kitti-de/data.yaml",
        help="Path to dataset configuration (data.yaml or directory)",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default="fgsm,depth_fog",
        help="Comma-separated list of attacks to defend against (e.g. fgsm,depth_fog,pgd)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["adversarial_training", "augmentation_mix", "robust_finetune"],
        default="adversarial_training",
        help="Defence strategy to apply",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights/defended",
        help="Directory where defended weights will be saved",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        help="Compute device ('0', '1', 'cpu')",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Emit explicitly non-scientific display estimates without writing a checkpoint.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.demo:
        print(
            json.dumps(
                {
                    "source_kind": "demo",
                    "scientific_evidence": False,
                    "demo_display_estimates": {
                        "clean_metric": 0.0,
                        "attacked_metric": 0.0,
                        "recovery_rate": 0.0,
                    },
                    "next_action": "Run a registered trainer and a locked ground-truth benchmark before uploading a checkpoint.",
                },
                sort_keys=True,
            )
        )
        return 0

    print("=" * 70)
    print("[DEFENCE] ADVERSAI LAB - DEFENCE & ROBUST FINE-TUNING RUNNER")
    print("=" * 70)
    print(f"[*] Base Checkpoint : {args.model}")
    print(f"[*] Dataset Config  : {args.dataset}")
    print(f"[*] Attack Recipe   : {args.recipe}")
    print(f"[*] Strategy        : {args.strategy.upper()}")
    print(f"[*] Hyperparams     : Epochs={args.epochs}, BatchSize={args.batch_size}, LR={args.lr}")
    print(f"[*] Compute Device  : {args.device}")
    print("=" * 70)

    print("[ERROR] TRAINER_NOT_AVAILABLE")
    print("No registered trainer is wired to scripts/train_defence.py in this runtime.")
    print("Refusing to manufacture a defended checkpoint, loss curve, AP, or recovery claim.")
    print("Use an installed registered trainer, then benchmark its checkpoint on the locked protocol.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
