#!/usr/bin/env python
"""Test Group F attacks (random_noise_linf + square_attack) on YOLO11s with KITTI images.

Usage:
    uv run python scripts/test_group_f_yolo11.py
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on sys.path when run directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time

import numpy as np

# ── load attacks & adapters ───────────────────────────────────────────────────
from src.attacks import get_attack, load_attacks
from src.attacks.base import AttackContext
from src.adapters import get_adapter
from src.datasets import get_dataset

SEVERITIES = [1, 3, 5]
LIMIT = 4  # số ảnh test
MODEL_NAME = "yolo11s"


def detection_summary(predictions) -> str:
    """One-line summary of detection results."""
    if not predictions or not predictions[0].boxes:
        return "0 detections"
    boxes = predictions[0].boxes
    by_class: dict[str, int] = {}
    total_conf = 0.0
    for b in boxes:
        by_class[b.label] = by_class.get(b.label, 0) + 1
        total_conf += b.score
    parts = [f"{count}×{label}" for label, count in sorted(by_class.items())]
    return f"{len(boxes)} det ({', '.join(parts)})  Σconf={total_conf:.2f}"


def main() -> int:
    print("=" * 72)
    print(f"  Group F attack test  —  model={MODEL_NAME}  limit={LIMIT}")
    print("=" * 72)

    # ── load model ────────────────────────────────────────────────────────
    print(f"\n▸ Loading {MODEL_NAME}...")
    adapter = get_adapter(MODEL_NAME)
    info = adapter.metadata()
    print(f"  Model: {info.name}  task={info.task}  gradients={info.supports_gradients}")

    # ── load dataset ──────────────────────────────────────────────────────
    print(f"\n▸ Loading KITTI dataset (limit={LIMIT})...")
    dataset = get_dataset("kitti_2d")
    samples = dataset.load(LIMIT)
    print(f"  Loaded {len(samples)} samples, shapes: {[s.image.shape for s in samples]}")

    # ── clean predictions ─────────────────────────────────────────────────
    print(f"\n▸ Clean predictions...")
    for sample in samples:
        preds = adapter.predict([sample])
        print(f"  {sample.sample_id}: {detection_summary(preds)}")

    # ── random_noise_linf ─────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  ATTACK: random_noise_linf  (Group F baseline)")
    print("─" * 72)

    attack_rn = get_attack("random_noise_linf")
    for severity in SEVERITIES:
        epsilon = attack_rn.params.epsilon_per_severity[severity - 1]
        print(f"\n  severity={severity}  ε={epsilon:.5f} ({epsilon*255:.1f}/255)")
        for sample in samples:
            ctx = AttackContext(rng=np.random.default_rng(42))
            t0 = time.perf_counter()
            attacked = attack_rn.run(sample, severity, ctx)
            dt = (time.perf_counter() - t0) * 1000

            linf = float(np.max(np.abs(attacked.image - sample.image)))
            l2 = float(np.linalg.norm(attacked.image - sample.image))

            preds_clean = adapter.predict([sample])
            preds_atk = adapter.predict([attacked])

            print(
                f"    {sample.sample_id}  L∞={linf:.5f}  L2={l2:.2f}  "
                f"dt={dt:.0f}ms  "
                f"clean={detection_summary(preds_clean)}  "
                f"attacked={detection_summary(preds_atk)}"
            )

    # ── square_attack ─────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  ATTACK: square_attack  (Group F black-box)")
    print("─" * 72)

    # Use smaller query budget for quick test
    attack_sq = get_attack(
        "square_attack",
        queries_per_severity=(30, 60, 100, 200, 500),
    )
    for severity in SEVERITIES:
        epsilon = attack_sq.params.epsilon_per_severity[severity - 1]
        queries = int(attack_sq.params.queries_per_severity[severity - 1])
        print(f"\n  severity={severity}  ε={epsilon:.5f} ({epsilon*255:.1f}/255)  queries={queries}")
        for sample in samples:
            ctx = AttackContext(rng=np.random.default_rng(42), model=adapter)
            t0 = time.perf_counter()
            attacked = attack_sq.run(sample, severity, ctx)
            dt = (time.perf_counter() - t0) * 1000

            linf = float(np.max(np.abs(attacked.image - sample.image)))
            l2 = float(np.linalg.norm(attacked.image - sample.image))

            preds_clean = adapter.predict([sample])
            preds_atk = adapter.predict([attacked])

            clean_conf = sum(b.score for b in preds_clean[0].boxes) if preds_clean[0].boxes else 0
            atk_conf = sum(b.score for b in preds_atk[0].boxes) if preds_atk[0].boxes else 0

            print(
                f"    {sample.sample_id}  L∞={linf:.5f}  L2={l2:.2f}  "
                f"dt={dt/1000:.1f}s  queries={queries}  "
                f"clean={detection_summary(preds_clean)}  "
                f"attacked={detection_summary(preds_atk)}  "
                f"ΔΣconf={atk_conf - clean_conf:+.2f}"
            )

    print("\n" + "=" * 72)
    print("  Done!")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
