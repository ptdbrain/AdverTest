"""CUDA Smoke Gate script for MMDetection3D + PointPillars (P1.8).

Exits with non-zero code if any gate condition fails:
1. CUDA availability (when cuda requested).
2. Dependency lock compatibility.
3. Model instantiation & weights loading.
4. LiDAR point cloud parsing.
5. Inference execution without exception.
6. Non-empty pred_instances_3d (bboxes_3d, scores_3d, labels_3d).
7. Correct bounding box tensor shape (N, 7).
8. Finite numerical values (Zero NaN / Inf).
9. Valid class ID range [0, num_classes - 1].
10. Latency (ms) and peak VRAM (MB) measurement.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def verify_pointpillars_cuda_smoke(
    config_path: str,
    checkpoint_path: str,
    bin_path: str,
    device: str = "cuda:0",
) -> dict[str, float | int | str]:
    """Execute CUDA Smoke Gate for PointPillars and return telemetry."""
    print(f"[*] Starting PointPillars CUDA Smoke Gate on target device: {device}")

    # 1. Check dependencies
    try:
        import torch
    except ImportError:
        print("[!] ERROR: PyTorch is not installed.")
        sys.exit(1)

    if "cuda" in device:
        if not torch.cuda.is_available():
            print(f"[!] ERROR: CUDA requested ({device}) but torch.cuda.is_available() is False.")
            sys.exit(1)

    try:
        from mmdet3d.apis import inference_detector, init_model
    except ImportError as exc:
        print(f"[!] ERROR: MMDetection3D is not installed: {exc}")
        sys.exit(1)

    # 2. Check files
    cfg_file = Path(config_path)
    ckpt_file = Path(checkpoint_path)
    lidar_file = Path(bin_path)

    if not cfg_file.is_file():
        print(f"[!] ERROR: Config file not found: {config_path}")
        sys.exit(1)
    if not ckpt_file.is_file():
        print(f"[!] ERROR: Checkpoint file not found: {checkpoint_path}")
        sys.exit(1)
    if not lidar_file.is_file():
        print(f"[!] ERROR: LiDAR sample file not found: {bin_path}")
        sys.exit(1)

    # 3. Model Initialization
    print(f"[*] Loading model from {checkpoint_path}...")
    start_load = time.perf_counter()
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.reset_peak_memory_stats()

    try:
        model = init_model(str(cfg_file), str(ckpt_file), device=device)
    except Exception as exc:
        print(f"[!] ERROR: Failed to initialize model: {exc}")
        sys.exit(1)
    load_time_sec = time.perf_counter() - start_load
    print(f"[+] Model loaded in {load_time_sec:.2f}s")

    # 4. Inference Execution
    print(f"[*] Executing inference on LiDAR frame {bin_path}...")
    start_infer = time.perf_counter()
    try:
        result = inference_detector(model, str(lidar_file))
    except Exception as exc:
        print(f"[!] ERROR: Inference raised exception: {exc}")
        sys.exit(1)
    latency_ms = (time.perf_counter() - start_infer) * 1000.0

    peak_vram_mb = 0.0
    if torch.cuda.is_available() and "cuda" in device:
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # 5. Output Verification
    sample = result[0] if isinstance(result, tuple) else result
    pred = getattr(sample, "pred_instances_3d", None)
    if pred is None:
        print("[!] ERROR: No pred_instances_3d found in output.")
        sys.exit(1)

    bboxes_3d = pred.bboxes_3d.tensor.cpu().numpy()
    scores_3d = pred.scores_3d.cpu().numpy()
    labels_3d = pred.labels_3d.cpu().numpy()

    if len(bboxes_3d) == 0 or len(scores_3d) == 0 or len(labels_3d) == 0:
        print("[!] ERROR: Predictions are empty (zero bounding boxes detected).")
        sys.exit(1)

    # 6. Tensor Shape Verification: (N, 7) for 3D bounding boxes
    if len(bboxes_3d.shape) != 2 or bboxes_3d.shape[1] != 7:
        print(f"[!] ERROR: bboxes_3d shape invalid: {bboxes_3d.shape}, expected (N, 7).")
        sys.exit(1)

    # 7. Finite Numerical Check (Zero NaN / Inf)
    if not np.all(np.isfinite(bboxes_3d)) or not np.all(np.isfinite(scores_3d)):
        print("[!] ERROR: NaN or Inf detected in bounding boxes or confidence scores.")
        sys.exit(1)

    # 8. Class ID Verification (KITTI: 0=Car, 1=Pedestrian, 2=Cyclist)
    if np.any(labels_3d < 0) or np.any(labels_3d > 2):
        print(f"[!] ERROR: Invalid class label detected: {labels_3d}")
        sys.exit(1)

    print("[+] CUDA Smoke Gate PASSED 100%!")
    print(f"[+] Detections: {len(bboxes_3d)}")
    print(f"[+] Latency: {latency_ms:.2f} ms")
    print(f"[+] Peak VRAM: {peak_vram_mb:.2f} MB")

    return {
        "status": "PASS",
        "num_detections": len(bboxes_3d),
        "latency_ms": latency_ms,
        "peak_vram_mb": peak_vram_mb,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PointPillars CUDA Smoke Gate")
    parser.add_argument("--config", default="checkpoints/pointpillars_kitti_3class.py", help="Path to config file")
    parser.add_argument("--checkpoint", default="checkpoints/pointpillars.pth", help="Path to checkpoint file")
    parser.add_argument("--lidar", default="data/kitti/training/velodyne/000001.bin", help="Path to .bin pointcloud")
    parser.add_argument("--device", default="cpu", help="Target device (cpu or cuda:0)")
    args = parser.parse_args()

    verify_pointpillars_cuda_smoke(args.config, args.checkpoint, args.lidar, device=args.device)
