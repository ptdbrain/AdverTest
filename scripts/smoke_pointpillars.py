"""Smoke script demonstrating standalone MMDetection3D + PointPillars inference on LiDAR data."""

from __future__ import annotations

import argparse
from pathlib import Path


def smoke_pointpillars(config_path: str, checkpoint_path: str, bin_path: str, device: str = "cpu") -> None:
    """Run standalone smoke inference with PointPillars to prove MMDet3D integration."""
    try:
        from mmdet3d.apis import inference_detector, init_model
    except ImportError as exc:
        print(f"[!] MMDetection3D is not installed in the active environment: {exc}")
        print("[!] To run real GPU inference, install mmdet3d into your virtual environment.")
        return

    cfg_file = Path(config_path)
    ckpt_file = Path(checkpoint_path)
    lidar_file = Path(bin_path)

    if not cfg_file.is_file():
        print(f"[!] Config file not found: {config_path}")
        return
    if not ckpt_file.is_file():
        print(f"[!] Checkpoint file not found: {checkpoint_path}")
        return
    if not lidar_file.is_file():
        print(f"[!] LiDAR .bin file not found: {bin_path}")
        return

    print(f"[*] Loading PointPillars model from {checkpoint_path} on {device}...")
    model = init_model(str(cfg_file), str(ckpt_file), device=device)
    print(f"[*] Running inference on LiDAR sample {bin_path}...")
    result = inference_detector(model, str(lidar_file))

    sample = result[0] if isinstance(result, tuple) else result
    pred = getattr(sample, "pred_instances_3d", None)

    if pred is None:
        print("[!] No 3D predictions found in output object.")
        return

    bboxes_3d = pred.bboxes_3d.tensor.cpu().numpy()
    scores_3d = pred.scores_3d.cpu().numpy()
    labels_3d = pred.labels_3d.cpu().numpy()

    print(f"[+] Successfully inferred {len(bboxes_3d)} 3D detections.")
    print(f"[+] Boxes shape: {bboxes_3d.shape}")
    print(f"[+] Scores shape: {scores_3d.shape}")
    print(f"[+] Labels shape: {labels_3d.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for PointPillars 3D detection")
    parser.add_argument("--config", default="configs/pointpillars_kitti.py", help="Path to model config")
    parser.add_argument("--checkpoint", default="checkpoints/pointpillars.pth", help="Path to .pth checkpoint")
    parser.add_argument("--lidar", default="data/kitti/training/velodyne/000001.bin", help="Path to .bin file")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda:0)")
    args = parser.parse_args()

    smoke_pointpillars(args.config, args.checkpoint, args.lidar, device=args.device)
