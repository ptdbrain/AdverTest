# AdverTest Platform Dependency & 3D Model Compatibility Matrix

This document defines the strict, locked compatibility matrix for 3D model runtimes (MMDetection3D + PointPillars) and foundation model architectures in AdverTest.

---

## 1. MMDetection3D Locked Compatibility Matrix (P1.6)

| Component | Locked Version / Range | Notes |
|---|---|---|
| **Python** | `3.10.x` / `3.11.x` | CPython runtime standard |
| **PyTorch** | `2.1.2` / `2.2.0` | Built with CUDA 11.8 or CUDA 12.1 support |
| **CUDA Toolkit** | `11.8` / `12.1` | Required for PointPillars ops compilation |
| **MMCV** | `2.1.0` (`openmim install mmcv==2.1.0`) | Pre-compiled wheel for PyTorch 2.1/2.2 |
| **MMEngine** | `0.10.3` | OpenMMLab runner foundation |
| **MMDetection** | `3.3.0` | 2D/3D head loss components |
| **MMDetection3D** | `1.4.0` | PointPillars, VoxelNet, LiDAR backbones |
| **Numpy** | `< 2.0.0` (`>= 1.24.0, < 2.0.0`) | Compatibility with PyTorch C-extensions |

---

## 2. Model & Checkpoint Provenance

### PointPillars KITTI 3-Class
- **Architecture**: `VoxelNet` / `PointPillarsScatter` / `SECOND` / `SECFPN` / `Anchor3DHead`
- **Configuration File**: [`checkpoints/pointpillars_kitti_3class.py`](file:///d:/Project/AIthucchien/P-195/checkpoints/pointpillars_kitti_3class.py)
- **Target Dataset**: KITTI 3D Object Detection (LiDAR Velodyne `.bin`)
- **Native Class Schema**:
  1. `Car` (Class ID 0, Anchor range `[0, -39.68, -1.78, 69.12, 39.68, -1.78]`, IoU Threshold: 0.7)
  2. `Pedestrian` (Class ID 1, Anchor range `[0, -39.68, -0.6, 69.12, 39.68, -0.6]`, IoU Threshold: 0.5)
  3. `Cyclist` (Class ID 2, Anchor range `[0, -39.68, -0.6, 69.12, 39.68, -0.6]`, IoU Threshold: 0.5)
- **Preflight Error Codes**:
  - `MMDET3D_NOT_INSTALLED`: MMDetection3D runtime missing in environment.
  - `CONFIG_NOT_FOUND`: Specified model config path missing.
  - `CHECKPOINT_NOT_FOUND`: Checkpoint weights file missing.
  - `LIDAR_SAMPLE_NOT_FOUND`: Test point cloud frame missing.

---

## 3. Supported Model Families & Validator Contracts (P1.7)

| Model Family ID | Task | Checkpoint Extension | Output Contract | Verification Mechanism |
|---|---|---|---|---|
| `yolo11` | `detection2d` | `.pt` | `ultralytics-results-v1` | Isolated process + Ultralytics names verification |
| `sam2` | `segmentation` | `.pt`, `.pth` | `sam2-mask-decoder-v1` | `torch.load(..., weights_only=True)` + finite tensor checks |
| `pointpillars` | `detection3d` | `.pth` | `mmdet3d-pointpillars-v1` | State dict inspection + 3D anchor & class schema validation |
