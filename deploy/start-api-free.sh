#!/bin/sh
set -eu

# Constrain glibc arenas and math thread pools to prevent memory fragmentation
# and thread explosion on Render's 512MB free container.
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=65536
export MALLOC_MMAP_THRESHOLD_=65536
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export SAVE_NUMPY_EVIDENCE=0
export CUDA_VISIBLE_DEVICES=""

# Ensure PyTorch CPU is used on Render 512MB free container.
# CUDA PyTorch wheels allocate ~350MB RSS on import; CPU wheels allocate ~50MB RSS.
if ! python -c "import torch; assert '+cpu' in torch.__version__" 2>/dev/null; then
    echo "Installing PyTorch CPU wheels and removing CUDA libraries to prevent OOM on 512MB container..."
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip uninstall -y nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-nccl-cu12 nvidia-nvjitlink-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 triton 2>/dev/null || true
fi

# Use the committed 100-image kitti2d_100 bundle and the real YOLO11s checkpoint.
# Portable demo mode is disabled so the scanner picks up runs/train/yolo_b0
# and the catalog resolver picks up data/drive_export_100/detection2d/kitti2d_100.
export BOOTSTRAP_PORTABLE_DEMO=false
export DEMO_BOOTSTRAP_ACCOUNTS=true
export DATA_ROOT=/opt/render/project/src/data
export RUNS_ROOT=/opt/render/project/src/runs

# Free Render instances do not support preDeployCommand. A single API instance
# can safely migrate before it begins accepting preview traffic.
alembic upgrade head
exec uvicorn src.main:app --host 0.0.0.0 --port "${PORT:-10000}"
