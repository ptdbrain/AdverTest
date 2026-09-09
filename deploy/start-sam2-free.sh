#!/bin/sh
set -eu

export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=65536
export MALLOC_MMAP_THRESHOLD_=65536
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

# The Linux dependency pins select CPU wheels during the Render build. Fail
# before binding the port if a future dependency change reintroduces CUDA.
python -c "import torch, torchvision; assert '+cpu' in torch.__version__; assert '+cpu' in torchvision.__version__"

export DATA_ROOT=/opt/render/project/src/data
export RUNS_ROOT=/opt/render/project/src/runs

exec uvicorn deploy.sam2_service:app --host 0.0.0.0 --port "${PORT:-10000}"
