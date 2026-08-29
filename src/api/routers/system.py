"""System and runtime hardware introspection router."""

from __future__ import annotations

import os
from typing import Any

import torch
from fastapi import APIRouter

router = APIRouter(prefix="/system", tags=["System & Runtime"])


@router.get("/runtime-specs")
async def get_runtime_specs() -> dict[str, Any]:
    """Auto-detect available computing hardware, GPU/CPU resources, and recommend runtime hyperparameters."""
    has_cuda = torch.cuda.is_available()
    device_name = "CPU"
    total_vram_gb = 0.0
    free_vram_gb = 0.0
    cuda_version = torch.version.cuda if has_cuda else None

    if has_cuda:
        try:
            device_name = torch.cuda.get_device_name(0)
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            total_vram_gb = round(total_bytes / (1024 ** 3), 2)
            free_vram_gb = round(free_bytes / (1024 ** 3), 2)
        except Exception:
            device_name = "NVIDIA CUDA Device"

    cpu_count = os.cpu_count() or 4

    # Calculate recommended batch size based on available compute
    if has_cuda and total_vram_gb >= 16:
        recommended_batch_size = 32
        recommended_precision = "FP16"
    elif has_cuda and total_vram_gb >= 8:
        recommended_batch_size = 16
        recommended_precision = "FP16"
    elif has_cuda and total_vram_gb >= 4:
        recommended_batch_size = 8
        recommended_precision = "FP16"
    elif has_cuda:
        recommended_batch_size = 4
        recommended_precision = "FP32"
    else:
        recommended_batch_size = 2
        recommended_precision = "FP32"

    return {
        "has_cuda": has_cuda,
        "device_target": "cuda:0" if has_cuda else "cpu",
        "device_name": device_name,
        "total_vram_gb": total_vram_gb,
        "free_vram_gb": free_vram_gb,
        "cuda_version": cuda_version,
        "cpu_count": cpu_count,
        "recommended_batch_size": recommended_batch_size,
        "recommended_precision": recommended_precision,
        "recommended_workers": min(4, cpu_count),
        "supported_precisions": ["FP16", "FP32"] if has_cuda else ["FP32"],
    }
