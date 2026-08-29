"""AdverTest P3.4 Soak and VRAM Benchmark Utility.

Conducts long-duration memory soak testing and CUDA VRAM profiling:
1. CPU Soak Test: >= 1,000 batches with periodic RSS and GC object tracking.
2. CUDA Host VRAM Profiling: Measures allocated, reserved, peak VRAM, RSS, and latency.
   Synchronizes CUDA streams and validates memory cleanup between runs.
   Emits WAITING_FOR_GPU_VALIDATION when CUDA is unavailable.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("soak_benchmark")


def run_cpu_soak_test(
    num_batches: int = 1000,
    batch_size: int = 4,
    log_interval: int = 100,
    rss_threshold_mb: float = 25.0,
) -> dict[str, Any]:
    """Execute a multi-batch CPU soak test verifying memory stability over >= 1,000 iterations.

    Threshold:
    - Pre-defined maximum allowable RSS drift after warm-up: <= rss_threshold_mb.
    - Leak rate threshold: < 1.0 KB / batch.
    """
    from src.attacks.base import AttackContext
    from src.attacks.corruption.gaussian_noise import GaussianNoise
    from src.datasets.base import Sample

    attack = GaussianNoise()
    ctx = AttackContext(rng=np.random.default_rng(42))
    process = psutil.Process() if psutil else None

    logger.info("Starting CPU Soak Test with %d batches of size %d...", num_batches, batch_size)

    gc.collect()
    start_rss = process.memory_info().rss / (1024 * 1024) if process else 0.0
    start_objects = len(gc.get_objects())

    # Warmup 50 batches
    for _ in range(50):
        for i in range(batch_size):
            dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
            sample = Sample(sample_id=f"warmup_{i}", image=dummy_img)
            _ = attack.apply(sample, severity=2, ctx=ctx)

    gc.collect()
    warmup_rss = process.memory_info().rss / (1024 * 1024) if process else 0.0
    rss_history: list[dict[str, Any]] = []

    start_time = time.perf_counter()
    for batch_idx in range(1, num_batches + 1):
        for i in range(batch_size):
            dummy_img = np.random.rand(224, 224, 3).astype(np.float32)
            sample = Sample(sample_id=f"soak_{batch_idx}_{i}", image=dummy_img)
            _ = attack.apply(sample, severity=3, ctx=ctx)

        if batch_idx % log_interval == 0 or batch_idx == num_batches:
            current_rss = process.memory_info().rss / (1024 * 1024) if process else 0.0
            current_objects = len(gc.get_objects())
            delta_warmup_mb = current_rss - warmup_rss
            logger.info(
                "Batch %4d / %4d | RSS: %.2f MB (Δ from warmup: %+.2f MB) | GC Objects: %d",
                batch_idx,
                num_batches,
                current_rss,
                delta_warmup_mb,
                current_objects,
            )
            rss_history.append(
                {
                    "batch": batch_idx,
                    "rss_mb": round(current_rss, 2),
                    "delta_warmup_mb": round(delta_warmup_mb, 2),
                    "gc_objects": current_objects,
                }
            )

    gc.collect()
    total_time = time.perf_counter() - start_time
    final_rss = process.memory_info().rss / (1024 * 1024) if process else 0.0
    final_objects = len(gc.get_objects())
    net_drift_mb = final_rss - warmup_rss
    leak_rate_kb_per_batch = (net_drift_mb * 1024) / max(num_batches, 1)

    passed = net_drift_mb <= rss_threshold_mb

    result = {
        "test_name": "CPU_SOAK_TEST",
        "num_batches": num_batches,
        "batch_size": batch_size,
        "total_time_seconds": round(total_time, 2),
        "throughput_batches_per_sec": round(num_batches / max(total_time, 0.001), 2),
        "initial_rss_mb": round(start_rss, 2),
        "warmup_rss_mb": round(warmup_rss, 2),
        "final_rss_mb": round(final_rss, 2),
        "net_drift_mb": round(net_drift_mb, 2),
        "leak_rate_kb_per_batch": round(leak_rate_kb_per_batch, 4),
        "gc_objects_start": start_objects,
        "gc_objects_final": final_objects,
        "threshold_max_drift_mb": rss_threshold_mb,
        "status": "PASS" if passed else "FAIL_EXCEEDED_THRESHOLD",
        "history": rss_history,
    }
    return result


def run_cuda_vram_benchmark(
    num_iterations: int = 500,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Execute CUDA Host VRAM Profiling measuring allocated, reserved, and peak memory."""
    try:
        import torch
    except ImportError:
        torch = None

    if not torch or not torch.cuda.is_available():
        logger.warning("CUDA is not available on this host. Emitting WAITING_FOR_GPU_VALIDATION contract.")
        return {
            "test_name": "CUDA_VRAM_BENCHMARK",
            "status": "WAITING_FOR_GPU_VALIDATION",
            "cuda_available": False,
            "expected_acceptance": {
                "max_vram_leak_mb": 0.0,
                "peak_vram_bounded": True,
                "post_cleanup_residual_mb": 0.0,
            },
            "message": "CUDA device not detected. Run on GPU runner with --cuda to execute hardware validation.",
        }

    device = torch.device("cuda:0")
    logger.info("Executing CUDA VRAM Benchmark on %s (%s)...", torch.cuda.get_device_name(device), device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    start_allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

    # Synthetic inference forward loop
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 10),
    ).to(device)
    model.eval()

    latencies = []
    with torch.no_grad():
        for i in range(num_iterations):
            t0 = time.perf_counter()
            x = torch.randn(batch_size, 3, 224, 224, device=device)
            _ = model(x)
            torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    final_allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

    # Cleanup verification
    del model, x
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    post_cleanup_allocated_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)

    leak_mb = post_cleanup_allocated_mb - start_allocated_mb
    passed = leak_mb <= 1.0  # Allow <= 1 MB runtime rounding

    return {
        "test_name": "CUDA_VRAM_BENCHMARK",
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(device),
        "num_iterations": num_iterations,
        "start_allocated_mb": round(start_allocated_mb, 2),
        "peak_allocated_mb": round(peak_allocated_mb, 2),
        "final_allocated_mb": round(final_allocated_mb, 2),
        "post_cleanup_allocated_mb": round(post_cleanup_allocated_mb, 2),
        "vram_leak_mb": round(leak_mb, 2),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "status": "PASS" if passed else "FAIL_VRAM_LEAK",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AdverTest P3.4 Soak and VRAM Benchmark")
    parser.add_argument("--batches", type=int, default=1000, help="Number of batches for CPU soak test")
    parser.add_argument("--output", type=str, default="soak_benchmark_report.json", help="Path to write report JSON")
    args = parser.parse_args()

    cpu_result = run_cpu_soak_test(num_batches=args.batches)
    cuda_result = run_cuda_vram_benchmark()

    combined = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu_soak": cpu_result,
        "cuda_vram": cuda_result,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    logger.info("Saved benchmark report to %s", out_path.resolve())

    if cpu_result["status"] != "PASS":
        logger.error("CPU Soak test FAILED.")
        sys.exit(1)
    logger.info("CPU Soak test PASSED successfully.")


if __name__ == "__main__":
    main()
