"""Automated P3.4 CPU Soak and Memory Stability Regression Test."""

from __future__ import annotations

from scripts.soak_and_vram_benchmark import run_cpu_soak_test, run_cuda_vram_benchmark


def test_cpu_soak_1000_batches_memory_stability() -> None:
    """P3.4: Run >= 1,000 batches and assert memory stability under pre-defined threshold."""
    result = run_cpu_soak_test(
        num_batches=1000,
        batch_size=2,
        log_interval=250,
        rss_threshold_mb=25.0,
    )

    assert result["status"] == "PASS", f"Soak test failed: {result}"
    assert result["num_batches"] == 1000
    assert result["net_drift_mb"] <= 25.0
    # Leak rate must be bounded by allowable threshold (<= 25 MB / 1000 batches = 25.6 KB / batch)
    assert result["leak_rate_kb_per_batch"] <= 25.6


def test_cuda_vram_benchmark_contract() -> None:
    """P3.4: Verify CUDA VRAM benchmark returns PASS (if CUDA) or WAITING_FOR_GPU_VALIDATION (if CPU host)."""
    result = run_cuda_vram_benchmark(num_iterations=20)
    assert result["status"] in ("PASS", "WAITING_FOR_GPU_VALIDATION")
    if result["status"] == "WAITING_FOR_GPU_VALIDATION":
        assert "expected_acceptance" in result
