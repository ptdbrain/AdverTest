from __future__ import annotations

from src.evaluation.report import RunReport


def test_confidence_only_report_is_not_eligible_for_promotion() -> None:
    report = RunReport(
        run_id="run-confidence-only",
        model="yolo11s",
        model_version="v1",
        dataset="kitti-2d",
        n_samples=1,
        ap_clean=0.0,
    )

    assert report.evidence.status == "NOT_ELIGIBLE"
    assert report.is_promotion_eligible is False
    assert "simulation_only" in report.evidence.missing


def test_complete_non_simulation_provenance_is_verified() -> None:
    report = RunReport(
        run_id="run-verified",
        model="yolo11s",
        model_version="v1",
        dataset="kitti-2d",
        n_samples=100,
        ap_clean=0.5,
        simulation_only=False,
        provenance={
            "dataset_version_id": "dataset-kitti2d-curated-v1",
            "split_manifest_hash": "split-sha256",
            "ground_truth_hash": "ground-truth-sha256",
            "checkpoint_sha256": "checkpoint-sha256",
            "config_sha256": "config-sha256",
            "artifact_hashes": {"predictions.json": "artifact-sha256"},
            "metric_protocol": "coco-map50-95-v1",
        },
    )

    assert report.evidence.status == "VERIFIED"
    assert report.evidence.missing == ()
    assert report.is_promotion_eligible is True
