"""Evidence classification must not overstate simulated measurements."""

from __future__ import annotations

from src.evaluation.report import RunReport


def _report(**overrides: object) -> RunReport:
    values: dict[str, object] = {
        "run_id": "evidence-run",
        "model": "pointpillars",
        "model_version": "1",
        "dataset": "kitti_val",
        "n_samples": 10,
        "ap_clean": 0.7,
    }
    values.update(overrides)
    return RunReport(**values)


def test_non_simulation_report_requires_reproducibility_provenance() -> None:
    incomplete = _report(simulation_only=False)
    assert incomplete.evidence.status == "NOT_ELIGIBLE"
    assert incomplete.is_promotion_eligible is False

    report = _report(
        simulation_only=False,
        provenance={
            "protocol_hash": "kitti-r40-locked",
            "metric_protocol": "kitti-r40",
            "dataset_version_id": "kitti-curated-v1",
            "ground_truth_hash": "d" * 64,
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "split_manifest_hash": "c" * 64,
            "artifact_hashes": {"predictions.json": "e" * 64},
        },
    )
    assert report.as_dict()["simulation_only"] is False
    assert report.evidence.status == "VERIFIED"
