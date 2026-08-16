from __future__ import annotations

from src.api.workflow_store import WorkflowJobStore


def test_workflow_events_remain_ordered_after_store_reopen(tmp_path) -> None:
    store = WorkflowJobStore(f"sqlite:///{tmp_path / 'jobs.db'}")

    job_id = store.create_job("generated_dataset", {"seed": 17})
    store.append_event(job_id, "VALIDATING", {"progress_ratio": 0.1})
    store.append_event(job_id, "GENERATING", {"progress_ratio": 0.5})

    reopened = WorkflowJobStore(f"sqlite:///{tmp_path / 'jobs.db'}")

    assert [event["sequence"] for event in reopened.events(job_id)] == [0, 1, 2]
    assert reopened.get_job(job_id)["status"] == "GENERATING"


def test_completed_job_retains_checkpoints_and_relative_artifacts(tmp_path) -> None:
    store = WorkflowJobStore(f"sqlite:///{tmp_path / 'jobs.db'}")
    job_id = store.create_job("generated_dataset", {"dataset_version_id": "dataset-a"})

    store.append_event(job_id, "GENERATING", {"progress_ratio": 0.5})
    store.complete_job(
        job_id,
        {
            "descriptor": {"version_id": "generated-a"},
            "artifact_root": "generated-datasets/job-a",
            "manifest_path": "generated-datasets/job-a/manifest.jsonl",
            "variants": [{"variant_id": "variant-a"}],
            "validation": {"valid": True},
            "lineage": {"report_hash": "lineage-a"},
        },
    )

    job = store.get_job(job_id)

    assert job is not None
    assert job["status"] == "COMPLETED"
    assert store.checkpoints(job_id)[-1]["payload"]["manifest_path"] == "generated-datasets/job-a/manifest.jsonl"
