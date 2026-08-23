"""Contract tests for GPU cold-start state; no GCP credentials required."""

from src.jobs.service import PlatformJobService
from src.persistence.database import PlatformDatabase


def test_waiting_for_gpu_keeps_job_claimable(tmp_path):
    database = PlatformDatabase(f"sqlite:///{tmp_path / 'platform.db'}")
    database.create_schema()
    jobs = PlatformJobService(database)
    job = jobs.create(
        project_id="demo", owner_user_id="user", job_type="benchmark_run", request={}, total_units=1
    )

    assert jobs.waiting_for_gpu(job["id"], "GPU starting") is True
    waiting = jobs.get("demo", job["id"])
    assert waiting["status"] == "QUEUED"
    assert waiting["stage"] == "GPU_STARTING"
    assert jobs.start(job["id"]) is True
    assert jobs.get("demo", job["id"])["status"] == "RUNNING"
