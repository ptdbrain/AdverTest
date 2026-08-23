"""Worker-side dispatch for platform job types."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from src.api.checkpoint_service import PlatformCheckpointService
from src.config import get_settings
from src.jobs.service import PlatformJobService
from src.pipeline.runner import RunConfig, TestRunner
from src.storage.base import ArtifactStorage
from src.storage.export_service import AttackedDatasetExportService


class PlatformWorker:
    """Consumes IDs from a queue and resolves all state through the database."""

    def __init__(
        self,
        jobs: PlatformJobService,
        checkpoints: PlatformCheckpointService,
        exports: AttackedDatasetExportService,
        storage: ArtifactStorage | None = None,
    ) -> None:
        self._jobs = jobs
        self._checkpoints = checkpoints
        self._exports = exports
        self._storage = storage
        self._runner = TestRunner()

    def process(self, job_id: str) -> None:
        job = self._jobs.request_for_worker(job_id)
        if job is None or not self._jobs.start(job_id):
            return
        try:
            result = self._run(job)
        except KeyError as exc:
            self._jobs.fail(job_id, str(exc).strip("'"), "Referenced platform record was not found")
            return
        except ValueError as exc:
            self._jobs.fail(job_id, str(exc), str(exc))
            return
        except Exception as exc:
            self._jobs.fail(job_id, "JOB_RUNTIME_FAILED", f"{type(exc).__name__}: {exc}")
            return
        self._jobs.complete(job_id, result)
        validation_job_id = result.get("validation_job_id") if isinstance(result, dict) else None
        if validation_job_id:
            self.process(str(validation_job_id))

    def _run(self, job: dict[str, Any]) -> dict[str, Any]:
        def progress(*, stage: str, completed: int, total: int, message: str) -> bool:
            return self._jobs.progress(job["id"], stage=stage, completed=completed, total=total, message=message)

        if job["type"] == "checkpoint_validation":
            return self._checkpoints.validate_job(job["id"], job["request"], progress)
        if job["type"] == "ultralytics_import":
            return self._checkpoints.import_ultralytics_job(
                job["id"], job["project_id"], job["owner_user_id"], job["request"], progress
            )
        if job["type"] == "attacked_dataset_export":
            progress(stage="EXPORTING", completed=1, total=2, message="Building attacked dataset archive")
            result = self._exports.run(project_id=job["project_id"], actor_id=job["owner_user_id"], request=job["request"])
            progress(stage="PERSISTING", completed=2, total=2, message="Export artifact stored")
            return result
        if job["type"] == "benchmark_run":
            config = RunConfig.model_validate(job["request"])
            settings = get_settings()
            evidence_root = Path(settings.runs_root).expanduser().resolve() / "platform-evidence" / job["id"]
            config = config.model_copy(update={"evidence_dir": str(evidence_root)})
            if config.model == "yolo11":
                adapter_params = dict(config.adapter_params)
                adapter_params.update(
                    {
                        "device": settings.model_device,
                        "half": settings.model_half_precision,
                        "batch_size": settings.model_batch_size,
                    }
                )
                config = config.model_copy(update={"adapter_params": adapter_params})

            estimate = self._runner.estimate(config)
            total = max(1, estimate.n_cells)

            def run_progress(stage: str, detail: dict[str, Any]) -> None:
                completed = int(detail.get("completed_cells", 0))
                progress(
                    stage=stage,
                    completed=min(completed, total),
                    total=total,
                    message=str(detail.get("attack") or detail.get("phase") or stage),
                )

            report = self._runner.run(
                config,
                progress=run_progress,
                should_cancel=lambda: self._jobs.cancel_requested(job["id"]),
                run_id=job["id"],
            )
            result = report.as_dict()
            if self._storage is not None:
                self._publish_evidence(
                    job["id"], result, evidence_root, settings.object_storage_signed_url_ttl_seconds,
                    settings.object_storage_bucket, settings.object_storage_endpoint_url,
                )
            return result
        raise ValueError("JOB_TYPE_UNSUPPORTED")

    def _publish_evidence(
        self, job_id: str, report: dict[str, Any], root: Path, ttl_seconds: int,
        bucket: str, endpoint_url: str | None,
    ) -> None:
        """Move worker-local PNG evidence to object storage before returning a report."""
        assert self._storage is not None
        path_fields = (
            "clean_image_path",
            "attacked_image_path",
            "clean_prediction_path",
            "attacked_prediction_path",
        )
        for sample in report.get("sample_results", []):
            for field in path_fields:
                value = sample.get(field)
                if not value:
                    continue
                local_path = Path(str(value)).resolve()
                if not local_path.is_file() or root not in local_path.parents:
                    raise RuntimeError(f"invalid evidence path: {local_path}")
                key = f"runs/{job_id}/evidence/{local_path.relative_to(root).as_posix()}"
                content = local_path.read_bytes()
                if endpoint_url == "https://storage.googleapis.com":
                    _put_gcs_bytes(bucket, key, content)
                else:
                    self._storage.put_bytes(key, content, mime_type="image/png")
                sample[field] = self._storage.signed_download_url(key, ttl_seconds)


def _put_gcs_bytes(bucket: str, key: str, content: bytes) -> None:
    """Upload worker-produced evidence with the attached GCE service account.

    GCS's JSON API avoids the incompatible HMAC PUT signature seen on this
    bucket while the configured S3 client remains responsible for signed reads.
    """
    metadata_url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
    token_request = Request(metadata_url, headers={"Metadata-Flavor": "Google"})
    with urlopen(token_request, timeout=10) as response:  # nosec B310 - fixed metadata endpoint
        token = json.loads(response.read())["access_token"]
    upload_url = (
        f"https://storage.googleapis.com/upload/storage/v1/b/{quote(bucket, safe='')}/o"
        f"?uploadType=media&name={quote(key, safe='')}"
    )
    request = Request(upload_url, data=content, method="POST")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Content-Type", "image/png")
    with urlopen(request, timeout=30):  # nosec B310 - fixed GCS endpoint
        pass
