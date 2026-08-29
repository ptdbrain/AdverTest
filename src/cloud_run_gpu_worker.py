"""Cloud Run GPU endpoint consuming Pub/Sub jobs without database access."""

from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from fastapi import Request as FastAPIRequest

from src.config import get_settings
from src.demo_bootstrap import ensure_demo_catalog, ensure_demo_checkpoint, ensure_demo_kitti
from src.pipeline.runner import RunConfig, TestRunner


def _bootstrap_demo_assets() -> None:
    """Prepare the disposable Cloud Run filesystem once per cold start."""
    settings = get_settings()
    if not (settings.bootstrap_demo_model or settings.bootstrap_demo_kitti or settings.bootstrap_demo_catalog):
        return
    from src.api.platform_dependencies import get_platform_storage

    storage = get_platform_storage()
    if settings.bootstrap_demo_model:
        ensure_demo_checkpoint(enabled=True, checkpoint_root=settings.checkpoint_root,
                               model_id=settings.bootstrap_demo_model_id, storage=storage,
                               storage_key=settings.demo_model_storage_key)
    if settings.bootstrap_demo_kitti:
        kitti_root = ensure_demo_kitti(enabled=True, storage=storage,
                                       storage_prefix=settings.demo_kitti_storage_prefix,
                                       data_root=settings.data_root)
        if kitti_root is not None:
            import os
            os.environ["ADVERTEST_KITTI_ROOT"] = str(kitti_root)
    if settings.bootstrap_demo_catalog:
        ensure_demo_catalog(enabled=True, storage=storage,
                            storage_prefix=settings.demo_catalog_storage_prefix,
                            data_root=settings.data_root)


class ControlPlaneClient:
    """Small authenticated client; Render remains the only PostgreSQL writer."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.worker_callback_api_url or not settings.worker_callback_token:
            raise RuntimeError("WORKER_CALLBACK_API_URL and WORKER_CALLBACK_TOKEN are required")
        self._base_url = settings.worker_callback_api_url.rstrip("/")
        self._token = settings.worker_callback_token

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
        content = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(f"{self._base_url}{path}", data=content, method=method)
        request.add_header("Authorization", f"Bearer {self._token}")
        if content is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urlopen(request, timeout=30) as response:  # nosec B310 - operator configured API URL
                body = response.read()
                return json.loads(body) if body else None
        except HTTPError as exc:
            if exc.code == 409:
                return None
            raise RuntimeError(f"CONTROL_PLANE_HTTP_{exc.code}") from exc

    def claim(self, job_id: str) -> dict[str, Any] | None:
        return self._request("POST", f"/api/v1/internal/worker/jobs/{job_id}/claim")

    def progress(self, job_id: str, *, stage: str, completed: int, total: int, message: str) -> None:
        self._request("POST", f"/api/v1/internal/worker/jobs/{job_id}/progress", {
            "stage": stage, "completed": completed, "total": total, "message": message,
        })

    def complete(self, job_id: str, result: dict[str, Any]) -> None:
        self._request("POST", f"/api/v1/internal/worker/jobs/{job_id}/complete", {"result": result})

    def fail(self, job_id: str, error_code: str, error_message: str) -> None:
        self._request("POST", f"/api/v1/internal/worker/jobs/{job_id}/fail", {
            "error_code": error_code, "error_message": error_message,
        })

    def cancelled(self, job_id: str) -> bool:
        response = self._request("GET", f"/api/v1/internal/worker/jobs/{job_id}/cancelled")
        return bool(response and response["cancel_requested"])


class CloudRunBenchmarkWorker:
    def __init__(self) -> None:
        self._control = ControlPlaneClient()
        self._runner = TestRunner()

    def process(self, job_id: str) -> None:
        job = self._control.claim(job_id)
        if job is None:  # duplicate Pub/Sub delivery or a cancelled job
            return
        try:
            config = RunConfig.model_validate(job["request"])
            settings = get_settings()
            evidence_root = Path(settings.runs_root).expanduser().resolve() / "platform-evidence" / job_id
            config = config.model_copy(update={"evidence_dir": str(evidence_root)})
            if config.model == "yolo11":
                params = dict(config.adapter_params)
                params.update({"device": settings.model_device, "half": settings.model_half_precision,
                               "batch_size": settings.model_batch_size})
                config = config.model_copy(update={"adapter_params": params})

            total = max(1, self._runner.estimate(config).n_cells)

            def run_progress(stage: str, detail: dict[str, Any]) -> None:
                self._control.progress(job_id, stage=stage,
                    completed=min(int(detail.get("completed_cells", 0)), total), total=total,
                    message=str(detail.get("attack") or detail.get("phase") or stage))

            report = self._runner.run(config, progress=run_progress,
                                      should_cancel=lambda: self._control.cancelled(job_id), run_id=job_id)
            result = report.as_dict()
            self._publish_evidence(job_id, result, evidence_root)
            self._control.complete(job_id, result)
        except Exception as exc:
            self._control.fail(job_id, "JOB_RUNTIME_FAILED", f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _publish_evidence(job_id: str, report: dict[str, Any], root: Path) -> None:
        """Upload through the Cloud Run service account, then generate read URLs."""
        from src.api.platform_dependencies import get_platform_storage

        settings, storage = get_settings(), get_platform_storage()
        fields = ("clean_image_path", "attacked_image_path", "clean_prediction_path", "attacked_prediction_path")
        for sample in report.get("sample_results", []):
            for field in fields:
                value = sample.get(field)
                if not value:
                    continue
                local_path = Path(str(value)).resolve()
                if not local_path.is_file() or root not in local_path.parents:
                    raise RuntimeError(f"invalid evidence path: {local_path}")
                key = f"runs/{job_id}/evidence/{local_path.relative_to(root).as_posix()}"
                content = local_path.read_bytes()
                if settings.object_storage_endpoint_url == "https://storage.googleapis.com":
                    _put_gcs_bytes(settings.object_storage_bucket, key, content)
                else:
                    storage.put_bytes(key, content, mime_type="image/png")
                sample[field] = storage.signed_download_url(key, settings.object_storage_signed_url_ttl_seconds)


def _put_gcs_bytes(bucket: str, key: str, content: bytes) -> None:
    """Upload using attached IAM identity, never an HMAC secret in the worker."""
    from urllib.parse import quote

    token_request = Request("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
                            headers={"Metadata-Flavor": "Google"})
    with urlopen(token_request, timeout=10) as response:  # nosec B310 - fixed GCP metadata host
        token = json.loads(response.read())["access_token"]
    upload_url = (f"https://storage.googleapis.com/upload/storage/v1/b/{quote(bucket, safe='')}/o"
                  f"?uploadType=media&name={quote(key, safe='')}")
    request = Request(upload_url, data=content, method="POST")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Content-Type", "image/png")
    with urlopen(request, timeout=30):  # nosec B310 - fixed Google endpoint
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    _bootstrap_demo_assets()
    app.state.worker = CloudRunBenchmarkWorker()
    yield


app = FastAPI(title="AdverTest Cloud Run GPU worker", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/pubsub")
async def consume_pubsub_push(request: FastAPIRequest) -> dict[str, bool]:
    payload: dict[str, Any] = await request.json()
    encoded = payload.get("message", {}).get("data")
    if not isinstance(encoded, str):
        raise HTTPException(status_code=400, detail="PUBSUB_MESSAGE_DATA_REQUIRED")
    try:
        job_id = str(json.loads(base64.b64decode(encoded).decode("utf-8"))["job_id"])
    except (KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="PUBSUB_MESSAGE_INVALID") from exc
    request.app.state.worker.process(job_id)
    return {"ok": True}
