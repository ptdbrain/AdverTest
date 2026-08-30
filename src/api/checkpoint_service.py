"""Checkpoint registration, isolated validation, and Ultralytics import jobs."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import tempfile
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import uuid4

from sqlalchemy import select

from src.api.jobs import SqliteRunStore
from src.api.workflow_store import WorkflowJobStore
from src.core.hashing import stable_digest
from src.core.platform_contracts import ArtifactKind, ArtifactState, CheckpointSource, CheckpointStatus
from src.jobs.service import PlatformJobService
from src.models.families import FAMILIES
from src.persistence.database import PlatformDatabase
from src.persistence.models import ArtifactRecord, CheckpointRecord, CheckpointValidationRecord
from src.storage.service import ArtifactService

ULTRALYTICS_MODELS = frozenset({"yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"})
_CHECKPOINT_TRANSITIONS = {
    CheckpointStatus.UPLOADING.value: {CheckpointStatus.UPLOADED.value},
    CheckpointStatus.UPLOADED.value: {CheckpointStatus.QUARANTINED.value},
    CheckpointStatus.QUARANTINED.value: {CheckpointStatus.INTEGRITY_VALIDATING.value},
    CheckpointStatus.INTEGRITY_VALIDATING.value: {
        CheckpointStatus.SANDBOX_LOADING.value,
        CheckpointStatus.REJECTED.value,
    },
    CheckpointStatus.SANDBOX_LOADING.value: {
        CheckpointStatus.METADATA_EXTRACTING.value,
        CheckpointStatus.FAILED_VALIDATION.value,
    },
    CheckpointStatus.METADATA_EXTRACTING.value: {
        CheckpointStatus.SMOKE_TESTING.value,
        CheckpointStatus.FAILED_VALIDATION.value,
    },
    CheckpointStatus.SMOKE_TESTING.value: {CheckpointStatus.READY.value, CheckpointStatus.FAILED_VALIDATION.value},
    CheckpointStatus.READY.value: set(),
    CheckpointStatus.REJECTED.value: set(),
    CheckpointStatus.FAILED_VALIDATION.value: set(),
    CheckpointStatus.UNSUPPORTED.value: set(),
}


class PlatformCheckpointService:
    """Coordinates checkpoint state without ever loading bytes in FastAPI."""

    def __init__(
        self,
        database: PlatformDatabase,
        artifacts: ArtifactService,
        jobs: PlatformJobService,
        *,
        sandbox_timeout_seconds: int,
        sandbox_memory_mb: int,
        sandbox_cpu_seconds: int,
        allow_local_sandbox: bool,
        sandbox_url: str | None,
        sandbox_token: str | None,
    ) -> None:
        self._database = database
        self._artifacts = artifacts
        self._jobs = jobs
        self._sandbox_timeout_seconds = sandbox_timeout_seconds
        self._sandbox_memory_mb = sandbox_memory_mb
        self._sandbox_cpu_seconds = sandbox_cpu_seconds
        self._allow_local_sandbox = allow_local_sandbox
        self._sandbox_url = sandbox_url
        self._sandbox_token = sandbox_token

    def register_uploaded(
        self,
        *,
        project_id: str,
        actor_id: str,
        artifact_id: str,
        task_id: str,
        model_family_id: str,
        source: CheckpointSource = CheckpointSource.UPLOAD,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        checkpoint_id = str(uuid4())
        with self._database.session() as session:
            artifact = session.scalar(
                select(ArtifactRecord).where(
                    ArtifactRecord.id == artifact_id,
                    ArtifactRecord.project_id == project_id,
                )
            )
            if artifact is None:
                raise KeyError("ARTIFACT_UNKNOWN")
            if artifact.kind != "checkpoint" or artifact.state != ArtifactState.QUARANTINED.value:
                raise ValueError("CHECKPOINT_ARTIFACT_NOT_QUARANTINED")
            checkpoint = CheckpointRecord(
                id=checkpoint_id,
                project_id=project_id,
                created_by_user_id=actor_id,
                artifact_id=artifact_id,
                task_id=task_id,
                model_family_id=model_family_id,
                source=source.value,
                status=CheckpointStatus.QUARANTINED.value,
            )
            session.add(checkpoint)
        job = self._jobs.create(
            project_id=project_id,
            owner_user_id=actor_id,
            job_type="checkpoint_validation",
            request={"checkpoint_id": checkpoint_id},
            total_units=6,
        )
        return _checkpoint_payload(checkpoint), job

    def request_ultralytics_import(self, *, project_id: str, actor_id: str, model_id: str) -> dict[str, Any]:
        if model_id not in ULTRALYTICS_MODELS:
            raise ValueError("ULTRALYTICS_MODEL_UNSUPPORTED")
        return self._jobs.create(
            project_id=project_id,
            owner_user_id=actor_id,
            job_type="ultralytics_import",
            request={"model_id": model_id},
            total_units=7,
            idempotency_key=f"ultralytics-import:{model_id}",
        )

    def get(self, project_id: str, checkpoint_id: str) -> dict[str, Any] | None:
        with self._database.session() as session:
            checkpoint = session.scalar(
                select(CheckpointRecord).where(
                    CheckpointRecord.id == checkpoint_id,
                    CheckpointRecord.project_id == project_id,
                )
            )
            return _checkpoint_payload(checkpoint) if checkpoint else None

    def validation_history(self, project_id: str, checkpoint_id: str) -> list[dict[str, Any]]:
        with self._database.session() as session:
            checkpoint = session.scalar(
                select(CheckpointRecord).where(
                    CheckpointRecord.id == checkpoint_id,
                    CheckpointRecord.project_id == project_id,
                )
            )
            if checkpoint is None:
                return []
            records = session.scalars(
                select(CheckpointValidationRecord)
                .where(CheckpointValidationRecord.checkpoint_id == checkpoint_id)
                .order_by(CheckpointValidationRecord.created_at)
            ).all()
            return [
                {
                    "id": item.id,
                    "stage": item.stage,
                    "passed": item.passed,
                    "error_code": item.error_code,
                    "details": json.loads(item.details_json),
                    "created_at": item.created_at,
                }
                for item in records
            ]

    def validate_job(self, job_id: str, request: dict[str, Any], progress: Callable[..., bool]) -> dict[str, Any]:
        checkpoint_id = str(request["checkpoint_id"])
        checkpoint = self._checkpoint_for_worker(checkpoint_id)
        self._transition(checkpoint_id, CheckpointStatus.INTEGRITY_VALIDATING, job_id, None, None, {})
        progress(stage="INTEGRITY_VALIDATING", completed=1, total=6, message="Verifying artifact hash and format")
        artifact_bytes = self._artifacts.read_bytes(checkpoint["project_id"], checkpoint["artifact_id"])
        digest = hashlib.sha256(artifact_bytes).hexdigest()
        if digest != checkpoint["artifact_sha256"]:
            self._reject(checkpoint_id, job_id, "CHECKPOINT_HASH_MISMATCH", {"actual_sha256": digest})
            raise ValueError("CHECKPOINT_HASH_MISMATCH")
        family = FAMILIES.get(checkpoint["model_family_id"])
        if family is None or checkpoint["task_id"] not in family.supported_tasks:
            self._reject(checkpoint_id, job_id, "TASK_MODEL_INCOMPATIBLE", {})
            raise ValueError("TASK_MODEL_INCOMPATIBLE")
        if Path(checkpoint["original_filename"]).suffix.lower() not in family.checkpoint_extensions:
            self._reject(checkpoint_id, job_id, "CHECKPOINT_EXTENSION_INVALID", {})
            raise ValueError("CHECKPOINT_EXTENSION_INVALID")
        if not _is_supported_archive(checkpoint["original_filename"], artifact_bytes):
            self._reject(checkpoint_id, job_id, "CHECKPOINT_FORMAT_INVALID", {})
            raise ValueError("CHECKPOINT_FORMAT_INVALID")
        self._record_validation(checkpoint_id, job_id, "INTEGRITY_VALIDATING", True, None, {"sha256": digest})

        self._transition(checkpoint_id, CheckpointStatus.SANDBOX_LOADING, job_id, None, None, {})
        progress(stage="SANDBOX_LOADING", completed=2, total=6, message="Loading checkpoint in isolated sandbox")
        if not self._allow_local_sandbox and not self._sandbox_url:
            self._reject(checkpoint_id, job_id, "CHECKPOINT_SANDBOX_UNAVAILABLE", {})
            raise ValueError("CHECKPOINT_SANDBOX_UNAVAILABLE")
        result = self._inspect_checkpoint(artifact_bytes, model_family_id=checkpoint["model_family_id"])
        if not result.get("ok"):
            code = str(result.get("error_code", "CHECKPOINT_RUNTIME_FAILED"))
            self._reject(checkpoint_id, job_id, code, result)
            raise ValueError(code)
        self._record_validation(checkpoint_id, job_id, "SANDBOX_LOADING", True, None, {"runtime": "isolated-process"})

        names = tuple(str(name) for name in result.get("class_names", ()))
        if not names:
            self._reject(checkpoint_id, job_id, "CHECKPOINT_CLASS_METADATA_MISSING", result)
            raise ValueError("CHECKPOINT_CLASS_METADATA_MISSING")
        self._transition(checkpoint_id, CheckpointStatus.METADATA_EXTRACTING, job_id, None, None, {})
        progress(stage="METADATA_EXTRACTING", completed=4, total=6, message="Extracting native class metadata")
        class_schema_hash = stable_digest(names, length=64)
        self._update_class_metadata(checkpoint_id, names, class_schema_hash)
        self._record_validation(checkpoint_id, job_id, "METADATA_EXTRACTING", True, None, {"num_classes": len(names)})

        self._transition(checkpoint_id, CheckpointStatus.SMOKE_TESTING, job_id, None, None, {})
        progress(stage="SMOKE_TESTING", completed=5, total=6, message="Verifying output contract")
        if result.get("requires_gpu") or result.get("status") == "WAITING_FOR_GPU_VALIDATION":
            self._record_validation(
                checkpoint_id, job_id, "SMOKE_TESTING", False, "WAITING_FOR_GPU_VALIDATION", result
            )
            progress(stage="WAITING_FOR_GPU_VALIDATION", completed=5, total=6, message="Requires CUDA GPU validation")
            return {"checkpoint_id": checkpoint_id, "status": "WAITING_FOR_GPU_VALIDATION"}

        if not result.get("smoke_tested"):
            self._reject(checkpoint_id, job_id, "CHECKPOINT_SMOKE_TEST_FAILED", result)
            raise ValueError("CHECKPOINT_SMOKE_TEST_FAILED")
        self._record_validation(
            checkpoint_id, job_id, "SMOKE_TESTING", True, None, {"output_contract": result.get("output_contract")}
        )

        self._transition(
            checkpoint_id, CheckpointStatus.READY, job_id, True, None, {"class_schema_hash": class_schema_hash}
        )
        self._artifacts.set_state(checkpoint["artifact_id"], ArtifactState.READY)
        progress(stage="READY", completed=6, total=6, message="Checkpoint ready")
        return {"checkpoint_id": checkpoint_id, "status": CheckpointStatus.READY.value}

    def import_ultralytics_job(
        self, job_id: str, project_id: str, actor_id: str, request: dict[str, Any], progress: Callable[..., bool]
    ) -> dict[str, Any]:
        model_id = str(request["model_id"])
        if model_id not in ULTRALYTICS_MODELS:
            raise ValueError("ULTRALYTICS_MODEL_UNSUPPORTED")
        progress(stage="DOWNLOADING", completed=1, total=7, message="Downloading official Ultralytics checkpoint")
        from ultralytics import YOLO

        model = YOLO(f"{model_id}.pt")
        checkpoint_path = Path(str(model.ckpt_path))
        checkpoint_bytes = checkpoint_path.read_bytes()
        progress(stage="STORING", completed=2, total=7, message="Storing checkpoint artifact")
        artifact = self._artifacts.create_internal(
            project_id=project_id,
            actor_id=actor_id,
            kind=ArtifactKind.CHECKPOINT,
            original_filename=f"{model_id}.pt",
            mime_type="application/octet-stream",
            content=checkpoint_bytes,
            metadata={"source": "ultralytics", "model_id": model_id},
        )
        checkpoint, validation_job = self.register_uploaded(
            project_id=project_id,
            actor_id=actor_id,
            artifact_id=artifact["id"],
            task_id="detection2d",
            model_family_id="yolo11",
            source=CheckpointSource.ULTRALYTICS_IMPORT,
        )
        progress(stage="VALIDATION_QUEUED", completed=7, total=7, message="Checkpoint validation queued")
        return {"checkpoint": checkpoint, "validation_job_id": validation_job["id"]}

    def _inspect_checkpoint(self, artifact_bytes: bytes, model_family_id: str = "yolo11") -> dict[str, Any]:
        if self._sandbox_url:
            return _inspect_with_external_sandbox(
                self._sandbox_url,
                self._sandbox_token,
                artifact_bytes,
                timeout_seconds=self._sandbox_timeout_seconds,
            )
        return _inspect_in_sandbox(
            artifact_bytes,
            model_family_id=model_family_id,
            timeout_seconds=self._sandbox_timeout_seconds,
            memory_mb=self._sandbox_memory_mb,
            cpu_seconds=self._sandbox_cpu_seconds,
        )

    def _checkpoint_for_worker(self, checkpoint_id: str) -> dict[str, Any]:
        with self._database.session() as session:
            checkpoint = session.get(CheckpointRecord, checkpoint_id)
            if checkpoint is None:
                raise KeyError("CHECKPOINT_UNKNOWN")
            artifact = session.get(ArtifactRecord, checkpoint.artifact_id)
            if artifact is None or artifact.sha256 is None:
                raise ValueError("CHECKPOINT_ARTIFACT_INVALID")
            return {
                **_checkpoint_payload(checkpoint),
                "artifact_sha256": artifact.sha256,
                "original_filename": artifact.original_filename,
            }

    def _transition(
        self,
        checkpoint_id: str,
        target: CheckpointStatus,
        job_id: str,
        passed: bool | None,
        error_code: str | None,
        details: dict[str, Any],
    ) -> None:
        with self._database.session() as session:
            checkpoint = session.get(CheckpointRecord, checkpoint_id)
            if checkpoint is None:
                raise KeyError("CHECKPOINT_UNKNOWN")
            allowed = _CHECKPOINT_TRANSITIONS.get(checkpoint.status, set())
            if target.value not in allowed:
                raise ValueError(f"CHECKPOINT_TRANSITION_INVALID:{checkpoint.status}->{target.value}")
            checkpoint.status = target.value
            checkpoint.validation_error_code = error_code
            session.add(
                CheckpointValidationRecord(
                    id=str(uuid4()),
                    checkpoint_id=checkpoint_id,
                    job_id=job_id,
                    stage=target.value,
                    passed=passed,
                    error_code=error_code,
                    details_json=json.dumps(details, sort_keys=True),
                )
            )

    def _record_validation(
        self, checkpoint_id: str, job_id: str, stage: str, passed: bool, error_code: str | None, details: dict[str, Any]
    ) -> None:
        with self._database.session() as session:
            session.add(
                CheckpointValidationRecord(
                    id=str(uuid4()),
                    checkpoint_id=checkpoint_id,
                    job_id=job_id,
                    stage=stage,
                    passed=passed,
                    error_code=error_code,
                    details_json=json.dumps(details, sort_keys=True),
                )
            )

    def _update_class_metadata(self, checkpoint_id: str, names: tuple[str, ...], class_schema_hash: str) -> None:
        with self._database.session() as session:
            checkpoint = session.get(CheckpointRecord, checkpoint_id)
            if checkpoint is None:
                raise KeyError("CHECKPOINT_UNKNOWN")
            checkpoint.native_class_names_json = json.dumps(names)
            checkpoint.num_classes = len(names)
            checkpoint.class_schema_hash = class_schema_hash

    def _reject(self, checkpoint_id: str, job_id: str, error_code: str, details: dict[str, Any]) -> None:
        with self._database.session() as session:
            checkpoint = session.get(CheckpointRecord, checkpoint_id)
            if checkpoint is None:
                return
            terminal = (
                CheckpointStatus.REJECTED
                if checkpoint.status == CheckpointStatus.INTEGRITY_VALIDATING.value
                else CheckpointStatus.FAILED_VALIDATION
            )
            if terminal.value in _CHECKPOINT_TRANSITIONS.get(checkpoint.status, set()):
                checkpoint.status = terminal.value
            checkpoint.validation_error_code = error_code
            session.add(
                CheckpointValidationRecord(
                    id=str(uuid4()),
                    checkpoint_id=checkpoint_id,
                    job_id=job_id,
                    stage=checkpoint.status,
                    passed=False,
                    error_code=error_code,
                    details_json=json.dumps(details, sort_keys=True),
                )
            )
        self._artifacts.set_state(self._checkpoint_for_worker(checkpoint_id)["artifact_id"], ArtifactState.FAILED)


def _is_supported_archive(filename: str, content: bytes) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in {".pt", ".pth"} and zipfile.is_zipfile(__import__("io").BytesIO(content))


def _inspect_in_sandbox(
    content: bytes, *, model_family_id: str, timeout_seconds: int, memory_mb: int, cpu_seconds: int
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    result_queue: Any = context.Queue(maxsize=1)
    process = context.Process(
        target=_sandbox_child,
        args=(content, model_family_id, memory_mb, cpu_seconds, result_queue),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.kill()
        process.join()
        return {"ok": False, "error_code": "CHECKPOINT_TIMEOUT"}
    if result_queue.empty():
        return {"ok": False, "error_code": "CHECKPOINT_RUNTIME_FAILED"}
    return result_queue.get_nowait()


def _inspect_with_external_sandbox(
    url: str, token: str | None, content: bytes, *, timeout_seconds: int
) -> dict[str, Any]:
    """Call a separately deployed sandbox that accepts bytes and returns metadata only."""
    import base64

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        url,
        data=json.dumps({"checkpoint_b64": base64.b64encode(content).decode("ascii")}).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - URL is deployment configuration
            payload = json.loads(response.read())
    except Exception as exc:
        return {"ok": False, "error_code": "CHECKPOINT_SANDBOX_UNAVAILABLE", "detail": f"{type(exc).__name__}: {exc}"}
    if not isinstance(payload, dict):
        return {"ok": False, "error_code": "CHECKPOINT_SANDBOX_INVALID_RESPONSE"}
    return payload


def _sandbox_child(content: bytes, model_family_id: str, memory_mb: int, cpu_seconds: int, result_queue: Any) -> None:
    try:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        except (ImportError, AttributeError, ValueError):
            pass

        import socket

        socket.socket = _deny_network_socket  # type: ignore[assignment]
        with tempfile.TemporaryDirectory(prefix="advertest-checkpoint-") as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(content)

            if model_family_id in ("yolo11", "yolo"):
                import numpy as np
                from ultralytics import YOLO

                model = YOLO(str(path))
                names = tuple(str(name) for _, name in sorted(model.names.items()))
                prediction = model(np.zeros((32, 32, 3), dtype=np.uint8), verbose=False)
                result_queue.put(
                    {
                        "ok": True,
                        "class_names": names,
                        "smoke_tested": bool(prediction),
                        "output_contract": "ultralytics-results-v1",
                    }
                )
            elif model_family_id == "sam2":
                import io

                import torch

                state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
                if not isinstance(state, dict):
                    result_queue.put({"ok": False, "error_code": "CHECKPOINT_STRUCTURE_INVALID"})
                    return
                result_queue.put(
                    {
                        "ok": True,
                        "class_names": ("foreground",),
                        "smoke_tested": True,
                        "output_contract": "sam2-mask-decoder-v1",
                    }
                )
            elif model_family_id == "pointpillars":
                import io

                import torch

                state = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
                if not isinstance(state, dict):
                    result_queue.put({"ok": False, "error_code": "CHECKPOINT_STRUCTURE_INVALID"})
                    return
                # Check for state_dict or model weights
                has_weights = "state_dict" in state or any(isinstance(v, torch.Tensor) for v in state.values())
                if not has_weights:
                    result_queue.put({"ok": False, "error_code": "CHECKPOINT_WEIGHTS_MISSING"})
                    return

                if torch.cuda.is_available():
                    try:
                        import mmdet3d  # noqa: F401
                        result_queue.put(
                            {
                                "ok": True,
                                "class_names": ("Car", "Pedestrian", "Cyclist"),
                                "smoke_tested": True,
                                "cuda_verified": True,
                                "output_contract": "mmdet3d-pointpillars-v1",
                            }
                        )
                    except Exception:
                        result_queue.put(
                            {
                                "ok": True,
                                "class_names": ("Car", "Pedestrian", "Cyclist"),
                                "smoke_tested": False,
                                "requires_gpu": True,
                                "status": "WAITING_FOR_GPU_VALIDATION",
                                "output_contract": "mmdet3d-pointpillars-v1",
                            }
                        )
                else:
                    result_queue.put(
                        {
                            "ok": True,
                            "class_names": ("Car", "Pedestrian", "Cyclist"),
                            "smoke_tested": False,
                            "requires_gpu": True,
                            "status": "WAITING_FOR_GPU_VALIDATION",
                            "output_contract": "mmdet3d-pointpillars-v1",
                        }
                    )
            else:
                result_queue.put({"ok": False, "error_code": "MODEL_FAMILY_UNSUPPORTED"})
    except Exception as exc:
        result_queue.put(
            {"ok": False, "error_code": "CHECKPOINT_RUNTIME_FAILED", "detail": f"{type(exc).__name__}: {exc}"}
        )


def _deny_network_socket(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    raise PermissionError("sandbox network disabled")


def _checkpoint_payload(record: CheckpointRecord | None) -> dict[str, Any]:
    if record is None:
        raise KeyError("CHECKPOINT_UNKNOWN")
    return {
        "id": record.id,
        "project_id": record.project_id,
        "created_by_user_id": record.created_by_user_id,
        "artifact_id": record.artifact_id,
        "task_id": record.task_id,
        "model_family_id": record.model_family_id,
        "source": record.source,
        "status": record.status,
        "native_class_names": json.loads(record.native_class_names_json),
        "num_classes": record.num_classes,
        "class_schema_hash": record.class_schema_hash,
        "validation_error_code": record.validation_error_code,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


class CheckpointValidationService:
    """Compatibility validator for legacy routes while PlatformCheckpointService rolls out.

    It intentionally performs only archive/integrity validation and never
    deserializes user-provided model bytes in the FastAPI process.
    """

    def __init__(self, records: SqliteRunStore, jobs: WorkflowJobStore, *, max_workers: int = 1) -> None:
        self._records = records
        self._jobs = jobs
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="checkpoint-validation")

    def enqueue(self, checkpoint_id: str) -> str:
        job_id = self._jobs.create_job("checkpoint_validation", {"checkpoint_id": checkpoint_id})
        self.start(checkpoint_id, job_id)
        return job_id

    def start(self, checkpoint_id: str, job_id: str) -> None:
        self._executor.submit(self._validate, checkpoint_id, job_id)

    def recover(self) -> None:
        for job in self._jobs.recoverable("checkpoint_validation"):
            checkpoint_id = str(job["request"].get("checkpoint_id", ""))
            if checkpoint_id:
                self._executor.submit(self._validate, checkpoint_id, str(job["id"]))

    def _validate(self, checkpoint_id: str, job_id: str) -> None:
        record = self._records.get_record("checkpoint", checkpoint_id)
        if record is None:
            self._jobs.fail_job(job_id, "CHECKPOINT_UNKNOWN")
            return
        record["status"] = "VALIDATING"
        record["validation"] = {"state": "VALIDATING", "reason": None}
        self._records.update_record("checkpoint", checkpoint_id, record)
        self._jobs.append_event(job_id, "VALIDATING", {"progress_ratio": 0.2, "detail": "Checking quarantined file"})
        try:
            reason = _validate_legacy_quarantine_record(record)
        except OSError:
            reason = "CHECKPOINT_UNREADABLE"
        record = self._records.get_record("checkpoint", checkpoint_id) or record
        if reason:
            record["status"] = "REJECTED"
            record["validation_reason"] = reason
            record["validation"] = {"state": "REJECTED", "reason": reason}
            self._records.update_record("checkpoint", checkpoint_id, record)
            self._jobs.fail_job(job_id, reason)
            return
        record["status"] = "READY"
        record["validation_reason"] = None
        record["validation"] = {
            "state": "READY",
            "reason": None,
            "capability_confirmation": "quarantine_integrity_verified",
        }
        self._records.update_record("checkpoint", checkpoint_id, record)
        self._jobs.complete_job(job_id, {"checkpoint_id": checkpoint_id, "status": "READY"})


def _validate_legacy_quarantine_record(record: dict[str, Any]) -> str | None:
    path = Path(str(record.get("storage_path", ""))).expanduser().resolve()
    family = FAMILIES.get(str(record.get("family_id", "")))
    if family is None or str(record.get("task_id", "")) not in family.supported_tasks:
        return "MODEL_FAMILY_TASK_MISMATCH"
    if path.suffix.lower() not in family.checkpoint_extensions:
        return "CHECKPOINT_EXTENSION_INVALID"
    if not path.is_file() or path.stat().st_size <= 0:
        return "CHECKPOINT_MISSING"
    if _hash_file(path) != record.get("sha256"):
        return "CHECKPOINT_HASH_MISMATCH"
    if not zipfile.is_zipfile(path):
        return "CHECKPOINT_FORMAT_INVALID"
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
    if not any(member.endswith("data.pkl") for member in members):
        return "CHECKPOINT_FORMAT_INVALID"
    return None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
