import json
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.dependencies import get_dataset_import_workers, get_generated_datasets, get_store, get_workflow_store
from src.api.generated_dataset_service import GeneratedDatasetService
from src.api.ingestion_validation import summarize_batch, validate_annotation
from src.api.jobs import SqliteRunStore
from src.api.schemas import AnnotationDocument
from src.api.schemas.dataset import (
    DatasetImportIn,
    GeneratedDatasetCreateIn,
    GeneratedDatasetEventsOut,
    GeneratedDatasetJobOut,
    GeneratedDatasetManifestOut,
    GeneratedDatasetValidationOut,
    GeneratedDatasetVariantsOut,
    UploadBatchCreateIn,
)
from src.api.workflow_store import WorkflowJobStore
from src.config import get_settings
from src.core.hashing import stable_digest
from src.datasets.folder import FolderDataset
from src.datasets.versioning import DatasetIngestor, IngestConfig

router = APIRouter(tags=["Datasets"])


def _upload_root() -> Path:
    root = Path(get_settings().data_root).expanduser().resolve() / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dataset_root() -> Path:
    root = Path(get_settings().data_root).expanduser().resolve() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root


class _DatasetImportError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _perform_dataset_import(
    body: DatasetImportIn, store: SqliteRunStore, workflow_store: WorkflowJobStore, job_id: str | None = None
) -> dict[str, Any]:
    root = Path(body.root).expanduser().resolve()
    if body.task_id != "detection2d":
        raise _DatasetImportError("TASK_DATASET_MISMATCH", "folder import accepts only image plus 2D boxes")
    if not root.is_dir():
        raise _DatasetImportError("DATASET_ROOT_MISSING", "dataset root does not exist")
    source = FolderDataset(
        root=str(root),
        input_format=body.input_format,
        anonymization_manifest=body.anonymization_manifest,
        max_samples=body.max_samples,
    )
    try:
        source.require_anonymized()
    except Exception as exc:
        raise _DatasetImportError("ANONYMISATION_REQUIRED", str(exc)) from exc
    if job_id:
        if workflow_store.cancel_requested(job_id):
            workflow_store.fail_job(job_id, "CANCELLED_BY_USER", cancelled=True)
            raise _DatasetImportError("CANCELLED_BY_USER", "dataset import cancelled")
        workflow_store.append_event(
            job_id, "IMPORTING", {"progress_ratio": 0.45, "detail": "Creating immutable dataset manifest"}
        )
    version = DatasetIngestor(_dataset_root() / "versions").ingest(
        source,
        IngestConfig(
            name=body.name,
            logical_source_id=body.logical_source_id,
            metadata={"input_format": body.input_format, "task_id": body.task_id},
        ),
    )
    payload = version.model_dump(mode="json")
    payload["generation_source"] = {
        "input_dir": str(root),
        "input_format": body.input_format,
        "anonymization_manifest": body.anonymization_manifest,
    }
    stored = store.put_record("dataset_version", version.version_id, payload)
    result = {
        **stored,
        "dataset": "folder_dataset",
        "dataset_params": {
            "root": str(root),
            "input_format": body.input_format,
            "anonymization_manifest": body.anonymization_manifest,
            "max_samples": body.max_samples,
        },
        "anonymized": True,
        "task_id": body.task_id,
        "input_schema": ["image"],
        "annotation_schema": ["boxes2d", "class_labels"],
        "annotation_status": "VALIDATED",
        "benchmark_ready": True,
    }
    if job_id:
        workflow_store.append_event(
            job_id, "FINALIZING", {"progress_ratio": 0.9, "detail": "Registering dataset version"}
        )
    return result


def _run_dataset_import_job(
    job_id: str, body: DatasetImportIn, store: SqliteRunStore, workflow_store: WorkflowJobStore
) -> None:
    if workflow_store.cancel_requested(job_id):
        workflow_store.fail_job(job_id, "CANCELLED_BY_USER", cancelled=True)
        return
    workflow_store.append_event(job_id, "VALIDATING", {"progress_ratio": 0.1, "detail": "Validating source contract"})
    try:
        dataset = _perform_dataset_import(body, store=store, workflow_store=workflow_store, job_id=job_id)
    except _DatasetImportError as exc:
        workflow_store.fail_job(job_id, exc.code)
        return
    except Exception as exc:
        workflow_store.fail_job(job_id, f"DATASET_IMPORT_FAILED:{type(exc).__name__}")
        return
    workflow_store.complete_job(job_id, dataset)


def _workflow_job_out(job_id: str, workflow_store: WorkflowJobStore) -> dict[str, Any]:
    job = workflow_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="WORKFLOW_JOB_UNKNOWN")
    events = workflow_store.events(job_id)
    progress = max((float(event["payload"].get("progress_ratio", 0.0)) for event in events), default=0.0)
    return {
        "job_id": job_id,
        "job_type": job["job_type"],
        "state": job["status"],
        "progress_ratio": progress,
        "detail": (events[-1]["payload"].get("detail") if events else None),
        "can_cancel": job["status"] not in {"COMPLETED", "FAILED", "CANCELLED"},
        "error": job["error"],
        "result": job["result"],
    }


def _require_generated_job(job_id: str, workflow_store: WorkflowJobStore) -> dict[str, Any]:
    item = workflow_store.get_job(job_id)
    if item is None or item["job_type"] != "generated_dataset":
        raise HTTPException(status_code=404, detail=f"unknown generated dataset job {job_id!r}")
    return item


def _generated_job_out(item: dict[str, Any] | None) -> GeneratedDatasetJobOut:
    if item is None:
        raise HTTPException(status_code=404, detail="unknown generated dataset job")
    result = item["result"] or {}
    return GeneratedDatasetJobOut(
        id=item["id"],
        status=item["status"],
        job_type=item["job_type"],
        error=item["error"],
        cancel_requested=item["cancel_requested"],
        descriptor=result.get("descriptor"),
        artifact_root=result.get("artifact_root"),
    )


def _write_annotation_export(batch: dict[str, Any], sample_id: str, document: AnnotationDocument) -> None:
    root = _upload_root() / str(batch["batch_id"])
    labels = root / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    if document.task_id == "detection2d":
        class_map = dict(batch.get("class_map", {}))
        payload = {
            "boxes": [
                {
                    "x1": annotation["bbox_xyxy"][0],
                    "y1": annotation["bbox_xyxy"][1],
                    "x2": annotation["bbox_xyxy"][2],
                    "y2": annotation["bbox_xyxy"][3],
                    "label": class_map[annotation["class_id"]],
                    "score": 1.0,
                }
                for annotation in document.annotations
            ]
        }
    else:
        payload = document.model_dump(mode="json")
    (labels / f"{sample_id}.json").write_text(json.dumps(payload), encoding="utf-8")


@router.post("/uploads/batches", status_code=201)
async def create_upload_batch(body: UploadBatchCreateIn, store: SqliteRunStore = Depends(get_store)) -> dict[str, Any]:
    batch_id = f"batch-{uuid.uuid4().hex[:12]}"
    payload: dict[str, Any] = {
        "batch_id": batch_id,
        "display_name": body.display_name,
        "task_id": body.task_id,
        "class_map": body.class_map,
        "anonymized": body.anonymized,
        "dataset_kind": body.dataset_kind,
        "attacked_manifest": body.attacked_manifest.model_dump(mode="json") if body.attacked_manifest else None,
        "samples": {},
        "status": "DRAFT",
    }
    payload["validation"] = summarize_batch(payload).model_dump(mode="json")
    return store.put_record("upload_batch", batch_id, payload)


@router.get("/uploads/batches/{batch_id}")
async def get_upload_batch(batch_id: str, store: SqliteRunStore = Depends(get_store)) -> dict[str, Any]:
    batch = store.get_record("upload_batch", batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="UPLOAD_BATCH_UNKNOWN")
    return batch


@router.put("/uploads/batches/{batch_id}/annotations/{sample_id}")
async def save_annotation(
    batch_id: str, sample_id: str, body: AnnotationDocument, store: SqliteRunStore = Depends(get_store)
) -> dict[str, Any]:
    batch = store.get_record("upload_batch", batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="UPLOAD_BATCH_UNKNOWN")
    samples = dict(batch.get("samples", {}))
    sample = samples.get(sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="UPLOAD_SAMPLE_UNKNOWN")
    if body.task_id != batch["task_id"]:
        raise HTTPException(status_code=422, detail={"code": "TASK_ANNOTATION_MISMATCH"})
    issues = validate_annotation(body, sample_id=sample_id, sample=sample, class_map=dict(batch.get("class_map", {})))
    if issues:
        raise HTTPException(
            status_code=422,
            detail={"code": "ANNOTATION_INVALID", "issues": [issue.model_dump(mode="json") for issue in issues]},
        )
    sample["annotation"] = body.model_dump(mode="json")
    sample["annotation_status"] = "VALID"
    samples[sample_id] = sample
    batch["samples"] = samples
    _write_annotation_export(batch, sample_id, body)
    batch["validation"] = summarize_batch(batch).model_dump(mode="json")
    return store.update_record("upload_batch", batch_id, batch)


@router.post("/uploads/batches/{batch_id}/finalize", status_code=201)
async def finalize_upload_batch(batch_id: str, store: SqliteRunStore = Depends(get_store)) -> dict[str, Any]:
    batch = store.get_record("upload_batch", batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="UPLOAD_BATCH_UNKNOWN")
    validation = summarize_batch(batch)
    batch["validation"] = validation.model_dump(mode="json")
    if not validation.benchmark_ready:
        store.update_record("upload_batch", batch_id, batch)
        raise HTTPException(
            status_code=422, detail={"code": "DATASET_NOT_BENCHMARK_READY", "validation": batch["validation"]}
        )
    batch["status"] = "FINALIZED"
    stored = store.update_record("upload_batch", batch_id, batch)
    identity = {"batch_id": batch_id, "samples": batch["samples"]}
    version_id = f"dataset-{stable_digest(identity, length=32)}"
    dataset_record = store.put_record(
        "dataset_version",
        version_id,
        {
            "version_id": version_id,
            "source_batch_id": batch_id,
            "name": batch["display_name"],
            "title": batch["display_name"],
            "dataset": "folder_dataset",
            "dataset_params": {"root": str(_upload_root() / batch_id), "input_format": "advertest"},
            "task_id": batch["task_id"],
            "dataset_kind": batch["dataset_kind"],
            "anonymized": batch["anonymized"],
            "validation": batch["validation"],
            "attacked_manifest": batch.get("attacked_manifest"),
            "input_schema": validation.input_schema,
            "annotation_schema": validation.annotation_schema,
            "benchmark_ready": True,
            "paired_comparison_ready": batch["dataset_kind"] == "attacked_paired",
        },
    )
    return {**stored, "dataset_version": dataset_record, "benchmark_ready": True}


def _validate_uploaded_image(payload: bytes) -> tuple[str, tuple[int, int]]:
    from io import BytesIO

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(BytesIO(payload)) as image:
            image.verify()
        with Image.open(BytesIO(payload)) as image:
            return image.format or "unknown", image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_IMAGE",
                "message": "Payload is not a decodable image.",
            },
        ) from exc


@router.post("/uploads/images", status_code=201)
async def upload_image(request: Request, store: SqliteRunStore = Depends(get_store)) -> dict[str, Any]:
    task_id = request.headers.get("x-task-id", "detection2d")
    batch_id = request.headers.get("x-upload-batch-id")
    batch = store.get_record("upload_batch", batch_id) if batch_id else None
    if task_id not in {"detection2d", "segmentation", "detection3d"}:
        raise HTTPException(status_code=422, detail={"code": "TASK_UNKNOWN", "task_id": task_id})
    if task_id != "detection2d" and batch is None:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "ANNOTATIONS_REQUIRED",
                "task_id": task_id,
                "message": "Create a task-bound batch before uploading data that requires annotations.",
            },
        )
    filename = request.headers.get("x-filename", "upload.bin")
    filename = Path(filename).name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", filename):
        raise HTTPException(status_code=422, detail="invalid x-filename")

    payload = await request.body()
    if not payload or len(payload) > 25 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="image payload must be between 1 byte and 25 MiB")

    image_format, (width, height) = _validate_uploaded_image(payload)

    if not batch_id or not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", batch_id):
        batch_id = f"batch-{uuid.uuid4().hex[:12]}"
    batch = store.get_record("upload_batch", batch_id)
    if batch is None:
        batch = {
            "batch_id": batch_id,
            "display_name": batch_id,
            "task_id": task_id,
            "class_map": {},
            "anonymized": False,
            "dataset_kind": "clean",
            "attacked_manifest": None,
            "samples": {},
            "status": "DRAFT",
        }
        batch["validation"] = summarize_batch(batch).model_dump(mode="json")
        store.put_record("upload_batch", batch_id, batch)
    elif batch["task_id"] != task_id:
        raise HTTPException(status_code=422, detail={"code": "TASK_UPLOAD_BATCH_MISMATCH"})

    requested_sample_id = request.headers.get("x-sample-id")
    if requested_sample_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", requested_sample_id):
        raise HTTPException(status_code=422, detail={"code": "INVALID_SAMPLE_ID"})
    sample_id = requested_sample_id or uuid.uuid4().hex[:12]

    target_dir = _upload_root() / batch_id
    images_dir = target_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    stored = images_dir / f"{sample_id}{Path(filename).suffix.lower()}"
    if stored.exists() or sample_id in batch.get("samples", {}):
        raise HTTPException(status_code=409, detail={"code": "DUPLICATE_SAMPLE_ID", "sample_id": sample_id})
    stored.write_bytes(payload)

    manifest = target_dir / "dataset.json"
    if not manifest.is_file():
        descriptor = {
            "anonymized": bool(batch["anonymized"]),
            "annotated": False,
            "benchmark_ready": False,
            "split": "upload",
        }
        manifest.write_text(json.dumps(descriptor), encoding="utf-8")

    samples = dict(batch.get("samples", {}))
    import hashlib

    samples[sample_id] = {
        "sample_id": sample_id,
        "filename": filename,
        "path": str(stored),
        "bytes": len(payload),
        "image_format": image_format,
        "width": width,
        "height": height,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "annotation_status": "MISSING",
    }
    batch["samples"] = samples
    batch["validation"] = summarize_batch(batch).model_dump(mode="json")
    store.update_record("upload_batch", batch_id, batch)

    return {
        "upload_id": sample_id,
        "batch_id": batch_id,
        "sample_id": sample_id,
        "filename": filename,
        "path": str(stored),
        "bytes": len(payload),
        "image_format": image_format,
        "width": width,
        "height": height,
        "benchmark_ready": False,
        "status": "RAW",
        "anonymized": batch["anonymized"],
        "annotation_status": "MISSING",
        "dataset": "folder_dataset",
        "dataset_params": {
            "root": str(target_dir),
            "input_format": "advertest",
        },
        "task_id": task_id,
    }


@router.post("/datasets/import", status_code=201)
async def import_dataset(
    body: DatasetImportIn,
    store: SqliteRunStore = Depends(get_store),
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> dict[str, Any]:
    try:
        return _perform_dataset_import(body, store=store, workflow_store=workflow_store)
    except _DatasetImportError as exc:
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": exc.message}) from exc


@router.post("/datasets/import-jobs", status_code=202)
async def create_dataset_import_job(
    body: DatasetImportIn,
    store: SqliteRunStore = Depends(get_store),
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
    workers: ThreadPoolExecutor = Depends(get_dataset_import_workers),
) -> dict[str, Any]:
    job_id = workflow_store.create_job("dataset_import", body.model_dump(mode="json"))
    workers.submit(_run_dataset_import_job, job_id, body, store, workflow_store)
    return _workflow_job_out(job_id, workflow_store)


@router.get("/datasets/import-jobs/{job_id}")
async def get_dataset_import_job(
    job_id: str, workflow_store: WorkflowJobStore = Depends(get_workflow_store)
) -> dict[str, Any]:
    job = workflow_store.get_job(job_id)
    if job is None or job["job_type"] != "dataset_import":
        raise HTTPException(status_code=404, detail="DATASET_IMPORT_JOB_UNKNOWN")
    return _workflow_job_out(job_id, workflow_store)


@router.post("/datasets/import-jobs/{job_id}/cancel")
async def cancel_dataset_import_job(
    job_id: str, workflow_store: WorkflowJobStore = Depends(get_workflow_store)
) -> dict[str, Any]:
    job = workflow_store.get_job(job_id)
    if job is None or job["job_type"] != "dataset_import":
        raise HTTPException(status_code=404, detail="DATASET_IMPORT_JOB_UNKNOWN")
    workflow_store.request_cancel(job_id)
    return _workflow_job_out(job_id, workflow_store)


@router.post("/generated-datasets", status_code=202, response_model=GeneratedDatasetJobOut)
async def create_generated_dataset(
    body: GeneratedDatasetCreateIn, generated_datasets: GeneratedDatasetService = Depends(get_generated_datasets)
) -> GeneratedDatasetJobOut:
    job_id = generated_datasets.enqueue(body)
    return _generated_job_out(generated_datasets.get(job_id))


@router.get("/generated-datasets/{job_id}", response_model=GeneratedDatasetJobOut)
async def get_generated_dataset(
    job_id: str, generated_datasets: GeneratedDatasetService = Depends(get_generated_datasets)
) -> GeneratedDatasetJobOut:
    return _generated_job_out(generated_datasets.get(job_id))


@router.post("/generated-datasets/{job_id}/cancel", response_model=GeneratedDatasetJobOut)
async def cancel_generated_dataset(
    job_id: str,
    generated_datasets: GeneratedDatasetService = Depends(get_generated_datasets),
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> GeneratedDatasetJobOut:
    if not workflow_store.request_cancel(job_id):
        raise HTTPException(status_code=404, detail=f"unknown generated dataset job {job_id!r}")
    return _generated_job_out(generated_datasets.get(job_id))


@router.get("/generated-datasets/{job_id}/manifest", response_model=GeneratedDatasetManifestOut)
async def get_generated_dataset_manifest(
    job_id: str,
    generated_datasets: GeneratedDatasetService = Depends(get_generated_datasets),
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> GeneratedDatasetManifestOut:
    payload = generated_datasets.manifest(job_id)
    if payload is None:
        _require_generated_job(job_id, workflow_store)
        raise HTTPException(status_code=409, detail="manifest is not available until generation completes")
    return GeneratedDatasetManifestOut(**payload)


@router.get("/generated-datasets/{job_id}/variants", response_model=GeneratedDatasetVariantsOut)
async def get_generated_dataset_variants(
    job_id: str,
    generated_datasets: GeneratedDatasetService = Depends(get_generated_datasets),
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> GeneratedDatasetVariantsOut:
    payload = generated_datasets.variants(job_id)
    if payload is None:
        _require_generated_job(job_id, workflow_store)
        raise HTTPException(status_code=409, detail="variants are not available until generation completes")
    return GeneratedDatasetVariantsOut(**payload)


@router.get("/generated-datasets/{job_id}/events", response_model=GeneratedDatasetEventsOut)
async def get_generated_dataset_events(
    job_id: str, workflow_store: WorkflowJobStore = Depends(get_workflow_store)
) -> GeneratedDatasetEventsOut:
    _require_generated_job(job_id, workflow_store)
    return GeneratedDatasetEventsOut(id=job_id, events=workflow_store.events(job_id))


@router.post("/generated-datasets/{job_id}/validate", response_model=GeneratedDatasetValidationOut)
async def get_generated_dataset_validation(
    job_id: str,
    generated_datasets: GeneratedDatasetService = Depends(get_generated_datasets),
    workflow_store: WorkflowJobStore = Depends(get_workflow_store),
) -> GeneratedDatasetValidationOut:
    payload = generated_datasets.validation(job_id)
    if payload is None:
        _require_generated_job(job_id, workflow_store)
        raise HTTPException(status_code=409, detail="validation is not available until generation completes")
    return GeneratedDatasetValidationOut(**payload)
