"""Cloud Run GPU endpoint that consumes authenticated Pub/Sub push messages."""

from __future__ import annotations

import base64
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from src.api.platform_dependencies import get_platform_worker
from src.config import get_settings
from src.demo_bootstrap import ensure_demo_catalog, ensure_demo_checkpoint, ensure_demo_kitti
from src.gce_worker import load_runtime_secrets


def _bootstrap_demo_assets() -> None:
    """Prepare the disposable Cloud Run filesystem once per cold start."""
    settings = get_settings()
    if not (settings.bootstrap_demo_model or settings.bootstrap_demo_kitti or settings.bootstrap_demo_catalog):
        return
    from src.api.platform_dependencies import get_platform_storage

    storage = get_platform_storage()
    if settings.bootstrap_demo_model:
        ensure_demo_checkpoint(
            enabled=True, checkpoint_root=settings.checkpoint_root,
            model_id=settings.bootstrap_demo_model_id, storage=storage,
            storage_key=settings.demo_model_storage_key,
        )
    if settings.bootstrap_demo_kitti:
        kitti_root = ensure_demo_kitti(
            enabled=True, storage=storage, storage_prefix=settings.demo_kitti_storage_prefix,
            data_root=settings.data_root,
        )
        if kitti_root is not None:
            import os
            os.environ["ADVERTEST_KITTI_ROOT"] = str(kitti_root)
    if settings.bootstrap_demo_catalog:
        ensure_demo_catalog(
            enabled=True, storage=storage, storage_prefix=settings.demo_catalog_storage_prefix,
            data_root=settings.data_root,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_runtime_secrets()
    _bootstrap_demo_assets()
    app.state.worker = get_platform_worker()
    yield


app = FastAPI(title="AdverTest Cloud Run GPU worker", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/pubsub")
async def consume_pubsub_push(request: Request) -> dict[str, bool]:
    """Handle one Pub/Sub push delivery and acknowledge only after persistence."""
    payload: dict[str, Any] = await request.json()
    encoded = payload.get("message", {}).get("data")
    if not isinstance(encoded, str):
        raise HTTPException(status_code=400, detail="PUBSUB_MESSAGE_DATA_REQUIRED")
    try:
        message = json.loads(base64.b64decode(encoded).decode("utf-8"))
        job_id = str(message["job_id"])
    except (KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="PUBSUB_MESSAGE_INVALID") from exc

    request.app.state.worker.process(job_id)
    return {"ok": True}
