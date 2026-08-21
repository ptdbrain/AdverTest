"""Cloud Run checkpoint-inspection endpoint, isolated from the Render API."""

from __future__ import annotations

import base64
import os
import secrets

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.api.checkpoint_service import _inspect_in_sandbox
from src.config import get_settings

app = FastAPI(title="AdverTest checkpoint sandbox")


class InspectRequest(BaseModel):
    checkpoint_b64: str


@app.post("/inspect")
def inspect_checkpoint(body: InspectRequest, authorization: str | None = Header(default=None)) -> dict:
    expected = os.environ.get("CHECKPOINT_SANDBOX_TOKEN", "")
    received = authorization.removeprefix("Bearer ") if authorization else ""
    if not expected or not secrets.compare_digest(expected, received):
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")
    try:
        content = base64.b64decode(body.checkpoint_b64, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="CHECKPOINT_ENCODING_INVALID") from exc
    settings = get_settings()
    return _inspect_in_sandbox(
        content,
        timeout_seconds=settings.checkpoint_validation_timeout_seconds,
        memory_mb=settings.checkpoint_validation_memory_mb,
        cpu_seconds=settings.checkpoint_validation_cpu_seconds,
    )


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}
