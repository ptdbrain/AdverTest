"""Cloud Run dispatcher: authenticated Render requests become Pub/Sub messages."""

from __future__ import annotations

import json
import os
import secrets
from typing import Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AdverTest GCP dispatcher")


class DispatchRequest(BaseModel):
    job_id: str


@app.post("/dispatch", status_code=202)
def dispatch(body: DispatchRequest, authorization: str | None = Header(default=None)) -> dict[str, str]:
    expected = os.environ.get("ADVERTEST_DISPATCH_TOKEN", "")
    received = authorization.removeprefix("Bearer ") if authorization else ""
    if not expected or not secrets.compare_digest(expected, received):
        raise HTTPException(status_code=401, detail="UNAUTHORIZED")
    project_id, topic = os.environ["GOOGLE_CLOUD_PROJECT"], os.environ["ADVERTEST_PUBSUB_TOPIC"]
    vm_state = _ensure_gpu_worker_running(project_id)
    from google.cloud import pubsub_v1

    publisher = pubsub_v1.PublisherClient()
    message_id = publisher.publish(
        publisher.topic_path(project_id, topic),
        json.dumps({"job_id": body.job_id}).encode("utf-8"),
    ).result(timeout=10)
    return {"job_id": body.job_id, "message_id": message_id, "gpu_state": vm_state}


def _ensure_gpu_worker_running(project_id: str) -> Literal["RUNNING", "STARTING"]:
    """Start the stopped GPU VM once; Pub/Sub retains the message during boot."""
    instance_name = os.environ.get("ADVERTEST_GPU_VM_NAME")
    zone = os.environ.get("ADVERTEST_GPU_VM_ZONE")
    if not instance_name or not zone:
        raise RuntimeError("ADVERTEST_GPU_VM_NAME and ADVERTEST_GPU_VM_ZONE are required")
    from google.cloud import compute_v1

    instances = compute_v1.InstancesClient()
    instance = instances.get(project=project_id, zone=zone, instance=instance_name)
    if instance.status == "RUNNING":
        return "RUNNING"
    if instance.status == "TERMINATED":
        instances.start(project=project_id, zone=zone, instance=instance_name)
    return "STARTING"


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}
