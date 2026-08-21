"""Cloud Run dispatcher: authenticated Render requests become Pub/Sub messages."""

from __future__ import annotations

import json
import os
import secrets

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
    from google.cloud import pubsub_v1

    publisher = pubsub_v1.PublisherClient()
    message_id = publisher.publish(
        publisher.topic_path(project_id, topic),
        json.dumps({"job_id": body.job_id}).encode("utf-8"),
    ).result(timeout=10)
    return {"job_id": body.job_id, "message_id": message_id}


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}
