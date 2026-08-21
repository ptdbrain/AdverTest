"""Long-running GCE consumer for the AdverTest Pub/Sub job subscription."""

from __future__ import annotations

import json
import os
import signal
from threading import Event

from src.api.platform_dependencies import get_platform_worker
from src.config import get_settings


def load_runtime_secrets() -> None:
    """Load worker-only secrets with the attached GCE service account.

    The Render database URL and GCS HMAC credentials never need to be passed
    through instance metadata or stored in a startup script.
    """
    required = {
        "PLATFORM_DATABASE_URL": "PLATFORM_DATABASE_URL",
        "OBJECT_STORAGE_ACCESS_KEY_ID": "OBJECT_STORAGE_ACCESS_KEY_ID",
        "OBJECT_STORAGE_SECRET_ACCESS_KEY": "OBJECT_STORAGE_SECRET_ACCESS_KEY",
    }
    missing = {key: secret for key, secret in required.items() if not os.environ.get(key)}
    if not missing:
        return

    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is required to load worker secrets")
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    for env_name, secret_name in missing.items():
        version = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
        os.environ[env_name] = client.access_secret_version(name=version).payload.data.decode()


def main() -> None:
    load_runtime_secrets()
    settings = get_settings()
    if not settings.gcp_pubsub_subscription:
        raise RuntimeError("GCP_PUBSUB_SUBSCRIPTION is required for the GCE worker")
    from google.cloud import pubsub_v1

    subscriber = pubsub_v1.SubscriberClient()
    worker = get_platform_worker()
    stop = Event()

    def consume(message: object) -> None:
        try:
            payload = json.loads(message.data.decode("utf-8"))  # type: ignore[attr-defined]
            job_id = str(payload["job_id"])
            worker.process(job_id)
        except Exception:
            message.nack()  # type: ignore[attr-defined]
            return
        message.ack()  # type: ignore[attr-defined]

    stream = subscriber.subscribe(settings.gcp_pubsub_subscription, callback=consume)
    signal.signal(signal.SIGTERM, lambda *_: (stream.cancel(), stop.set()))
    signal.signal(signal.SIGINT, lambda *_: (stream.cancel(), stop.set()))
    try:
        stream.result()
    finally:
        stream.cancel()
        stop.set()


if __name__ == "__main__":
    main()
