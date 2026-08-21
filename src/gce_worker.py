"""Long-running GCE consumer for the AdverTest Pub/Sub job subscription."""

from __future__ import annotations

import json
import signal
from threading import Event

from src.api.platform_dependencies import get_platform_worker
from src.config import get_settings


def main() -> None:
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
