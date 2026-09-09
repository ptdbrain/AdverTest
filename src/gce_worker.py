"""Long-running GCE consumer for the AdverTest Pub/Sub job subscription."""

from __future__ import annotations

import json
import os
import signal
from threading import Event, Lock, Timer

from src.api.platform_dependencies import get_platform_worker
from src.config import get_settings
from src.demo_bootstrap import ensure_demo_catalog, ensure_demo_checkpoint, ensure_demo_kitti


def _metadata(path: str) -> str:
    """Read an identity field from the GCE metadata server."""
    from urllib.request import Request, urlopen

    request = Request(
        f"http://metadata.google.internal/computeMetadata/v1/{path}",
        headers={"Metadata-Flavor": "Google"},
    )
    with urlopen(request, timeout=2) as response:  # noqa: S310 - fixed GCE metadata host
        return response.read().decode("utf-8")


def stop_this_instance() -> None:
    """Stop rather than suspend: GCE does not support suspending GPU VMs."""
    from google.cloud import compute_v1

    project = _metadata("project/project-id")
    zone = _metadata("instance/zone").rsplit("/", 1)[-1]
    instance = _metadata("instance/name")
    compute_v1.InstancesClient().stop(project=project, zone=zone, instance=instance)


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
    storage = None
    if settings.bootstrap_demo_model:
        from src.api.platform_dependencies import get_platform_storage

        storage = get_platform_storage()
        checkpoint = ensure_demo_checkpoint(
            enabled=True,
            checkpoint_root=settings.checkpoint_root,
            model_id=settings.bootstrap_demo_model_id,
            storage=storage,
            storage_key=settings.demo_model_storage_key,
        )
        print(f"Demo checkpoint ready: {checkpoint}")
    if settings.bootstrap_demo_kitti:
        from src.api.platform_dependencies import get_platform_storage

        kitti_root = ensure_demo_kitti(
            enabled=True,
            storage=storage or get_platform_storage(),
            storage_prefix=settings.demo_kitti_storage_prefix,
            data_root=settings.data_root,
        )
        if kitti_root is not None:
            os.environ["ADVERTEST_KITTI_ROOT"] = str(kitti_root)
            print(f"Demo KITTI ready: {kitti_root}")
    if settings.bootstrap_demo_catalog:
        catalog_root = ensure_demo_catalog(
            enabled=True,
            storage=storage or get_platform_storage(),
            storage_prefix=settings.demo_catalog_storage_prefix,
            data_root=settings.data_root,
        )
        if catalog_root is not None:
            os.environ["ADVERTEST_DEMO_CATALOG_ROOT"] = str(catalog_root)
            print(f"Demo catalog ready: {catalog_root}")
    if not settings.gcp_pubsub_subscription:
        raise RuntimeError("GCP_PUBSUB_SUBSCRIPTION is required for the GCE worker")
    from google.cloud import pubsub_v1

    subscriber = pubsub_v1.SubscriberClient()
    worker = get_platform_worker()
    stop = Event()
    idle_seconds = int(os.environ.get("GPU_IDLE_SHUTDOWN_SECONDS", "900"))
    active_lock = Lock()
    active_messages = 0
    idle_timer: Timer | None = None

    def cancel_idle_timer() -> None:
        nonlocal idle_timer
        if idle_timer is not None:
            idle_timer.cancel()
            idle_timer = None

    def stop_if_idle() -> None:
        nonlocal idle_timer
        with active_lock:
            idle_timer = None
            if active_messages or stop.is_set():
                return
        try:
            print(f"GPU worker idle for {idle_seconds}s; stopping this VM")
            stop_this_instance()
        except Exception as exc:  # keep worker alive and retry on a later idle period
            print(f"Unable to stop idle GPU VM: {exc}")

    def arm_idle_timer() -> None:
        nonlocal idle_timer
        if idle_seconds <= 0:
            return
        cancel_idle_timer()
        idle_timer = Timer(idle_seconds, stop_if_idle)
        idle_timer.daemon = True
        idle_timer.start()

    def consume(message: object) -> None:
        nonlocal active_messages
        with active_lock:
            cancel_idle_timer()
            active_messages += 1
        try:
            payload = json.loads(message.data.decode("utf-8"))  # type: ignore[attr-defined]
            job_id = str(payload["job_id"])
            worker.process(job_id)
        except Exception:
            message.nack()  # type: ignore[attr-defined]
        else:
            message.ack()  # type: ignore[attr-defined]
        finally:
            with active_lock:
                active_messages -= 1
                if active_messages == 0:
                    arm_idle_timer()

    stream = subscriber.subscribe(settings.gcp_pubsub_subscription, callback=consume)
    # A manually started VM must not remain billable forever when there are no
    # messages to trigger the callback above.
    with active_lock:
        arm_idle_timer()
    signal.signal(signal.SIGTERM, lambda *_: (stream.cancel(), stop.set()))
    signal.signal(signal.SIGINT, lambda *_: (stream.cancel(), stop.set()))
    try:
        stream.result()
    finally:
        stream.cancel()
        stop.set()
        with active_lock:
            cancel_idle_timer()


if __name__ == "__main__":
    main()
