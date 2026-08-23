from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # App
    app_name: str = "AdverTest"
    app_env: Literal["development", "production", "test"] = "development"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_host: str = "0.0.0.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Test-run defaults (plan §5: cheap by default, opt into expensive scans)
    default_model: str = "blob_detector"
    default_dataset: str = "synthetic_shapes"
    default_sample_limit: int = Field(default=8, ge=1)
    default_severities: str = "1,3,5"
    run_seed: int = 20260730
    #: Calibration for the pre-run cost estimate: seconds per unit of attack cost.
    seconds_per_cost_unit: float = Field(default=0.05, gt=0.0)

    # Human-in-the-loop gate (plan §7): degradation above this needs a Reviewer.
    review_degradation_threshold: float = Field(default=0.30, ge=0.0, le=1.0)

    # Storage (in-memory today; PostgreSQL + MinIO per plan §4)
    data_root: str = str(PROJECT_ROOT / "data")
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'app.db').as_posix()}"
    worker_max_concurrency: int = Field(default=1, ge=1, le=8)
    runs_root: str = str(PROJECT_ROOT / "runs")
    artifact_root: str = str(PROJECT_ROOT / "data" / "artifacts")
    checkpoint_root: str = str(PROJECT_ROOT / "data" / "checkpoints")

    # Platform persistence, object storage, and worker dispatch.  The legacy
    # SQLite stores continue to support existing local product routes while
    # these settings drive the project-scoped production platform.
    platform_database_url: str | None = None
    object_storage_backend: Literal["local", "s3"] = "local"
    object_storage_bucket: str = "advertest-artifacts"
    object_storage_endpoint_url: str | None = None
    object_storage_region: str = "auto"
    object_storage_access_key_id: str | None = None
    object_storage_secret_access_key: str | None = None
    object_storage_signed_url_ttl_seconds: int = Field(default=900, ge=60, le=86_400)
    queue_backend: Literal["local", "redis", "http_dispatcher"] = "local"
    redis_url: str | None = None
    external_queue_dispatch_url: str | None = None
    external_queue_dispatch_token: str | None = None
    gcp_pubsub_subscription: str | None = None
    platform_worker_poll_seconds: float = Field(default=1.0, gt=0.0, le=60.0)
    checkpoint_validation_timeout_seconds: int = Field(default=120, ge=5, le=3600)
    checkpoint_validation_memory_mb: int = Field(default=2048, ge=128, le=65_536)
    checkpoint_validation_cpu_seconds: int = Field(default=90, ge=1, le=3600)
    checkpoint_sandbox_url: str | None = None
    checkpoint_sandbox_token: str | None = None
    external_gpu_worker_url: str | None = None
    external_gpu_worker_token: str | None = None
    # Benchmark jobs normally use the lightweight local worker. Production can
    # opt into the durable Postgres + dispatcher + GCE execution plane.
    run_execution_backend: Literal["local", "platform"] = "local"
    platform_default_project_id: str = "advertest-demo"
    platform_default_user_id: str = "demo-user"
    # Demo-only bootstrap: downloads a fixed, vendor-maintained checkpoint into
    # the disposable runtime filesystem. User uploads never use this path.
    bootstrap_demo_model: bool = False
    bootstrap_demo_model_id: Literal["yolo11n"] = "yolo11n"
    demo_model_storage_key: str = "catalog/models/yolo11n/v1/yolo11n.pt"
    bootstrap_demo_kitti: bool = False
    demo_kitti_storage_prefix: str = "catalog/datasets/kitti/v1/anonymized/kitti-de/"

    # Execution hardware defaults
    model_device: str = "cpu"
    model_half_precision: bool = False
    model_batch_size: int = Field(default=1, ge=1)

    @property
    def severity_list(self) -> list[int]:
        """``default_severities`` parsed into integers."""
        return [int(part) for part in self.default_severities.split(",") if part.strip()]

    @property
    def resolved_platform_database_url(self) -> str:
        return self.platform_database_url or self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
