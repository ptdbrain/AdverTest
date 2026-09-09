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

    # Security & Auth
    jwt_secret: str = "advertest-insecure-development-secret-key-2026"
    auth_cookie_name: str = "advertest_session"
    auth_cookie_secure: bool = False
    auth_cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    allow_dev_bootstrap_accounts: bool = False
    admin_default_password: str = "AdminPassword123!"
    google_client_id: str = ""
    google_client_secret: str | None = None
    wandb_encryption_key: str = "replace-this-with-a-fernet-key-in-production"

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
    object_storage_endpoint: str | None = None
    object_storage_region: str = "auto"
    object_storage_access_key_id: str | None = None
    object_storage_access_key: str | None = None
    object_storage_secret_access_key: str | None = None
    object_storage_secret_key: str | None = None
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
    # Cloud Run workers never connect to PostgreSQL directly.  They claim a
    # job and report progress through this authenticated API callback instead.
    worker_callback_api_url: str | None = None
    worker_callback_token: str | None = None
    # Benchmark jobs normally use the lightweight local worker. Production can
    # opt into the durable Postgres + dispatcher + GCE execution plane.
    run_execution_backend: Literal["local", "platform"] = "local"
    # Presentation metadata for the remote execution plane.  This must stay
    # separate from MODEL_DEVICE: Render hosts the API on CPU while Cloud Run
    # materialises and executes benchmark jobs on this on-demand GPU.
    remote_gpu_name: str = "NVIDIA L4"
    remote_gpu_vram_gb: float = Field(default=24.0, ge=0.0)
    platform_default_project_id: str = "advertest-demo"
    platform_default_user_id: str = "demo-user"
    # Demo-only bootstrap: downloads a fixed, vendor-maintained checkpoint into
    # the disposable runtime filesystem. User uploads never use this path.
    bootstrap_demo_model: bool = False
    bootstrap_demo_model_id: Literal["yolo11n"] = "yolo11n"
    demo_model_storage_key: str = "catalog/models/yolo11n/v1/yolo11n.pt"
    # Opt-in, credential-free 2D smoke profile for fresh laptops and preview
    # deploys. It exposes the code-owned CPU reference detector and tracked
    # KITTI bundle; it never substitutes for a reviewed model artifact.
    bootstrap_portable_demo: bool = False
    bootstrap_demo_kitti: bool = False
    demo_kitti_storage_prefix: str = "catalog/datasets/kitti/v1/anonymized/kitti-de/"
    bootstrap_demo_catalog: bool = False
    demo_catalog_storage_prefix: str = "catalog/datasets/demo-catalog/v1/"
    # Demo-only sign-in: seed a public engineer + admin account pair and expose
    # one-click login cards on the sign-in page. Keep disabled in production
    # unless this deployment is intentionally a public demo.
    demo_bootstrap_accounts: bool = False
    demo_engineer_password: str = "EngineerPassword123!"
    # Opt-in, static fake workspace for a fast product walkthrough. It never
    # loads a model or queues an inference job.
    demo_fake_sessions: bool = False
    demo_fake_engineer_email: str = "demo-engineer@advertest.ai"
    demo_fake_engineer_password: str = "DemoEngineer2026!"
    demo_fake_engineer_id: str = "usr-demo-engineer"
    bootstrap_cityscapes_catalog: bool = False
    cityscapes_catalog_storage_prefix: str = "catalog/datasets/cityscapes-200/v1/"
    bootstrap_kitti_catalog: bool = False
    kitti_catalog_storage_prefix: str = "catalog/datasets/kitti-200/v1/"
    bootstrap_kitti3d_catalog: bool = False
    kitti3d_catalog_storage_prefix: str = "catalog/datasets/kitti3d-200/v1/"
    # Reviewed replacement catalog imported from the 2026-08-31 Drive export.
    # Production materialises these fixed 100-sample bundles on demand.
    bootstrap_drive_export_catalog: bool = False
    drive_export_kitti2d_storage_prefix: str = "catalog/datasets/kitti2d-100/v1/"
    drive_export_cityscapes_storage_prefix: str = "catalog/datasets/cityscapes-instance-100/v1/"
    drive_export_nuscenes_storage_prefix: str = "catalog/datasets/nuscenes-mini-100/v1/"

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

    @property
    def resolved_object_storage_endpoint(self) -> str | None:
        return self.object_storage_endpoint_url or self.object_storage_endpoint

    @property
    def resolved_object_storage_access_key(self) -> str | None:
        return self.object_storage_access_key_id or self.object_storage_access_key

    @property
    def resolved_object_storage_secret_key(self) -> str | None:
        return self.object_storage_secret_access_key or self.object_storage_secret_key

    def validate_production_environment(self) -> None:
        """Validate that all security secrets and production settings meet production hardening standards."""
        if self.app_env != "production":
            return

        failed_keys: list[str] = []

        # 1. JWT Secret validation (must be secure, >= 32 chars, not contain development/insecure keywords)
        insecure_jwt_markers = ("insecure", "development", "secret", "default", "changeme", "advertest")
        if (
            not self.jwt_secret
            or len(self.jwt_secret.strip()) < 32
            or any(marker in self.jwt_secret.lower() for marker in insecure_jwt_markers)
        ):
            failed_keys.append("JWT_SECRET (must be >= 32 chars and not use default/insecure value)")

        # 2. Admin Default Password validation
        if not self.admin_default_password or self.admin_default_password in (
            "AdminPassword123!",
            "admin",
            "password",
            "12345678",
        ):
            failed_keys.append("ADMIN_DEFAULT_PASSWORD (must be changed from default)")

        # 3. Database URL validation (PostgreSQL required in production, SQLite forbidden)
        db_url = self.resolved_platform_database_url
        if db_url.startswith("sqlite:"):
            failed_keys.append("DATABASE_URL / PLATFORM_DATABASE_URL (SQLite is not permitted in production)")

        # 4. Storage configuration validation
        if self.object_storage_backend == "s3":
            if (
                not self.object_storage_access_key_id
                or self.object_storage_access_key_id in ("minioadmin", "admin")
                or not self.object_storage_secret_access_key
                or self.object_storage_secret_access_key in ("minioadminpassword", "password")
            ):
                failed_keys.append(
                    "OBJECT_STORAGE_ACCESS_KEY_ID / OBJECT_STORAGE_SECRET_ACCESS_KEY (must be securely configured)"
                )

        # 5. Queue backend validation
        if self.queue_backend == "redis" and (not self.redis_url or not self.redis_url.strip()):
            failed_keys.append("REDIS_URL (required when QUEUE_BACKEND=redis)")

        # 6. Google OAuth is optional. When configured, reject only the
        # placeholder value; password authentication remains a supported
        # production mode for deployments that have not enabled SSO.
        if self.google_client_id == "your-google-client-id.apps.googleusercontent.com":
            failed_keys.append("GOOGLE_CLIENT_ID (must not use the placeholder value)")

        if failed_keys:
            # Strictly do not log raw secrets or credentials in error messages
            raise ValueError(
                "CRITICAL PRODUCTION CONFIGURATION ERROR: The following settings failed security validation: "
                + ", ".join(failed_keys)
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
