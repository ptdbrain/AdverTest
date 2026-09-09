"""AdverTest application entry point.

SIMULATION ONLY: this service evaluates perception models against generated
corruptions and attacks. It has no path to a deployment pipeline, and a low
RobustScore never blocks or approves anything automatically — a human Reviewer
decides (plan §7).
"""

import os
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.adapters import load_adapters
from src.api.platform_dependencies import get_platform_storage
from src.api.routers import (
    admin,
    advisor,
    analytics,
    artifacts,
    auth,
    catalog,
    checkpoints,
    datasets,
    exports,
    jobs,
    live_inference,
    platform_datasets,
    projects,
    risk_rubric,
    runs,
    sessions,
    system,
    worker_callbacks,
)
from src.api.routers import settings as settings_router
from src.api.routes import router
from src.attacks import load_attacks
from src.config import get_settings
from src.core.registry import UnknownPluginError
from src.datasets import load_datasets
from src.datasets.base import AnonymizationRequiredError
from src.demo_bootstrap import (
    ensure_demo_catalog,
    ensure_demo_checkpoint,
    ensure_drive_export_catalog,
)
from src.demo_bootstrap import (
    ensure_demo_kitti as _ensure_demo_kitti,
)

# Compatibility symbol for deployments/tests that verify the retired legacy
# bootstrap is never called. The lifespan intentionally does not invoke it.
ensure_demo_kitti = _ensure_demo_kitti

SIMULATION_BANNER = "SIMULATION ONLY — chưa validate, không dùng để quyết định triển khai"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every plugin once at start-up so the catalog is ready to serve."""
    settings = get_settings()
    if settings.app_env == "production":
        settings.validate_production_environment()
    storage = None
    if settings.bootstrap_demo_model:
        storage = get_platform_storage()
        checkpoint = ensure_demo_checkpoint(
            enabled=True,
            checkpoint_root=settings.checkpoint_root,
            model_id=settings.bootstrap_demo_model_id,
            storage=storage,
            storage_key=settings.demo_model_storage_key,
        )
        print(f"Demo checkpoint ready: {checkpoint}")
    # Dataset hydration belongs to the execution worker. The old KITTI demo
    # prefix has been retired, so the API must never make its availability
    # dependent on optional legacy objects during startup.
    # The Render API is the control plane: it must become healthy before any
    # optional object downloads. Cloud Run workers hydrate benchmark bundles
    # when they claim a job. Keeping this opt-in also prevents Render's port
    # scanner from timing out while several catalog bundles download.
    if settings.bootstrap_drive_export_catalog:
        catalog_roots = ensure_drive_export_catalog(
            storage=storage or get_platform_storage(),
            data_root=settings.data_root,
            prefixes={
                "kitti2d-100": settings.drive_export_kitti2d_storage_prefix,
                "cityscapes-instance-100": settings.drive_export_cityscapes_storage_prefix,
                "nuscenes-mini-100": settings.drive_export_nuscenes_storage_prefix,
            },
        )
        print(f"Drive export catalog ready: {catalog_roots}")
    elif settings.bootstrap_demo_catalog:
        catalog_root = ensure_demo_catalog(
            enabled=True,
            storage=storage or get_platform_storage(),
            storage_prefix=settings.demo_catalog_storage_prefix,
            data_root=settings.data_root,
        )
        print(f"Demo catalog ready: {catalog_root}")
    attacks, models, datasets = load_attacks(), load_adapters(), load_datasets()
    from src.auth.dependencies import get_auth_service

    if (
        settings.app_env == "test"
        or getattr(settings, "allow_dev_bootstrap_accounts", False)
        or getattr(settings, "demo_bootstrap_accounts", False)
    ):
        get_auth_service().ensure_default_accounts()
    if getattr(settings, "demo_fake_sessions", False):
        from src.demo_fake_bootstrap import ensure_fake_demo_workspace

        prepared = ensure_fake_demo_workspace()
        print(f"Fake demo workspace ready: {len(prepared)} prepared sessions")
    print(
        f"Starting {settings.app_name} in {settings.app_env} mode — "
        f"{len(attacks)} attacks, {len(models)} adapters, {len(datasets)} datasets"
    )
    import gc

    from src.core.memory import trim_memory
    gc.collect()
    trim_memory()
    yield
    print("Shutting down...")


app = FastAPI(
    title="AdverTest",
    description="Adversarial generation & robustness testing for perception models (simulation only)",
    version="0.1.0",
    lifespan=lifespan,
)

settings = get_settings()
data_root = Path(settings.data_root).expanduser().resolve()
artifact_root = Path(settings.artifact_root).expanduser().resolve()
data_root.mkdir(parents=True, exist_ok=True)
artifact_root.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(catalog.router, prefix="/api/v1")
app.include_router(runs.router, prefix="/api/v1")
app.include_router(datasets.router, prefix="/api/v1")
app.include_router(projects.router, prefix="/api/v1")
app.include_router(artifacts.router, prefix="/api/v1")
app.include_router(checkpoints.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(exports.router, prefix="/api/v1")
app.include_router(platform_datasets.router, prefix="/api/v1")
app.include_router(worker_callbacks.router, prefix="/api/v1")
# The legacy router below already owns every defence endpoint.  Mounting both
# routers registers identical paths twice, which makes route resolution depend
# on registration order and produces duplicate OpenAPI operation IDs.
app.include_router(analytics.router, prefix="/api/v1")
app.include_router(advisor.router, prefix="/api/v1")
app.include_router(system.router, prefix="/api/v1")
app.include_router(live_inference.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")
app.include_router(settings_router.router, prefix="/api/v1")
app.include_router(risk_rubric.router, prefix="/api/v1")
app.include_router(router, prefix="/api/v1")
# Generated evidence may use a different persistent volume from reviewed
# catalog data. Mount its explicit namespace first so `/data` cannot shadow it.
app.mount("/data/artifacts", StaticFiles(directory=str(artifact_root)), name="artifacts")
app.mount("/data", StaticFiles(directory=str(data_root)), name="data")


@app.middleware("http")
async def add_simulation_banner(
    request: Request,
    call_next: Callable[[Request], Awaitable[JSONResponse]],
) -> JSONResponse:
    """Stamp every response, so no client can forget what these numbers are."""
    response = await call_next(request)
    response.headers["X-Simulation-Only"] = "true"
    return response


@app.exception_handler(UnknownPluginError)
async def handle_unknown_plugin(request: Request, exc: UnknownPluginError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(AnonymizationRequiredError)
async def handle_anonymization_gate(request: Request, exc: AnonymizationRequiredError) -> JSONResponse:
    """Plan §6: a dataset without an anonymisation manifest is a hard stop."""
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.exception_handler(ValidationError)
async def handle_plugin_params(request: Request, exc: ValidationError) -> JSONResponse:
    """Bad attack/dataset parameters reach us as pydantic errors."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/health")
async def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "env": settings.app_env,
        "simulation_only": True,
        "banner": SIMULATION_BANNER,
        "build_sha": os.getenv("BUILD_SHA", "unknown"),
        "execution_profile": "gpu" if settings.model_device.startswith("cuda") else "cpu",
    }
