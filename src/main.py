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
    analytics,
    artifacts,
    catalog,
    checkpoints,
    datasets,
    exports,
    jobs,
    platform_datasets,
    runs,
)
from src.api.routes import router
from src.attacks import load_attacks
from src.config import get_settings
from src.core.registry import UnknownPluginError
from src.datasets import load_datasets
from src.datasets.base import AnonymizationRequiredError
from src.demo_bootstrap import ensure_demo_checkpoint, ensure_demo_kitti

SIMULATION_BANNER = "SIMULATION ONLY — chưa validate, không dùng để quyết định triển khai"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every plugin once at start-up so the catalog is ready to serve."""
    settings = get_settings()
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
    if settings.bootstrap_demo_kitti:
        kitti_root = ensure_demo_kitti(
            enabled=True,
            storage=storage or get_platform_storage(),
            storage_prefix=settings.demo_kitti_storage_prefix,
            data_root=settings.data_root,
        )
        if kitti_root is not None:
            os.environ["ADVERTEST_KITTI_ROOT"] = str(kitti_root)
            print(f"Demo KITTI ready: {kitti_root}")
    attacks, models, datasets = load_attacks(), load_adapters(), load_datasets()
    print(
        f"Starting {settings.app_name} in {settings.app_env} mode — "
        f"{len(attacks)} attacks, {len(models)} adapters, {len(datasets)} datasets"
    )
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
data_root.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(catalog.router, prefix="/api/v1")
app.include_router(runs.router, prefix="/api/v1")
app.include_router(datasets.router, prefix="/api/v1")
app.include_router(artifacts.router, prefix="/api/v1")
app.include_router(checkpoints.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(exports.router, prefix="/api/v1")
app.include_router(platform_datasets.router, prefix="/api/v1")
# The legacy router below already owns every defence endpoint.  Mounting both
# routers registers identical paths twice, which makes route resolution depend
# on registration order and produces duplicate OpenAPI operation IDs.
app.include_router(analytics.router, prefix="/api/v1")
app.include_router(router, prefix="/api/v1")
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
