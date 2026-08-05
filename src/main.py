"""AdverTest application entry point.

SIMULATION ONLY: this service evaluates perception models against generated
corruptions and attacks. It has no path to a deployment pipeline, and a low
RobustScore never blocks or approves anything automatically — a human Reviewer
decides (plan §7).
"""

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from src.adapters import load_adapters
from src.api.routes import router
from src.attacks import load_attacks
from src.config import get_settings
from src.core.registry import UnknownPluginError
from src.datasets import load_datasets
from src.datasets.base import AnonymizationRequiredError

SIMULATION_BANNER = "SIMULATION ONLY — chưa validate, không dùng để quyết định triển khai"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every plugin once at start-up so the catalog is ready to serve."""
    settings = get_settings()
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.mount("/data", StaticFiles(directory="data"), name="data")


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
    return {"status": "ok", "env": settings.app_env, "simulation_only": True, "banner": SIMULATION_BANNER}
