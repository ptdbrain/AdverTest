import importlib

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.adapters import get_adapter
from src.adapters.base import ModelAdapter
from src.attacks.base import AttackContext
from src.config import get_settings
from src.core.types import Sample
from src.datasets import get_dataset

#: Fixed so every test compares against the same pixels.
TEST_SEED = 4242


@pytest_asyncio.fixture
async def client(tmp_path, monkeypatch):
    """Async HTTP client for testing API endpoints."""
    test_root = tmp_path / "api-state"
    artifact_root = test_root / "artifacts"
    runs_root = test_root / "runs"
    temp_root = test_root / "tmp"
    static_root = test_root / "data"
    checkpoint_root = test_root / "checkpoints"
    database_path = test_root / "app.db"

    for path in (artifact_root, runs_root, temp_root, static_root, checkpoint_root):
        path.mkdir(parents=True, exist_ok=True)

    surrogates_dir = checkpoint_root / "surrogates"
    surrogates_dir.mkdir(parents=True, exist_ok=True)
    (surrogates_dir / "yolo11s.pt").write_bytes(b"mock weights")

    monkeypatch.chdir(test_root)
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{database_path.as_posix()}")
    monkeypatch.setenv("ARTIFACT_ROOT", str(artifact_root))
    monkeypatch.setenv("DATA_ROOT", str(static_root))
    monkeypatch.setenv("CHECKPOINT_ROOT", str(checkpoint_root))
    monkeypatch.setenv("RUNS_ROOT", str(runs_root))
    monkeypatch.setenv("TEMP", str(temp_root))
    monkeypatch.setenv("TMP", str(temp_root))
    monkeypatch.setenv("TMPDIR", str(temp_root))
    monkeypatch.setenv("CORS_ORIGINS", "http://test,http://127.0.0.1:3000")
    get_settings.cache_clear()

    import src.api.dependencies as deps_module
    import src.api.platform_dependencies as platform_deps_module
    import src.api.routers.admin as admin_module
    import src.api.routers.advisor as advisor_module
    import src.api.routers.analytics as analytics_module
    import src.api.routers.artifacts as artifacts_module
    import src.api.routers.auth as auth_module
    import src.api.routers.catalog as catalog_module
    import src.api.routers.datasets as datasets_module
    import src.api.routers.defence as defence_module
    import src.api.routers.projects as projects_module
    import src.api.routers.risk_rubric as risk_rubric_module
    import src.api.routers.runs as runs_module
    import src.api.routers.sessions as sessions_module
    import src.api.routers.settings as settings_module
    import src.api.routes as routes_module
    import src.auth.dependencies as auth_deps_module
    import src.main as main_module

    deps_module.get_store.cache_clear()
    deps_module.get_worker.cache_clear()
    if hasattr(deps_module, "get_workflow_store"):
        deps_module.get_workflow_store.cache_clear()
    if hasattr(deps_module, "get_generated_datasets"):
        deps_module.get_generated_datasets.cache_clear()
    if hasattr(deps_module, "get_training_jobs"):
        deps_module.get_training_jobs.cache_clear()
    if hasattr(deps_module, "get_checkpoint_validations"):
        deps_module.get_checkpoint_validations.cache_clear()
    if hasattr(deps_module, "get_advisor_service") and hasattr(deps_module.get_advisor_service, "cache_clear"):
        deps_module.get_advisor_service.cache_clear()
    if hasattr(auth_deps_module, "get_auth_service") and hasattr(auth_deps_module.get_auth_service, "cache_clear"):
        auth_deps_module.get_auth_service.cache_clear()
    if hasattr(platform_deps_module, "get_platform_database"):
        platform_deps_module.get_platform_database.cache_clear()
    if hasattr(platform_deps_module, "get_platform_storage"):
        platform_deps_module.get_platform_storage.cache_clear()
    if hasattr(platform_deps_module, "get_platform_artifacts"):
        platform_deps_module.get_platform_artifacts.cache_clear()

    importlib.reload(deps_module)
    importlib.reload(auth_deps_module)
    importlib.reload(platform_deps_module)
    importlib.reload(projects_module)
    importlib.reload(artifacts_module)
    importlib.reload(sessions_module)
    importlib.reload(settings_module)
    importlib.reload(routes_module)
    importlib.reload(catalog_module)
    importlib.reload(runs_module)
    importlib.reload(datasets_module)
    importlib.reload(defence_module)
    importlib.reload(analytics_module)
    importlib.reload(advisor_module)
    importlib.reload(auth_module)
    importlib.reload(admin_module)
    importlib.reload(risk_rubric_module)
    main_module = importlib.reload(main_module)

    app = main_module.app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        registration = await ac.post(
            "/api/v1/auth/register",
            json={
                "email": f"test-{tmp_path.name}@example.com",
                "password": "StrongPassword123!",
                "display_name": "Test Project Owner",
            },
        )
        assert registration.status_code == 201, registration.text
        auth_headers = {"Authorization": f"Bearer {registration.json()['access_token']}"}
        project = await ac.post(
            "/api/v1/projects",
            headers=auth_headers,
            json={"name": "Test project"},
        )
        assert project.status_code == 201, project.text
        ac.headers.update(auth_headers)
        default_project_id = project.json()["id"]
        setattr(ac, "default_project_id", default_project_id)

        async def add_project_scope(request):
            if request.url.path.startswith("/api/v1/") and "project_id" not in request.url.params:
                request.url = request.url.copy_merge_params({"project_id": default_project_id})

        ac.event_hooks["request"].append(add_project_scope)
        yield ac
    get_settings.cache_clear()


@pytest.fixture
def adapter() -> ModelAdapter:
    """Reference detector: no weights, no GPU, gradients available."""
    return get_adapter("blob_detector")


@pytest.fixture
def samples() -> list[Sample]:
    """Small deterministic batch from the reference dataset."""
    return get_dataset("synthetic_shapes", n_samples=4, seed=TEST_SEED).load()


@pytest.fixture
def sample(samples: list[Sample]) -> Sample:
    return samples[0]


@pytest.fixture
def context(adapter: ModelAdapter) -> AttackContext:
    """Attack context with a seeded generator and the reference model."""
    return AttackContext(rng=np.random.default_rng(TEST_SEED), model=adapter)
