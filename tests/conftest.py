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

    import src.api.routes as routes_module
    import src.main as main_module

    importlib.reload(routes_module)
    main_module = importlib.reload(main_module)

    app = main_module.app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
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
