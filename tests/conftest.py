import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.adapters import get_adapter
from src.adapters.base import ModelAdapter
from src.attacks.base import AttackContext
from src.core.types import Sample
from src.datasets import get_dataset
from src.main import app

#: Fixed so every test compares against the same pixels.
TEST_SEED = 4242


@pytest_asyncio.fixture
async def client():
    """Async HTTP client for testing API endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


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
