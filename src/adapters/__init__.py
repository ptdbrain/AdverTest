"""Model adapters (M1–M6 of plan §1.2).

Drop a new adapter file in this package and decorate the class with
``@MODELS.register`` — auto-discovery picks it up, no core file changes.
"""

from __future__ import annotations

from src.adapters.base import GradientsNotSupportedError, ModelAdapter
from src.core.registry import Registry, discover

MODELS: Registry[ModelAdapter] = Registry("model adapter")

_loaded = False


def load_adapters() -> Registry[ModelAdapter]:
    """Import every adapter module once, then return the populated registry."""
    global _loaded
    if not _loaded:
        _loaded = True
        discover(__name__)
    return MODELS


def get_adapter(name: str, **kwargs: object) -> ModelAdapter:
    """Instantiate an adapter by registered name."""
    return load_adapters().get(name)(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "MODELS",
    "GradientsNotSupportedError",
    "ModelAdapter",
    "get_adapter",
    "load_adapters",
]
