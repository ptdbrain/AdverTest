"""Dataset registry (plan §1.1).

``synthetic_shapes`` is the reference source used by tests and demos. Real
loaders (KITTI 2D, nuScenes mini/trainval, BDD100K) are open slots: add one file
per dataset, decorate with ``@DATASETS.register``, and mind the anonymisation
gate in :mod:`src.datasets.base`.
"""

from __future__ import annotations

from src.core.registry import Registry, discover
from src.datasets.base import (
    AnonymizationRequiredError,
    DatasetInfo,
    DatasetParams,
    DatasetSource,
)

DATASETS: Registry[DatasetSource] = Registry("dataset")

_loaded = False


def load_datasets() -> Registry[DatasetSource]:
    """Import every dataset module once, then return the populated registry."""
    global _loaded
    if not _loaded:
        _loaded = True
        discover(__name__)
    return DATASETS


def get_dataset(name: str, **params: object) -> DatasetSource:
    """Instantiate a dataset by registered name."""
    return load_datasets().get(name)(**params)  # type: ignore[arg-type]


__all__ = [
    "DATASETS",
    "AnonymizationRequiredError",
    "DatasetInfo",
    "DatasetParams",
    "DatasetSource",
    "get_dataset",
    "load_datasets",
]
