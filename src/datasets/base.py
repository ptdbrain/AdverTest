"""Dataset contract, including the mandatory anonymisation gate (plan §6).

A dataset that is not anonymised cannot enter a test run: the runner refuses it
with :class:`AnonymizationRequiredError` and there is no bypass flag. Loaders for
real data (KITTI, nuScenes, BDD100K) therefore have to either point at an
already-anonymised export or run the anonymiser first and set
``anonymized = True`` only afterwards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from src.core.types import CLASSES, Modality, Sample


class DatasetParams(BaseModel):
    """Base class for per-dataset parameter models."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AnonymizationRequiredError(RuntimeError):
    """Dataset has no anonymisation manifest, so it may not be evaluated."""


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    name: str
    anonymized: bool
    modality: Modality = "image"
    classes: tuple[str, ...] = CLASSES
    note: str = ""


class DatasetSource(ABC):
    """One dataset (or one split of it)."""

    name: ClassVar[str]
    #: Faces and plates blurred, with a manifest. Never hardcode True for real data.
    anonymized: ClassVar[bool] = False
    modality: ClassVar[Modality] = "image"
    owner: ClassVar[str] = "unassigned"
    params_model: ClassVar[type[DatasetParams]] = DatasetParams

    def __init__(self, **params: Any) -> None:
        self.params = self.params_model(**params)

    @abstractmethod
    def load(self, limit: int | None = None) -> list[Sample]:
        """Return samples; ``limit`` keeps dev runs cheap (plan §5 tier-1 scan)."""

    def info(self) -> DatasetInfo:
        return DatasetInfo(name=self.name, anonymized=self.anonymized, modality=self.modality)

    def require_anonymized(self) -> None:
        """Gate called by the runner before any inference happens."""
        if not self.anonymized:
            raise AnonymizationRequiredError(
                f"dataset {self.name!r} has no anonymisation manifest; "
                "run the anonymiser before creating a test run (plan §6)"
            )

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {
            "name": cls.name,
            "title": (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else "",
            "anonymized": cls.anonymized,
            "modality": cls.modality,
            "owner": cls.owner,
            "params_schema": cls.params_model.model_json_schema(),
        }
