"""Person D service facade consumed by owner A, CLI, and compute workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.attacks import ATTACK_CATALOG, load_attacks
from src.attacks.catalog import AttackCatalog
from src.attacks.recipes import AttackRecipe, RandomNRequest, RecipeBuilder, RecipeValidation
from src.datasets.splits import SplitBuilder
from src.datasets.versioning import DatasetIngestor
from src.evaluation.model_comparison import checkpoint_gate, compare_models
from src.pipeline.generator import AttackDatasetGenerator
from src.training.dataset_builder import TrainingDatasetBuilder
from src.training.registry import TrainerRegistry
from src.training.worker import ComputeWorker

_ALL_CAPABILITIES = frozenset(
    {
        "input_gradient",
        "detection_loss",
        "objectness",
        "class_logits",
        "dense_proposals",
        "segmentation_loss",
        "class_margin",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetService:
    ingestor_type: type[DatasetIngestor] = DatasetIngestor
    splits: SplitBuilder = SplitBuilder()


@dataclass(frozen=True, slots=True)
class RecipeService:
    catalog: AttackCatalog
    builder: RecipeBuilder

    def validate(self, recipe: AttackRecipe, **context: Any) -> RecipeValidation:
        defaults = {
            "task": "detection2d",
            "model_capabilities": _ALL_CAPABILITIES,
            "annotation_types": frozenset({"boxes", "mask"}),
            "modality": "multi",
            "online": False,
        }
        return self.builder.validate(recipe, self.catalog, **{**defaults, **context})

    def sample(self, request: RandomNRequest) -> list[AttackRecipe]:
        return self.builder.random_n(request, self.catalog)


@dataclass(frozen=True, slots=True)
class GenerationService:
    generator: AttackDatasetGenerator


@dataclass(frozen=True, slots=True)
class TrainingDatasetService:
    builder: TrainingDatasetBuilder


@dataclass(frozen=True, slots=True)
class BenchmarkService:
    runner_type: type


@dataclass(frozen=True, slots=True)
class ComparisonService:
    compare: Any = compare_models
    gate: Any = checkpoint_gate


@dataclass(frozen=True, slots=True)
class TrainingComputeService:
    registry: TrainerRegistry

    def worker(self) -> ComputeWorker:
        return ComputeWorker(self.registry)


@dataclass(frozen=True, slots=True)
class PersonDServices:
    datasets: DatasetService
    recipes: RecipeService
    generation: GenerationService
    training_data: TrainingDatasetService
    benchmarks: BenchmarkService
    comparisons: ComparisonService
    training: TrainingComputeService

    @classmethod
    def default(cls) -> PersonDServices:
        from src.pipeline.generic_benchmark import BenchmarkRunner

        load_attacks()
        registry = TrainerRegistry()
        return cls(
            datasets=DatasetService(),
            recipes=RecipeService(ATTACK_CATALOG, RecipeBuilder()),
            generation=GenerationService(AttackDatasetGenerator()),
            training_data=TrainingDatasetService(TrainingDatasetBuilder()),
            benchmarks=BenchmarkService(BenchmarkRunner),
            comparisons=ComparisonService(),
            training=TrainingComputeService(registry),
        )
