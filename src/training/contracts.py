"""Owner-neutral defense and training configuration contracts."""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SamplingStrategy = Literal[
    "random",
    "class_balanced",
    "object_size_balanced",
    "failure_cluster_targeted",
    "severity_distribution",
]


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DefenseProfile(_FrozenContract):
    profile_id: str
    version: str = "1.0.0"
    recipe_ids: tuple[str, ...] = ()
    clean_replay_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    generated_ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    hard_example_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    sampling_strategy: SamplingStrategy = "random"
    max_variants_per_source: int = Field(default=5, ge=1)
    max_variants_per_recipe: int = Field(default=10_000, ge=1)
    online_generation: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ratios(self) -> Self:
        if self.clean_replay_ratio + self.generated_ratio <= 0.0:
            raise ValueError("defense profile requires clean or generated samples")
        return self


class TrainingRunConfig(_FrozenContract):
    run_id: str
    trainer_name: str
    model_version: str
    dataset_version_id: str
    split_manifest_id: str
    defense_profile_id: str
    seed: int = Field(ge=0)
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    max_gpu_hours: float | None = Field(default=None, gt=0.0)
    max_storage_bytes: int | None = Field(default=None, gt=0)
    max_wall_time_seconds: int | None = Field(default=None, gt=0)
    contract_version: Literal["1.0.0"] = "1.0.0"
    metadata: dict[str, Any] = Field(default_factory=dict)
