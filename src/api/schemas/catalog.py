from typing import Any

from pydantic import BaseModel, Field


class AttackCatalogItem(BaseModel):
    """One entry of the attack catalog (mirrors ``BaseAttack.describe``)."""

    name: str
    version: str = "1.0.0"
    group: str
    title: str = ""
    modality: str
    cost_class: str
    severity_levels: int
    needs_model: bool
    needs_gradients: bool
    required_annotations: list[str] = Field(default_factory=list)
    required_capabilities: list[str] = Field(default_factory=list)
    required_tasks: list[str] = Field(default_factory=list)
    required_sensors: list[str] = Field(default_factory=list)
    affected_sensors: list[str] = Field(default_factory=list)
    generation_mode: str = "per_sample"
    category: str = "adversarial"
    owner: str
    reference: str = ""
    params_schema: dict[str, Any] = Field(default_factory=dict)
    catalog_version: str = "1.0.0"
    implementation_version: str = "1.0.0"
    display_name: str = ""
    plain_summary: str = ""
    technical_summary: str = ""
    scenario: str = ""
    rationale: str = ""
    failure_symptoms: list[str] = Field(default_factory=list)
    severity_map: dict[int, str] = Field(default_factory=dict)
    compatibility: dict[str, Any] = Field(default_factory=dict)
    runtime_class: str = "instant"
    defense_hint: str = ""
    deterministic: bool = True
    supports_online: bool = True
    supports_offline: bool = True
    production_status: str = "production"
    threat_model: str = "model_agnostic"
    attack_type: str = "corruption"
    scenario_kind: str = "environmental_degradation"
    task_ids: list[str] = Field(default_factory=list)
    available: bool = True
    reason: str | None = None


class ModelCatalogItem(BaseModel):
    """One entry of the model-adapter catalog."""

    name: str
    task: str
    version: str
    modality: str
    supports_gradients: bool
    capabilities: list[str] = Field(default_factory=list)
    runnable: bool = True
    owner: str
    docstring: str = ""
    is_local: bool = True


class DatasetCatalogItem(BaseModel):
    """One entry of the dataset catalog; ``anonymized`` gates test runs."""

    name: str
    title: str = ""
    anonymized: bool
    modality: str
    owner: str
    params_schema: dict[str, Any] = Field(default_factory=dict)
    task_id: str
    input_schema: list[str] = Field(default_factory=list)
    annotation_schema: list[str] = Field(default_factory=list)
    class_map: dict[str, str] = Field(default_factory=dict)
    split_manifest: str | None = None
    ground_truth_status: str = "available"
    sample_count: int | None = Field(default=None, ge=0)
    dataset_params: dict[str, Any] = Field(default_factory=dict)
    demo: bool = False
