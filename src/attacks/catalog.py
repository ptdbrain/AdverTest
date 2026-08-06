"""Central, versioned narrative and compatibility catalog for attack plugins."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from src.core.objectives import RequiredAnnotation, SurrogateCapability
from src.core.types import AttackGroup, CostClass, Modality, SensorKind, Task

ProductionStatus = Literal["production", "experimental", "disabled"]
RuntimeClass = Literal["instant", "short", "long"]


class AttackCatalogError(RuntimeError):
    """The executable registry and the curated catalog disagree."""


class _FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class CatalogEntry(_FrozenContract):
    expected_version: str
    group: AttackGroup
    severity_levels: int
    display_name: str
    plain_summary: str
    technical_summary: str
    scenario: str
    rationale: str
    failure_symptoms: tuple[str, ...]
    defense_hint: str
    reference: str
    deterministic: bool = True
    supports_online: bool = True
    supports_offline: bool = True
    production_status: ProductionStatus = "production"
    artifact_requirements: tuple[str, ...] = ()


class AttackCompatibility(_FrozenContract):
    tasks: tuple[Task, ...] = ()
    modality: Modality
    required_annotations: tuple[RequiredAnnotation, ...] = ()
    model_capabilities: tuple[SurrogateCapability, ...] = ()
    required_sensors: tuple[SensorKind, ...] = ()
    artifact_requirements: tuple[str, ...] = ()


class AttackMetadata(_FrozenContract):
    catalog_version: str
    name: str
    implementation_version: str
    group: AttackGroup
    display_name: str
    plain_summary: str
    technical_summary: str
    scenario: str
    rationale: str
    failure_symptoms: tuple[str, ...]
    severity_map: dict[int, str]
    compatibility: AttackCompatibility
    cost_class: CostClass
    runtime_class: RuntimeClass
    defense_hint: str
    reference: str
    deterministic: bool
    supports_online: bool
    supports_offline: bool
    production_status: ProductionStatus


class AttackExclusion(_FrozenContract):
    name: str
    reasons: tuple[str, ...]


class AttackCatalogResult(_FrozenContract):
    selected: tuple[AttackMetadata, ...]
    exclusions: tuple[AttackExclusion, ...]


_GROUP_TEXT: dict[AttackGroup, dict[str, Any]] = {
    "A": {
        "plain": "Applies a common image corruption to measure everyday visual robustness.",
        "technical": "Uses a seeded corruption transform with an ordered, versioned severity scale.",
        "scenario": "Camera imagery degraded by capture, compression, optics, or ambient conditions.",
        "rationale": "Common corruptions reveal brittle visual features without requiring model access.",
        "symptoms": ("confidence drop", "missed detections", "localization drift"),
        "defense": "Use corruption-aware augmentation and monitor clean-versus-corrupted calibration.",
    },
    "B": {
        "plain": "Simulates physically motivated weather effects on camera or LiDAR sensors.",
        "technical": "Perturbs sensor measurements using seeded depth- or range-aware weather models.",
        "scenario": "Operation in fog, rain, or snow with reduced visibility and sensor returns.",
        "rationale": "Weather robustness must cover sensor physics beyond generic pixel noise.",
        "symptoms": ("range attenuation", "visibility loss", "weather-induced false positives"),
        "defense": "Train on calibrated weather variants and fuse complementary sensors where available.",
    },
    "C": {
        "plain": "Simulates occlusion, dropped data, frozen frames, or partial sensor failure.",
        "technical": "Removes or replaces declared sensor regions while preserving immutable ground truth.",
        "scenario": "Objects or sensor channels become temporarily unavailable in deployment.",
        "rationale": "Graceful degradation under missing evidence is a core safety requirement.",
        "symptoms": ("object disappearance", "stale predictions", "partial-scene blind spots"),
        "defense": "Use temporal checks, sensor-health monitoring, and occlusion-aware training.",
    },
    "D": {
        "plain": "Generates a white-box adversarial perturbation against model behavior.",
        "technical": "Optimizes a versioned attack objective through declared surrogate capabilities.",
        "scenario": "An attacker can differentiate through or closely approximate the deployed model.",
        "rationale": "Gradient attacks expose worst-case local sensitivity hidden by average-case testing.",
        "symptoms": ("targeted label change", "object vanishing", "adversarial mask failure"),
        "defense": "Use objective-matched adversarial training and independently validate held-out attacks.",
    },
    "E": {
        "plain": "Places a physical-plausible adversarial patch near an annotated object.",
        "technical": "Composites a versioned patch artifact with deterministic geometry and severity.",
        "scenario": "Printed or displayed patterns enter the camera field of view.",
        "rationale": "Localized physical attacks test robustness beyond imperceptible perturbations.",
        "symptoms": ("localized suppression", "patch-triggered false positives", "label confusion"),
        "defense": "Use patch augmentation, region consistency checks, and artifact provenance controls.",
    },
    "F": {
        "plain": "Tests black-box or transfer-style perturbations without model gradients.",
        "technical": "Uses seeded random or query-based search under an explicit inference budget.",
        "scenario": "An attacker can query predictions or transfer perturbations from another model.",
        "rationale": "Gradient secrecy does not eliminate query-based or transfer attacks.",
        "symptoms": ("query-driven confidence loss", "transfer failure", "budget-sensitive evasion"),
        "defense": "Limit query exposure, detect abnormal query patterns, and train on transfer examples.",
    },
}

_GROUPS: dict[AttackGroup, tuple[str, ...]] = {
    "A": (
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "saturate",
        "shot_noise",
        "snow",
        "spatter",
        "speckle_noise",
        "zoom_blur",
    ),
    "B": ("depth_fog", "depth_rain", "depth_snow", "lidar_fog", "lidar_snow"),
    "C": (
        "camera_dropout",
        "frame_freeze",
        "lidar_beam_drop",
        "lidar_sector_drop",
        "object_occlusion",
        "random_erasing",
        "sensor_fault",
    ),
    "D": ("cw_l2", "dag", "fgsm", "mi_fgsm", "pgd", "sam2_pgd", "tog"),
    "E": ("dpatch", "thys_patch"),
    "F": ("random_noise_linf", "square_attack"),
}

_VERSION_OVERRIDES = {"cw_l2": "2.0.0", "dpatch": "2.0.0", "thys_patch": "2.0.0"}
_SEVERITY_OVERRIDES = {"dpatch": 4, "thys_patch": 4, "square_attack": 3}
_OFFLINE_ONLY = {"cw_l2", "dag", "dpatch", "lidar_snow", "square_attack", "thys_patch"}
_ARTIFACTS = {"dpatch": ("patch_artifact",), "thys_patch": ("patch_artifact",)}


def _entry(name: str, group: AttackGroup) -> CatalogEntry:
    text = _GROUP_TEXT[group]
    display_name = name.replace("_", " ").title()
    return CatalogEntry(
        expected_version=_VERSION_OVERRIDES.get(name, "1.0.0"),
        group=group,
        severity_levels=_SEVERITY_OVERRIDES.get(name, 5),
        display_name=display_name,
        plain_summary=f"{display_name}: {text['plain']}",
        technical_summary=f"{display_name}: {text['technical']}",
        scenario=text["scenario"],
        rationale=text["rationale"],
        failure_symptoms=text["symptoms"],
        defense_hint=text["defense"],
        reference="AdverTest Person D platform specification, attack catalog v1",
        supports_online=name not in _OFFLINE_ONLY,
        artifact_requirements=_ARTIFACTS.get(name, ()),
    )


CATALOG_ENTRIES: dict[str, CatalogEntry] = {
    name: _entry(name, group)
    for group, names in _GROUPS.items()
    for name in names
}


class AttackCatalog:
    version = "1.0.0"

    def __init__(self, entries: dict[str, CatalogEntry]) -> None:
        self._entries = dict(entries)
        self._attack_classes: dict[str, type[Any]] = {}

    def validate_registry(self, attack_classes: list[type[Any]]) -> None:
        for attack_cls in attack_classes:
            entry = self._entries.get(attack_cls.name)
            if entry is None:
                raise AttackCatalogError(
                    f"attack {attack_cls.name!r} has no centralized catalog entry"
                )
            if entry.expected_version != attack_cls.version:
                raise AttackCatalogError(
                    f"attack {attack_cls.name!r} version {attack_cls.version!r} "
                    f"does not match catalog version {entry.expected_version!r}"
                )
            if entry.group != attack_cls.group:
                raise AttackCatalogError(
                    f"attack {attack_cls.name!r} group does not match catalog"
                )
            if entry.severity_levels != attack_cls.severity_levels:
                raise AttackCatalogError(
                    f"attack {attack_cls.name!r} severity map does not match implementation"
                )

    def bind(self, attack_classes: list[type[Any]]) -> None:
        self.validate_registry(attack_classes)
        registered = {attack_cls.name for attack_cls in attack_classes}
        orphaned = sorted(set(self._entries) - registered)
        if orphaned:
            raise AttackCatalogError(
                f"catalog entries have no registered implementation: {', '.join(orphaned)}"
            )
        self._attack_classes = {
            attack_cls.name: attack_cls for attack_cls in attack_classes
        }

    def get(self, name: str) -> AttackMetadata:
        try:
            attack_cls = self._attack_classes[name]
        except KeyError:
            raise AttackCatalogError(f"unknown catalog attack: {name!r}") from None
        return metadata_for_attack(attack_cls)

    def list(
        self,
        *,
        task: Task,
        model_capabilities: frozenset[SurrogateCapability],
        annotation_types: frozenset[RequiredAnnotation],
        modality: Modality,
        online: bool,
        production_only: bool = True,
        available_artifacts: frozenset[str] | None = None,
    ) -> AttackCatalogResult:
        if not self._attack_classes:
            raise AttackCatalogError("attack catalog is not bound to the executable registry")
        selected: list[AttackMetadata] = []
        exclusions: list[AttackExclusion] = []
        for name, attack_cls in sorted(self._attack_classes.items()):
            metadata = metadata_for_attack(attack_cls)
            reasons = _exclusion_reasons(
                metadata,
                task=task,
                model_capabilities=model_capabilities,
                annotation_types=annotation_types,
                modality=modality,
                online=online,
                production_only=production_only,
                available_artifacts=available_artifacts,
            )
            if reasons:
                exclusions.append(AttackExclusion(name=name, reasons=tuple(reasons)))
            else:
                selected.append(metadata)
        return AttackCatalogResult(
            selected=tuple(selected),
            exclusions=tuple(exclusions),
        )


def metadata_for_attack(attack_cls: type[Any]) -> AttackMetadata:
    entry = CATALOG_ENTRIES.get(attack_cls.name)
    if entry is None:
        raise AttackCatalogError(
            f"attack {attack_cls.name!r} has no centralized catalog entry"
        )
    if entry.expected_version != attack_cls.version:
        raise AttackCatalogError(
            f"attack {attack_cls.name!r} implementation/catalog version mismatch"
        )
    runtime_class: RuntimeClass = (
        "instant"
        if attack_cls.cost_class == "cheap"
        else "short"
        if attack_cls.cost_class == "medium"
        else "long"
    )
    labels = ("no-op", "very low", "low", "medium", "high", "critical")
    return AttackMetadata(
        catalog_version=AttackCatalog.version,
        name=attack_cls.name,
        implementation_version=attack_cls.version,
        group=attack_cls.group,
        display_name=entry.display_name,
        plain_summary=entry.plain_summary,
        technical_summary=entry.technical_summary,
        scenario=entry.scenario,
        rationale=entry.rationale,
        failure_symptoms=entry.failure_symptoms,
        severity_map={
            severity: labels[min(severity, len(labels) - 1)]
            for severity in range(attack_cls.severity_levels + 1)
        },
        compatibility=AttackCompatibility(
            tasks=tuple(sorted(attack_cls.required_tasks)),
            modality=attack_cls.modality,
            required_annotations=tuple(sorted(attack_cls.required_annotations)),
            model_capabilities=tuple(sorted(attack_cls.required_capabilities)),
            required_sensors=tuple(sorted(attack_cls.required_sensors)),
            artifact_requirements=entry.artifact_requirements,
        ),
        cost_class=attack_cls.cost_class,
        runtime_class=runtime_class,
        defense_hint=entry.defense_hint,
        reference=attack_cls.reference or entry.reference,
        deterministic=entry.deterministic,
        supports_online=entry.supports_online,
        supports_offline=entry.supports_offline,
        production_status=entry.production_status,
    )


def _exclusion_reasons(
    metadata: AttackMetadata,
    *,
    task: Task,
    model_capabilities: frozenset[SurrogateCapability],
    annotation_types: frozenset[RequiredAnnotation],
    modality: Modality,
    online: bool,
    production_only: bool,
    available_artifacts: frozenset[str] | None,
) -> list[str]:
    reasons: list[str] = []
    compatibility = metadata.compatibility
    if compatibility.tasks and task not in compatibility.tasks:
        reasons.append(f"task_mismatch:{','.join(compatibility.tasks)}")
    if modality != "multi" and compatibility.modality not in {modality, "image"}:
        reasons.append(f"modality_mismatch:{compatibility.modality}")
    for annotation in compatibility.required_annotations:
        if annotation not in annotation_types:
            reasons.append(f"missing_annotation:{annotation}")
    for capability in compatibility.model_capabilities:
        if capability not in model_capabilities:
            reasons.append(f"missing_model_capability:{capability}")
    if online and not metadata.supports_online:
        reasons.append("online_not_supported")
    if not online and not metadata.supports_offline:
        reasons.append("offline_not_supported")
    if production_only and metadata.production_status != "production":
        reasons.append(f"production_status:{metadata.production_status}")
    if available_artifacts is not None:
        for artifact in compatibility.artifact_requirements:
            if artifact not in available_artifacts:
                reasons.append(f"missing_artifact:{artifact}")
    return reasons
