"""Adversarial defence profiles and named attack mix presets.

Implements the multi-threat attack mix design (plan C4/C5/C6) so models
are not trained in isolation on single perturbations, but on principled,
leakage-safe distributions (R1 Robust Mix, R2 Targeted Repair).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from src.training.contracts import DefenseProfile, SamplingStrategy


@dataclass(frozen=True, slots=True)
class AttackMixRatio:
    """Configurable proportions for defense training data mixtures."""

    clean_ratio: float = 0.30
    weather_ratio: float = 0.15
    noise_ratio: float = 0.15
    occlusion_ratio: float = 0.15
    sensor_fault_ratio: float = 0.10
    adversarial_ratio: float = 0.10
    residual_failure_ratio: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {
            "clean_ratio": self.clean_ratio,
            "weather_ratio": self.weather_ratio,
            "noise_ratio": self.noise_ratio,
            "occlusion_ratio": self.occlusion_ratio,
            "sensor_fault_ratio": self.sensor_fault_ratio,
            "adversarial_ratio": self.adversarial_ratio,
            "residual_failure_ratio": self.residual_failure_ratio,
        }


#: Default starting mix from plan §43 (C5): 30% clean, 65% perturbations, 5% residual failures.
ROBUST_MIX_RATIO = AttackMixRatio()

#: Targeted repair mix: 50% clean replay, 30% specific failure repairs, 20% hard examples.
TARGETED_REPAIR_RATIO = AttackMixRatio(
    clean_ratio=0.50,
    weather_ratio=0.0,
    noise_ratio=0.0,
    occlusion_ratio=0.10,
    sensor_fault_ratio=0.0,
    adversarial_ratio=0.20,
    residual_failure_ratio=0.20,
)

#: Standard 3-way partition of attack groups to test seen vs unseen generalization (plan C6).
TRAIN_ATTACKS: tuple[str, ...] = (
    "gaussian_noise",
    "fog",
    "motion_blur",
    "object_occlusion",
    "patch_attack",
    "frame_freeze",
)

VALIDATION_ATTACKS: tuple[str, ...] = (
    "snow",
    "sensor_fault",
    "shot_noise",
    "zoom_blur",
)

HELDOUT_ATTACKS: tuple[str, ...] = (
    "pgd",
    "fgsm",
    "cw_l2",
    "depth_fog_prior",
    "adversarial_patch",
)


def create_robust_mix_profile(
    profile_id: str,
    recipe_ids: tuple[str, ...] | list[str] = (),
    *,
    sampling_strategy: SamplingStrategy = "severity_distribution",
    max_variants_per_source: int = 5,
    metadata: Mapping[str, Any] | None = None,
) -> DefenseProfile:
    """Build a standard R1 Robust Mix DefenseProfile (30% clean / 70% generated).

    Args:
        profile_id: Unique identifier for this defense profile.
        recipe_ids: Tuple or list of recipe IDs to include in the training mix.
        sampling_strategy: Sampling distribution strategy across variants.
        max_variants_per_source: Upper cap on variants generated per clean image.
        metadata: Optional extra metadata dictionary.

    Returns:
        A validated, immutable DefenseProfile instance.
    """
    meta: dict[str, Any] = {
        "preset": "robust_mix_r1",
        "mix_ratios": ROBUST_MIX_RATIO.as_dict(),
        "train_attacks": list(TRAIN_ATTACKS),
        "heldout_attacks": list(HELDOUT_ATTACKS),
        **(dict(metadata) if metadata else {}),
    }
    return DefenseProfile(
        profile_id=profile_id,
        recipe_ids=tuple(dict.fromkeys(recipe_ids)),
        clean_replay_ratio=0.30,
        generated_ratio=0.70,
        hard_example_ratio=0.05,
        sampling_strategy=sampling_strategy,
        max_variants_per_source=max_variants_per_source,
        metadata=meta,
    )


def create_targeted_repair_profile(
    profile_id: str,
    recipe_ids: tuple[str, ...] | list[str] = (),
    *,
    failure_cluster_id: str | None = None,
    sampling_strategy: SamplingStrategy = "failure_cluster_targeted",
    max_variants_per_source: int = 5,
    metadata: Mapping[str, Any] | None = None,
) -> DefenseProfile:
    """Build an R2 Targeted Repair DefenseProfile (50% clean replay / 50% targeted).

    Args:
        profile_id: Unique identifier for this defense profile.
        recipe_ids: Tuple or list of recipe IDs representing the targeted failures.
        failure_cluster_id: Optional ID of the failure cluster being addressed.
        sampling_strategy: Targeted sampling strategy.
        max_variants_per_source: Maximum variants per source sample.
        metadata: Optional extra metadata dictionary.

    Returns:
        A validated, immutable DefenseProfile instance.
    """
    meta: dict[str, Any] = {
        "preset": "targeted_repair_r2",
        "mix_ratios": TARGETED_REPAIR_RATIO.as_dict(),
        "failure_cluster_id": failure_cluster_id,
        **(dict(metadata) if metadata else {}),
    }
    return DefenseProfile(
        profile_id=profile_id,
        recipe_ids=tuple(dict.fromkeys(recipe_ids)),
        clean_replay_ratio=0.50,
        generated_ratio=0.50,
        hard_example_ratio=0.20,
        sampling_strategy=sampling_strategy,
        max_variants_per_source=max_variants_per_source,
        metadata=meta,
    )
