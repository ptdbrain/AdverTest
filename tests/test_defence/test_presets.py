"""Unit tests for defence presets and attack mix partitioning."""

from __future__ import annotations

from src.defence.presets import (
    HELDOUT_ATTACKS,
    TRAIN_ATTACKS,
    VALIDATION_ATTACKS,
    create_robust_mix_profile,
    create_targeted_repair_profile,
)


def test_attack_partition_disjoint() -> None:
    """Ensure TRAIN, VALIDATION, and HELDOUT attacks do not overlap."""
    train_set = set(TRAIN_ATTACKS)
    val_set = set(VALIDATION_ATTACKS)
    heldout_set = set(HELDOUT_ATTACKS)

    assert len(train_set) > 0
    assert len(val_set) > 0
    assert len(heldout_set) > 0

    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(heldout_set)
    assert val_set.isdisjoint(heldout_set)


def test_create_robust_mix_profile() -> None:
    profile = create_robust_mix_profile(
        profile_id="dp-robust-01",
        recipe_ids=["rec_fog", "rec_noise"],
    )

    assert profile.profile_id == "dp-robust-01"
    assert profile.clean_replay_ratio == 0.30
    assert profile.generated_ratio == 0.70
    assert profile.hard_example_ratio == 0.05
    assert profile.recipe_ids == ("rec_fog", "rec_noise")
    assert profile.metadata["preset"] == "robust_mix_r1"


def test_create_targeted_repair_profile() -> None:
    profile = create_targeted_repair_profile(
        profile_id="dp-repair-01",
        recipe_ids=["rec_targeted_1"],
        failure_cluster_id="cluster-99",
    )

    assert profile.profile_id == "dp-repair-01"
    assert profile.clean_replay_ratio == 0.50
    assert profile.generated_ratio == 0.50
    assert profile.hard_example_ratio == 0.20
    assert profile.metadata["preset"] == "targeted_repair_r2"
    assert profile.metadata["failure_cluster_id"] == "cluster-99"
