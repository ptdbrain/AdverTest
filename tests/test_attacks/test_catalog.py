from __future__ import annotations

import pytest

from src.attacks import ATTACK_CATALOG, load_attacks
from src.attacks.base import BaseAttack
from src.attacks.catalog import CATALOG_ENTRIES, AttackCatalog, AttackCatalogError


def test_catalog_covers_every_registered_attack_and_all_groups() -> None:
    attacks = load_attacks()
    result = ATTACK_CATALOG.list(
        task="detection2d",
        model_capabilities=frozenset(
            {
                "input_gradient",
                "detection_loss",
                "objectness",
                "class_logits",
                "dense_proposals",
                "class_margin",
            }
        ),
        annotation_types=frozenset({"boxes", "mask"}),
        modality="multi",
        online=False,
    )

    assert set(CATALOG_ENTRIES) == set(attacks.names())
    assert {entry.group for entry in result.selected} == {"A", "B", "C", "D", "E", "F"}


def test_catalog_returns_exclusion_reasons_instead_of_silent_drops() -> None:
    load_attacks()

    result = ATTACK_CATALOG.list(
        task="detection2d",
        model_capabilities=frozenset(),
        annotation_types=frozenset(),
        modality="image",
        online=True,
    )
    exclusions = {item.name: set(item.reasons) for item in result.exclusions}

    assert "missing_model_capability:input_gradient" in exclusions["pgd"]
    assert "missing_annotation:boxes" in exclusions["object_occlusion"]
    assert "task_mismatch:segmentation" in exclusions["sam2_pgd"]
    assert "modality_mismatch:lidar" in exclusions["lidar_fog"]
    assert "online_not_supported" in exclusions["cw_l2"]


def test_catalog_validates_missing_entries_and_version_drift() -> None:
    class MissingAttack(BaseAttack):
        name = "missing_catalog"
        group = "A"

        def apply(self, sample, severity, ctx):
            return sample

    with pytest.raises(AttackCatalogError, match="missing_catalog"):
        AttackCatalog(CATALOG_ENTRIES).validate_registry([MissingAttack])

    brightness = load_attacks().get("brightness")
    drifted = type(
        "DriftedBrightness",
        (brightness,),
        {"version": "999.0.0"},
    )
    with pytest.raises(AttackCatalogError, match="version"):
        AttackCatalog(CATALOG_ENTRIES).validate_registry([drifted])
