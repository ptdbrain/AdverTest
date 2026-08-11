"""Wave 1 — BDD100K class mapping and leakage detection tests."""

from __future__ import annotations

import pytest

from src.datasets.bdd100k_detection import (
    BDD100K_CLASS_MAPPING_V1,
    ClassMapping,
    class_mapping_report,
)


class TestClassMapping:
    def test_car_maps_to_car(self):
        assert BDD100K_CLASS_MAPPING_V1.translate("car") == "Car"

    def test_pedestrian_maps_to_pedestrian(self):
        assert BDD100K_CLASS_MAPPING_V1.translate("pedestrian") == "Pedestrian"

    def test_rider_merges_to_cyclist(self):
        assert BDD100K_CLASS_MAPPING_V1.translate("rider") == "Cyclist"

    def test_bicycle_merges_to_cyclist(self):
        assert BDD100K_CLASS_MAPPING_V1.translate("bicycle") == "Cyclist"

    def test_motorcycle_merges_to_cyclist(self):
        assert BDD100K_CLASS_MAPPING_V1.translate("motorcycle") == "Cyclist"

    def test_bus_is_dropped(self):
        assert BDD100K_CLASS_MAPPING_V1.translate("bus") is None

    def test_truck_is_dropped(self):
        assert BDD100K_CLASS_MAPPING_V1.translate("truck") is None

    def test_train_is_dropped(self):
        assert BDD100K_CLASS_MAPPING_V1.translate("train") is None

    def test_traffic_sign_is_dropped(self):
        assert BDD100K_CLASS_MAPPING_V1.translate("traffic sign") is None

    def test_unknown_class_returns_none(self):
        assert BDD100K_CLASS_MAPPING_V1.translate("unknown_class") is None


class TestClassMappingReport:
    def test_report_has_version(self):
        report = class_mapping_report(BDD100K_CLASS_MAPPING_V1)
        assert report["version"] == "bdd100k-det-v1"

    def test_canonical_classes(self):
        report = class_mapping_report(BDD100K_CLASS_MAPPING_V1)
        assert sorted(report["canonical_classes"]) == ["Car", "Cyclist", "Pedestrian"]

    def test_dropped_classes_listed(self):
        report = class_mapping_report(BDD100K_CLASS_MAPPING_V1)
        assert "bus" in report["dropped_classes"]
        assert "truck" in report["dropped_classes"]

    def test_merged_classes_listed(self):
        report = class_mapping_report(BDD100K_CLASS_MAPPING_V1)
        assert "rider" in report["merged_classes"]
        assert report["merged_classes"]["rider"] == "Cyclist"


class TestClassMappingCustom:
    def test_custom_mapping(self):
        custom = ClassMapping(
            version="custom-v1",
            mapping={"car": "Vehicle", "person": "Pedestrian"},
            dropped=("animal",),
        )
        assert custom.translate("car") == "Vehicle"
        assert custom.translate("animal") is None
        assert custom.translate("unknown") is None
