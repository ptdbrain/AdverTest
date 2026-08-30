"""Unit tests for the AI Advisor deterministic rule engine and service."""

from __future__ import annotations

from src.agents.advisor_service import AIAdvisorService
from src.agents.rule_engine import evaluate_rules
from src.api.jobs import SqliteRunStore


def test_advisor_recommends_dataset_annotation_when_unlabeled(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    store = SqliteRunStore(f"sqlite:///{db_path}")

    # Insert an unlabeled dataset
    store.put_record(
        "dataset_version",
        "ds-unlabeled-1",
        {
            "id": "ds-unlabeled-1",
            "name": "Custom Drone Dataset",
            "status": "UNLABELED",
        },
    )

    recs = evaluate_rules(store)
    assert len(recs) >= 1
    ds_rec = next(r for r in recs if r.action_type == "LABEL_DATASET")
    assert ds_rec.priority == "CRITICAL"
    assert "ds-unlabeled-1" in ds_rec.evidence[0]


def test_advisor_recommends_defense_generation_on_severe_degradation(tmp_path) -> None:
    from src.pipeline import RunConfig

    db_path = tmp_path / "test.db"
    store = SqliteRunStore(f"sqlite:///{db_path}")

    # Insert a run with severe fog degradation (>25%)
    run_id = store.create(RunConfig(model="yolo11n", dataset="kitti", attacks=["fog"]))
    store.complete(
        run_id,
        {
            "run_id": run_id,
            "task": "detection2d",
            "ap_clean": 0.80,
            "cells": [
                {"attack": "fog", "severity": 3, "ap": 0.30, "group": "weather"},  # 62.5% degradation
                {"attack": "noise", "severity": 3, "ap": 0.75, "group": "corruption"},
            ],
            "worst_cases": [],
        },
    )

    recs = evaluate_rules(store, run_id=run_id)
    assert any(r.action_type == "GENERATE_DEFENCE_DATASET" for r in recs)
    fog_rec = next(r for r in recs if r.action_type == "GENERATE_DEFENCE_DATASET")
    assert fog_rec.priority == "HIGH"
    assert "fog" in fog_rec.suggested_parameters.get("target_attack", "")


def test_advisor_service_dismissal_behavior(tmp_path) -> None:
    db_path = tmp_path / "test.db"
    store = SqliteRunStore(f"sqlite:///{db_path}")

    store.put_record(
        "checkpoint",
        "ckpt-pending-1",
        {
            "checkpoint_id": "ckpt-pending-1",
            "display_name": "YOLO11s Custom",
            "status": "PENDING_VALIDATION",
        },
    )

    service = AIAdvisorService(store)
    recs_out = service.get_recommendations()
    assert recs_out.total_count >= 1

    first_id = recs_out.recommendations[0].id
    service.dismiss(first_id)

    recs_after = service.get_recommendations()
    assert all(r.id != first_id for r in recs_after.recommendations)
