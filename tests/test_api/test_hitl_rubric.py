"""Tests for HITL 5-tier smart funnel, risk rubric, smart clustering, and downstream actions."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.evaluation.risk_rubric import (
    assess_risk,
    classify_risk_level,
    generate_risk_summary,
)
from src.evaluation.smart_clustering import (
    SmartCluster,
    SmartFailureClusterer,
    _infer_attack_group,
)


@pytest.fixture
def client() -> TestClient:
    import src.main as main_module

    return TestClient(main_module.app)


# ── Risk Classification Unit Tests ───────────────────────────────────────


def test_classify_risk_level_critical_safety_near():
    """Safety-critical class near with high drop or false negative -> CRITICAL."""
    risk = classify_risk_level(
        degradation_percent=55.0,
        affected_class="Pedestrian",
        distance_meters=12.0,
        has_false_negative=True,
    )
    assert risk == "CRITICAL"


def test_classify_risk_level_critical_extreme_drop():
    """Extreme degradation >= 80% -> CRITICAL."""
    risk = classify_risk_level(
        degradation_percent=85.0,
        affected_class="Car",
        distance_meters=45.0,
    )
    assert risk == "CRITICAL"


def test_classify_risk_level_high_drop():
    """High degradation 50-80% -> HIGH."""
    risk = classify_risk_level(
        degradation_percent=60.0,
        affected_class="Car",
        distance_meters=30.0,
    )
    assert risk == "HIGH"


def test_classify_risk_level_medium_drop():
    """Medium degradation 30-50% -> MEDIUM."""
    risk = classify_risk_level(
        degradation_percent=35.0,
        affected_class="Car",
    )
    assert risk == "MEDIUM"


def test_classify_risk_level_low_drop():
    """Low degradation 15-30% -> LOW."""
    risk = classify_risk_level(
        degradation_percent=20.0,
        affected_class="Car",
    )
    assert risk == "LOW"


def test_classify_risk_level_auto_pass():
    """Degradation < 15% -> AUTO_PASS."""
    risk = classify_risk_level(
        degradation_percent=8.0,
        affected_class="Car",
    )
    assert risk == "AUTO_PASS"


# ── Risk Assessment Engine Tests ─────────────────────────────────────────


def test_assess_risk_structure():
    review = {
        "review_id": "REV-test01",
        "attack": "fog",
        "severity": 4,
        "degradation_percent": 65.0,
        "affected_class": "Pedestrian",
        "distance_meters": 15.0,
        "has_false_negative": True,
    }
    assessment = assess_risk(review)
    assert assessment.risk_level == "CRITICAL"
    assert assessment.recommended_decision in ("BLOCK_DEPLOY", "REQUEST_RETRAIN")
    assert len(assessment.downstream_actions) > 0
    assert len(assessment.contributing_factors) > 0
    assert assessment.confidence > 0.7


def test_generate_risk_summary():
    reviews = [
        {"review_id": "r1", "risk_level": "CRITICAL", "affected_class": "Pedestrian", "attack": "fgsm"},
        {"review_id": "r2", "risk_level": "HIGH", "affected_class": "Car", "attack": "fog"},
        {"review_id": "r3", "risk_level": "MEDIUM", "affected_class": "Car", "attack": "snow"},
    ]
    summary = generate_risk_summary(reviews)
    assert summary.total_reviews == 3
    assert summary.critical_count == 1
    assert summary.high_count == 1
    assert summary.medium_count == 1
    assert summary.dominant_risk_level == "CRITICAL"


# ── Smart Clustering Tests ───────────────────────────────────────────────


def test_smart_clustering_groups_similar_failures():
    reviews = [
        # Group 1: Fog weather failures
        {
            "review_id": "r1",
            "attack": "fog",
            "severity": 4,
            "degradation_percent": 55.0,
            "risk_level": "HIGH",
            "affected_class": "Pedestrian",
        },
        {
            "review_id": "r2",
            "attack": "depth_fog",
            "severity": 4,
            "degradation_percent": 52.0,
            "risk_level": "HIGH",
            "affected_class": "Pedestrian",
        },
        {
            "review_id": "r3",
            "attack": "snow",
            "severity": 3,
            "degradation_percent": 48.0,
            "risk_level": "MEDIUM",
            "affected_class": "Pedestrian",
        },
        # Group 2: Gradient adversarial failures
        {
            "review_id": "r4",
            "attack": "fgsm",
            "severity": 1,
            "degradation_percent": 82.0,
            "risk_level": "CRITICAL",
            "affected_class": "Car",
        },
        {
            "review_id": "r5",
            "attack": "pgd",
            "severity": 2,
            "degradation_percent": 88.0,
            "risk_level": "CRITICAL",
            "affected_class": "Car",
        },
    ]
    clusterer = SmartFailureClusterer(max_clusters=4, distance_threshold=0.4)
    clusters = clusterer.cluster(reviews)

    assert len(clusters) >= 1
    assert all(isinstance(c, SmartCluster) for c in clusters)
    assert all(len(c.member_review_ids) > 0 for c in clusters)
    assert all(c.representative_review_id != "" for c in clusters)


def test_infer_attack_group():
    assert _infer_attack_group("gaussian_noise") == "corruption"
    assert _infer_attack_group("depth_fog") == "weather"
    assert _infer_attack_group("fgsm") == "adversarial"
    assert _infer_attack_group("dpatch") == "patch"
    assert _infer_attack_group("square_attack") == "blackbox"


# ── API Router Tests ─────────────────────────────────────────────────────


def test_api_get_risk_rubric(client: TestClient):
    res = client.get("/api/v1/risk-rubric")
    assert res.status_code == 200
    rubric = res.json()
    assert isinstance(rubric, list)
    assert len(rubric) == 5
    levels = [item["risk_level"] for item in rubric]
    assert "CRITICAL" in levels
    assert "AUTO_PASS" in levels


def test_api_risk_session_summary(client: TestClient):
    res = client.get("/api/v1/risk-rubric/session-summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_reviews" in data
    assert "critical_count" in data
    assert "dominant_risk_level" in data


def test_api_auto_group_clusters_empty_or_valid(client: TestClient):
    res = client.post("/api/v1/failure-clusters/auto-group")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_api_resolve_review_with_new_decisions(client: TestClient):
    import uuid

    unique_run_id = f"test-run-hitl-{uuid.uuid4().hex[:8]}"

    # 1. Create a manual review with unique run_id
    create_res = client.post(
        "/api/v1/reviews",
        json={
            "run_id": unique_run_id,
            "attack": "fog",
            "severity": 4,
            "degradation": 0.45,
            "dataset": "KITTI",
            "model": "YOLO11s",
            "flagged_by": "manual",
            "notes": "Test review case",
        },
    )
    assert create_res.status_code == 201
    rev = create_res.json()
    review_id = rev["review_id"]

    # 2. Assess risk
    assess_res = client.post(f"/api/v1/risk-rubric/assess?review_id={review_id}")
    assert assess_res.status_code == 200
    assert "recommended_decision" in assess_res.json()

    # 3. Resolve with BLOCK_DEPLOY
    resolve_res = client.patch(
        f"/api/v1/reviews/{review_id}",
        json={
            "decision": "BLOCK_DEPLOY",
            "decision_note": "Chặn triển khai do sụt giảm nghiêm trọng",
            "resolved_by": "Staff Reviewer",
        },
    )
    assert resolve_res.status_code == 200
    resolved = resolve_res.json()
    assert resolved["status"] == "RESOLVED"
    assert resolved["decision"] == "BLOCK_DEPLOY"
