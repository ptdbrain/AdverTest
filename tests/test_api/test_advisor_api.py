"""API integration tests for the AI Advisor endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_advisor_service, get_store
from src.main import app


@pytest.fixture
def client():
    # Clear dismissed recommendations before each test
    store = app.dependency_overrides.get(get_store, get_store)()
    advisor = get_advisor_service(store=store)
    advisor.clear_dismissed()
    return TestClient(app)


def test_advisor_recommendations_endpoint_returns_200(client) -> None:
    response = client.get("/api/v1/advisor/recommendations")
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert "total_count" in data
    assert isinstance(data["recommendations"], list)


def test_advisor_dismiss_endpoint(client) -> None:
    # Populate a pending checkpoint to trigger a recommendation
    store = app.dependency_overrides.get(get_store, get_store)()
    store.put_record(
        "checkpoint",
        "ckpt-test-dismiss",
        {
            "checkpoint_id": "ckpt-test-dismiss",
            "display_name": "Dismiss Test Checkpoint",
            "status": "PENDING_VALIDATION",
        },
    )

    recs_res = client.get("/api/v1/advisor/recommendations")
    assert recs_res.status_code == 200
    recs = recs_res.json()["recommendations"]
    assert len(recs) > 0

    target_id = recs[0]["id"]
    dismiss_res = client.post(
        "/api/v1/advisor/dismiss",
        json={"recommendation_id": target_id, "reason": "Not relevant right now"},
    )
    assert dismiss_res.status_code == 200
    assert dismiss_res.json()["status"] == "dismissed"

    recs_after = client.get("/api/v1/advisor/recommendations").json()["recommendations"]
    assert all(r["id"] != target_id for r in recs_after)
