"""Tests for Experiment Sessions & Multi-Run History API."""

from __future__ import annotations

from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_list_sessions_endpoint():
    """Verify GET /api/v1/sessions returns list of sessions."""
    response = client.get("/api/v1/sessions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "id" in data[0]
    assert "runs" in data[0]


def test_create_and_get_session_flow():
    """Verify creating a new session and retrieving it."""
    test_session_id = "EXP-TEST-SESSION-001"
    payload = {
        "id": test_session_id,
        "name": "Test Robustness Session",
        "description": "Session for testing multi-run history",
        "task_id": "detection2d",
        "task_name": "Object Detection",
        "model_id": "local_yolo11s_clean",
        "model_name": "YOLO11s (Ultralytics Vision)",
        "dataset_id": "kitti_anonymized_de",
        "dataset_name": "KITTI De-anonymized Set",
        "created_at": "12/05/2025 10:00:00",
        "updated_at": "12/05/2025 10:00:00",
        "runs": [],
    }
    # Create
    create_res = client.post("/api/v1/sessions", json=payload)
    assert create_res.status_code == 200
    assert create_res.json()["id"] == test_session_id

    # Retrieve
    get_res = client.get(f"/api/v1/sessions/{test_session_id}")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Test Robustness Session"

    # Add a run
    run_payload = {
        "id": "RUN-001",
        "name": "Lần 1: Depth Rain Cấp 4",
        "timestamp": "12/05/2025 10:05:00",
        "attack_type": "depth_rain",
        "attack_name": "Depth Rain (Mưa giông)",
        "severity": 4,
        "clean_map": 0.82,
        "attacked_map": 0.22,
        "map_drop_pct": 73.2,
        "clean_conf": 0.92,
        "attacked_conf": 0.35,
        "psnr": "22.15 dB",
        "ssim": "0.710",
        "inference_ms": 52.8,
        "robustness_score": 26.8,
        "clean_bbox_count": 10,
        "attacked_bbox_count": 2,
        "sample_id": "000000",
        "clean_miou": 0.75,
        "attacked_miou": 0.42,
        "miou_drop_pct": 44.0,
        "is_combined": False,
        "note": "Initial note",
    }
    add_run_res = client.post(f"/api/v1/sessions/{test_session_id}/runs", json=run_payload)
    assert add_run_res.status_code == 200
    assert len(add_run_res.json()["runs"]) == 1
    assert add_run_res.json()["runs"][0]["clean_miou"] == 0.75

    # Update note on run
    note_res = client.patch(
        f"/api/v1/sessions/{test_session_id}/runs/RUN-001/note",
        json={"note": "Severe rain causes substantial mIoU drop on small objects"},
    )
    assert note_res.status_code == 200
    assert note_res.json()["runs"][0]["note"] == "Severe rain causes substantial mIoU drop on small objects"

    # End session
    end_res = client.post(f"/api/v1/sessions/{test_session_id}/end")
    assert end_res.status_code == 200
    assert end_res.json()["status"] == "completed"
    assert end_res.json()["ended_at"] is not None

    # Ending again should return 409
    end_again_res = client.post(f"/api/v1/sessions/{test_session_id}/end")
    assert end_again_res.status_code == 409

    # Cleanup
    del_res = client.delete(f"/api/v1/sessions/{test_session_id}")
    assert del_res.status_code == 200
