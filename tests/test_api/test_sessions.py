"""Tests for Experiment Sessions & Multi-Run History API."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

import src.main as main_module


def _session_context(client: TestClient) -> tuple[str, dict[str, str]]:
    registration = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"session-{uuid.uuid4().hex[:10]}@example.com",
            "password": "StrongPassword123!",
            "display_name": "Session Researcher",
        },
    )
    assert registration.status_code == 201, registration.text
    headers = {"Authorization": f"Bearer {registration.json()['access_token']}"}
    project = client.post(
        "/api/v1/projects",
        headers=headers,
        json={"name": f"Session project {uuid.uuid4().hex[:10]}"},
    )
    assert project.status_code == 201, project.text
    return project.json()["id"], headers


def test_list_sessions_endpoint():
    """Verify GET /api/v1/sessions returns list of sessions without mock seed."""
    client = TestClient(main_module.app)
    project_id, headers = _session_context(client)
    response = client.get("/api/v1/sessions", params={"project_id": project_id}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

    # Create a test session
    session_payload = {
        "id": f"EXP-SEED-LIST-{uuid.uuid4().hex[:12]}",
        "name": "Listing Test Session",
        "description": "Session for listing test",
        "task_id": "detection2d",
        "task_name": "Object Detection",
        "model_id": "local_yolo11s_clean",
        "model_name": "YOLO11s",
        "dataset_id": "kitti_anonymized_de",
        "dataset_name": "KITTI De-anonymized Set",
        "created_at": "12/05/2025 10:00:00",
        "updated_at": "12/05/2025 10:00:00",
        "runs": [],
    }
    create_res = client.post(
        "/api/v1/sessions", params={"project_id": project_id}, headers=headers, json=session_payload
    )
    assert create_res.status_code == 200

    # Verify session is in list
    response2 = client.get("/api/v1/sessions", params={"project_id": project_id}, headers=headers)
    assert response2.status_code == 200
    data2 = response2.json()
    assert len(data2) >= 1
    assert any(s["id"] == session_payload["id"] for s in data2)


def test_create_and_get_session_flow():
    """Verify creating a new session and retrieving it."""
    client = TestClient(main_module.app)
    project_id, headers = _session_context(client)
    # The application database can outlive a direct TestClient process; never
    # rely on a globally reusable primary key for this integration contract.
    test_session_id = f"EXP-TEST-SESSION-{uuid.uuid4().hex[:12]}"
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
    create_res = client.post("/api/v1/sessions", params={"project_id": project_id}, headers=headers, json=payload)
    assert create_res.status_code == 200
    assert create_res.json()["id"] == test_session_id

    # Retrieve
    get_res = client.get(f"/api/v1/sessions/{test_session_id}", params={"project_id": project_id}, headers=headers)
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "Test Robustness Session"

    # Add a run
    run_id = f"RUN-{uuid.uuid4().hex[:12]}"
    run_payload = {
        "id": run_id,
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
    add_run_res = client.post(
        f"/api/v1/sessions/{test_session_id}/runs",
        params={"project_id": project_id},
        headers=headers,
        json=run_payload,
    )
    assert add_run_res.status_code == 200
    assert len(add_run_res.json()["runs"]) == 1
    assert add_run_res.json()["runs"][0]["clean_miou"] == 0.75

    # Update note on run
    note_res = client.patch(
        f"/api/v1/sessions/{test_session_id}/runs/{run_id}/note",
        params={"project_id": project_id},
        headers=headers,
        json={"note": "Severe rain causes substantial mIoU drop on small objects"},
    )
    assert note_res.status_code == 200
    assert note_res.json()["runs"][0]["note"] == "Severe rain causes substantial mIoU drop on small objects"

    # End session
    end_res = client.post(f"/api/v1/sessions/{test_session_id}/end", params={"project_id": project_id}, headers=headers)
    assert end_res.status_code == 200
    assert end_res.json()["status"] == "completed"
    assert end_res.json()["ended_at"] is not None

    # Ending again should return 409
    end_again_res = client.post(
        f"/api/v1/sessions/{test_session_id}/end", params={"project_id": project_id}, headers=headers
    )
    assert end_again_res.status_code == 409

    # Cleanup
    del_res = client.delete(f"/api/v1/sessions/{test_session_id}", params={"project_id": project_id}, headers=headers)
    assert del_res.status_code == 200
