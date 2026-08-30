"""Regression coverage for the HttpOnly browser-session contract."""

from uuid import uuid4

from fastapi.testclient import TestClient

from src.main import app


def test_register_sets_http_only_session_and_me_survives() -> None:
    client = TestClient(app)
    email = f"persist-{uuid4().hex}@example.test"
    registration = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "correct-horse", "display_name": "Persist User"},
    )
    assert registration.status_code == 201
    assert "access_token" not in registration.json()
    assert "advertest_session=" in registration.headers["set-cookie"]
    assert "HttpOnly" in registration.headers["set-cookie"]
    assert client.get("/api/v1/auth/me").json()["email"] == email


def test_logout_clears_cookie_and_blocks_me() -> None:
    client = TestClient(app)
    client.post(
        "/api/v1/auth/register",
        json={"email": f"logout-{uuid4().hex}@example.test", "password": "correct-horse", "display_name": "Logout User"},
    )
    assert client.post("/api/v1/auth/logout").status_code == 204
    assert client.get("/api/v1/auth/me").status_code == 401
