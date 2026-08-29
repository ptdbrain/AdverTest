"""Unit tests for password hashing and JWT token signing."""

from __future__ import annotations

from datetime import timedelta

from src.auth.security import create_access_token, decode_access_token, hash_password, verify_password


def test_password_hashing_and_verification() -> None:
    password = "SuperSecretPassword123!"
    hashed = hash_password(password)

    assert hashed.startswith("pbkdf2_sha256$100000$")
    assert verify_password(password, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False


def test_jwt_token_encode_decode_roundtrip() -> None:
    payload = {"sub": "usr-12345", "email": "engineer@advertest.ai", "role": "ADMIN"}
    token = create_access_token(payload, expires_delta=timedelta(minutes=30))

    claims = decode_access_token(token)
    assert claims is not None
    assert claims["sub"] == "usr-12345"
    assert claims["email"] == "engineer@advertest.ai"
    assert claims["role"] == "ADMIN"
    assert "exp" in claims


def test_jwt_token_tampering_rejected() -> None:
    payload = {"sub": "usr-12345", "role": "USER"}
    token = create_access_token(payload)

    parts = token.split(".")
    # Tamper with payload part
    tampered_token = f"{parts[0]}.eyJyZXBsYWNlZCI6dHJ1ZX0.{parts[2]}"
    assert decode_access_token(tampered_token) is None
