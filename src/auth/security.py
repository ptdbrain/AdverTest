"""Cryptographic security utilities: password hashing (PBKDF2) and JWT token signing."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

# Default secret for local development; should be overridden in production via settings
DEFAULT_JWT_SECRET = "advertest-insecure-development-secret-key-2026"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


def hash_password(password: str, *, salt: str | None = None) -> str:
    """Hash a plaintext password using PBKDF2-HMAC-SHA256 with a cryptographically random salt."""
    salt_hex = salt or secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_hex.encode("utf-8"),
        100_000,
    )
    return f"pbkdf2_sha256$100000${salt_hex}${derived.hex()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify that a plaintext password matches its PBKDF2 hash."""
    parts = hashed_password.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False
    salt = parts[2]
    expected_derived = parts[3]
    actual_derived = hashlib.pbkdf2_hmac(
        "sha256",
        plain_password.encode("utf-8"),
        salt.encode("utf-8"),
        int(parts[1]),
    )
    return hmac.compare_digest(actual_derived.hex(), expected_derived)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    padding = 4 - (len(s) % 4)
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s.encode("ascii"))


def create_access_token(
    payload: dict[str, Any],
    *,
    secret_key: str = DEFAULT_JWT_SECRET,
    expires_delta: timedelta | None = None,
) -> str:
    """Generate an HMAC-SHA256 signed JSON Web Token."""
    delta = expires_delta or timedelta(hours=JWT_EXPIRATION_HOURS)
    now = datetime.now(UTC)
    exp = int((now + delta).timestamp())

    token_claims = {
        **payload,
        "iat": int(now.timestamp()),
        "exp": exp,
    }

    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(token_claims, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode()

    signature = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(signature)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def decode_access_token(token: str, *, secret_key: str = DEFAULT_JWT_SECRET) -> dict[str, Any] | None:
    """Decode and verify signature and expiration of an access token."""
    parts = token.split(".")
    if len(parts) != 3:
        return None

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(secret_key.encode("utf-8"), signing_input, hashlib.sha256).digest()
    actual_sig = _b64url_decode(sig_b64)

    if not hmac.compare_digest(expected_sig, actual_sig):
        return None

    try:
        claims = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception:
        return None

    now_ts = int(datetime.now(UTC).timestamp())
    if "exp" in claims and claims["exp"] < now_ts:
        return None

    return claims
