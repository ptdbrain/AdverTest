"""Content-addressed keys for the variant cache (plan §5).

A variant is uniquely identified by ``(sample, attack, params, severity,
model_version)``. Same key means the forward pass can be skipped.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def stable_digest(payload: Any, *, length: int = 16) -> str:
    """SHA-256 of a JSON-normalised payload, truncated for readability."""
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def array_digest(array: np.ndarray, *, length: int = 16) -> str:
    """Digest of raw array bytes; use to assert two images are identical."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()[:length]


def variant_key(
    *,
    sample_id: str,
    attack: str,
    params: dict[str, Any],
    severity: int,
    model_version: str,
) -> str:
    """Cache key for one (image, attack, params, severity, model) combination."""
    return stable_digest(
        {
            "sample": sample_id,
            "attack": attack,
            "params": params,
            "severity": severity,
            "model": model_version,
        }
    )


def clean_key(*, sample_id: str, model_version: str) -> str:
    """Cache key for a clean prediction (reused across every comparison)."""
    return stable_digest({"sample": sample_id, "model": model_version, "attack": "clean"})
