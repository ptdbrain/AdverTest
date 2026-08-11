"""Machine-readable error codes for the AdverTest API.

Every error response includes a `code` field so clients can branch on specific
failures without parsing human-readable messages.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


# Standard error code constants
WAITING_FOR_ARTIFACTS = "WAITING_FOR_ARTIFACTS"
PROTOCOL_MISMATCH = "PROTOCOL_MISMATCH"
LEAKAGE_DETECTED = "LEAKAGE_DETECTED"
CHECKPOINT_NOT_RUNNABLE = "CHECKPOINT_NOT_RUNNABLE"
UNPAIRED_COMPARISON = "UNPAIRED_COMPARISON"
TASK_INCOMPATIBLE = "TASK_INCOMPATIBLE"
DATASET_VERSION_CONFLICT = "DATASET_VERSION_CONFLICT"
DATASET_ROOT_MISSING = "DATASET_ROOT_MISSING"
TRAINING_RUN_UNKNOWN = "TRAINING_RUN_UNKNOWN"
TRAINER_NOT_AVAILABLE = "TRAINER_NOT_AVAILABLE"
BACKLOG_UNKNOWN = "BACKLOG_UNKNOWN"
BACKLOG_NOT_DRAFT = "BACKLOG_NOT_DRAFT"
BACKLOG_EMPTY = "BACKLOG_EMPTY"
MISSING_FIELDS = "MISSING_FIELDS"
INVALID_PROTOCOL = "INVALID_PROTOCOL"
INVALID_TRANSITION = "INVALID_TRANSITION"
RUN_NOT_COMPLETED = "RUN_NOT_COMPLETED"
RUN_UNKNOWN = "RUN_UNKNOWN"


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    context: dict[str, Any] | None = None,
) -> HTTPException:
    """Create a structured error response with machine-readable code."""
    detail: dict[str, Any] = {"code": code, "message": message}
    if context:
        detail["context"] = context
    return HTTPException(status_code=status_code, detail=detail)
