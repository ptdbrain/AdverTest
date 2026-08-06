"""Versioned job requests and progress events for compute handoffs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

JobType = Literal["generation", "benchmark", "training"]


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    job_id: str
    job_type: JobType
    requested_by: str
    payload: dict[str, Any] = Field(default_factory=dict)
    seed: int = Field(ge=0)
    contract_version: Literal["1.0.0"] = "1.0.0"
    created_at: datetime


class ProgressEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    job_id: str
    job_type: JobType
    state: str
    progress_ratio: float = Field(ge=0.0, le=1.0)
    sequence: int = Field(ge=0)
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
