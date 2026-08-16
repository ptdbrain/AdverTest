"""Wave 0 — Common event envelope contract tests."""

from __future__ import annotations

import pytest

from src.core.events import EventEnvelope


class TestEventEnvelope:
    """Verify the common WebSocket envelope schema."""

    def test_valid_envelope(self):
        e = EventEnvelope(
            sequence=0,
            job_id="job-001",
            job_type="benchmark",
            state="GENERATING",
            progress=0.42,
            payload={"completed_cells": 5},
            created_at="2026-08-11T12:00:00Z",
        )
        assert e.sequence == 0
        assert e.progress == 0.42
        assert e.contract_version == "1.0.0"

    def test_envelope_all_job_types(self):
        for job_type in ("generation", "benchmark", "training"):
            e = EventEnvelope(
                sequence=1,
                job_id="j1",
                job_type=job_type,
                state="RUNNING",
                created_at="2026-08-11T12:00:00Z",
            )
            assert e.job_type == job_type

    def test_progress_bounds(self):
        with pytest.raises(ValueError):
            EventEnvelope(
                sequence=0, job_id="j", job_type="benchmark",
                state="X", progress=-0.1, created_at="2026-08-11T12:00:00Z",
            )
        with pytest.raises(ValueError):
            EventEnvelope(
                sequence=0, job_id="j", job_type="benchmark",
                state="X", progress=1.1, created_at="2026-08-11T12:00:00Z",
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValueError):
            EventEnvelope(
                sequence=0, job_id="j", job_type="benchmark",
                state="X", created_at="2026-08-11T12:00:00Z",
                unknown_field="bad",
            )

    def test_serialization_roundtrip(self):
        e = EventEnvelope(
            sequence=3,
            job_id="job-002",
            job_type="training",
            state="COMPLETED",
            progress=1.0,
            payload={"epoch": 50, "loss": 0.01},
            created_at="2026-08-11T15:30:00Z",
        )
        data = e.model_dump(mode="json")
        restored = EventEnvelope(**data)
        assert restored == e
