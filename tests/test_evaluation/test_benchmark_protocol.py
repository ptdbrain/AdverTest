"""Tests for the typed BenchmarkProtocol and state transitions."""

from __future__ import annotations

import pytest

from src.evaluation.benchmark_protocol import (
    BenchmarkProtocol,
    BenchmarkRunManifest,
    validate_protocol_transition,
)


class TestBenchmarkProtocol:
    def test_draft_protocol_creation(self):
        protocol = BenchmarkProtocol(
            protocol_id="proto-001",
            dataset_version_id="ds-v1",
            model_version_id="yolo-b0",
            task="detection2d",
            recipe={"steps": [{"attack": "noise", "severity": 3}]},
            recipe_hash="abc123",
        )
        assert protocol.state == "DRAFT"
        assert protocol.protocol_id == "proto-001"

    def test_locked_protocol_requires_sample_ids(self):
        with pytest.raises(ValueError, match="sample_ids"):
            BenchmarkProtocol(
                protocol_id="proto-002",
                state="LOCKED",
                dataset_version_id="ds-v1",
                model_version_id="yolo-b0",
                task="detection2d",
                recipe={},
                recipe_hash="abc",
                sample_ids=(),
            )

    def test_locked_protocol_with_samples(self):
        protocol = BenchmarkProtocol(
            protocol_id="proto-003",
            state="LOCKED",
            dataset_version_id="ds-v1",
            model_version_id="yolo-b0",
            task="detection2d",
            recipe={"steps": []},
            recipe_hash="abc",
            sample_ids=("s1", "s2"),
            sample_hashes=("h1", "h2"),
        )
        assert protocol.state == "LOCKED"

    def test_identity_hash_stable(self):
        kwargs = dict(
            protocol_id="proto-004",
            dataset_version_id="ds-v1",
            model_version_id="yolo-b0",
            task="detection2d",
            recipe={},
            recipe_hash="abc",
            sample_ids=("s1",),
            sample_hashes=("h1",),
        )
        a = BenchmarkProtocol(**kwargs)
        b = BenchmarkProtocol(**kwargs)
        assert a.identity_hash() == b.identity_hash()

    def test_identity_changes_with_sample(self):
        base = dict(
            protocol_id="proto-005",
            dataset_version_id="ds-v1",
            model_version_id="yolo-b0",
            task="detection2d",
            recipe={},
            recipe_hash="abc",
        )
        a = BenchmarkProtocol(**base, sample_ids=("s1",), sample_hashes=("h1",))
        b = BenchmarkProtocol(**base, sample_ids=("s1", "s2"), sample_hashes=("h1", "h2"))
        assert a.identity_hash() != b.identity_hash()

    def test_identity_changes_with_recipe(self):
        base = dict(
            protocol_id="proto-006",
            dataset_version_id="ds-v1",
            model_version_id="yolo-b0",
            task="detection2d",
            recipe={},
        )
        a = BenchmarkProtocol(**base, recipe_hash="abc")
        b = BenchmarkProtocol(**base, recipe_hash="def")
        assert a.identity_hash() != b.identity_hash()

    def test_identity_changes_with_threshold(self):
        base = dict(
            protocol_id="proto-007",
            dataset_version_id="ds-v1",
            model_version_id="yolo-b0",
            task="detection2d",
            recipe={},
            recipe_hash="abc",
        )
        a = BenchmarkProtocol(**base, thresholds={"iou": 0.5})
        b = BenchmarkProtocol(**base, thresholds={"iou": 0.75})
        assert a.identity_hash() != b.identity_hash()

    def test_identity_changes_with_metric_version(self):
        base = dict(
            protocol_id="proto-008",
            dataset_version_id="ds-v1",
            model_version_id="yolo-b0",
            task="detection2d",
            recipe={},
            recipe_hash="abc",
        )
        a = BenchmarkProtocol(**base, metric_versions={"ap": "1.0"})
        b = BenchmarkProtocol(**base, metric_versions={"ap": "2.0"})
        assert a.identity_hash() != b.identity_hash()


class TestProtocolTransitions:
    def test_draft_to_validated(self):
        assert validate_protocol_transition("DRAFT", "VALIDATED") is True

    def test_validated_to_locked(self):
        assert validate_protocol_transition("VALIDATED", "LOCKED") is True

    def test_locked_cannot_change(self):
        assert validate_protocol_transition("LOCKED", "DRAFT") is False
        assert validate_protocol_transition("LOCKED", "VALIDATED") is False
        assert validate_protocol_transition("LOCKED", "LOCKED") is False

    def test_backward_transition_blocked(self):
        assert validate_protocol_transition("VALIDATED", "DRAFT") is False

    def test_skip_transition_blocked(self):
        assert validate_protocol_transition("DRAFT", "LOCKED") is False


class TestBenchmarkRunManifest:
    def test_manifest_creation(self):
        manifest = BenchmarkRunManifest(
            run_id="run-001",
            protocol_id="proto-001",
            checkpoint_hash="sha256abcdef",
            sample_order=("s1", "s2"),
        )
        assert manifest.protocol_id == "proto-001"
        assert manifest.checkpoint_hash == "sha256abcdef"

    def test_protocol_and_manifest_decoupled(self):
        """Checkpoint hash in manifest, not protocol — so B0/R1/R2 share protocol."""
        proto = BenchmarkProtocol(
            protocol_id="proto-010",
            dataset_version_id="ds-v1",
            model_version_id="yolo-b0",
            task="detection2d",
            recipe={},
            recipe_hash="abc",
            sample_ids=("s1",),
            sample_hashes=("h1",),
        )
        m_b0 = BenchmarkRunManifest(
            run_id="run-b0", protocol_id=proto.protocol_id, checkpoint_hash="hash_b0",
        )
        m_r1 = BenchmarkRunManifest(
            run_id="run-r1", protocol_id=proto.protocol_id, checkpoint_hash="hash_r1",
        )
        # Same protocol, different checkpoints
        assert m_b0.protocol_id == m_r1.protocol_id
        assert m_b0.checkpoint_hash != m_r1.checkpoint_hash
