"""Wave 2 — Closed-loop retraining pipeline tests.

Tests the full state machine: FailureCase → Cluster → Backlog → Defense →
Dataset → Training → Checkpoint → Gate → Model → ReBenchmark → Recovery

Verifies:
- Valid transitions advance state and record audit entries
- Invalid transitions are rejected
- Terminal states are unreachable from
- Lineage chain is traceable through all artifacts
- Failure from any state works
"""

from __future__ import annotations

import pytest

from src.training.closed_loop import (
    ClosedLoopAuditEntry,
    ClosedLoopTracker,
    RecoveryReport,
    TrainingDatasetManifest,
    validate_transition,
)


class TestClosedLoopStateMachine:
    """Verify every valid transition and reject invalid ones."""

    def test_full_happy_path(self):
        tracker = ClosedLoopTracker(loop_id="loop-001")
        assert tracker.state == "FAILURE_IDENTIFIED"

        tracker.advance("CLUSTER_FORMED", "cluster-001", "failure_cluster")
        assert tracker.state == "CLUSTER_FORMED"

        tracker.advance("BACKLOG_CREATED", "backlog-001", "retraining_backlog")
        assert tracker.state == "BACKLOG_CREATED"

        tracker.advance("BACKLOG_APPROVED", "backlog-001", "retraining_backlog")
        assert tracker.state == "BACKLOG_APPROVED"

        tracker.advance("DEFENSE_PROFILED", "defense-001", "defense_profile")
        assert tracker.state == "DEFENSE_PROFILED"

        tracker.advance("DATASET_MANIFEST_CREATED", "manifest-001", "training_dataset")
        assert tracker.state == "DATASET_MANIFEST_CREATED"

        tracker.advance("TRAINING_STARTED", "train-001", "training_run")
        assert tracker.state == "TRAINING_STARTED"

        tracker.advance("TRAINING_COMPLETED", "train-001", "training_run")
        assert tracker.state == "TRAINING_COMPLETED"

        tracker.advance("CHECKPOINT_VALIDATED", "ckpt-001", "checkpoint")
        assert tracker.state == "CHECKPOINT_VALIDATED"

        tracker.advance("GATE_EVALUATED", "gate-001", "gate_result")
        assert tracker.state == "GATE_EVALUATED"

        tracker.advance("MODEL_REGISTERED", "yolo-r1", "model_version")
        assert tracker.state == "MODEL_REGISTERED"

        tracker.advance("RE_BENCHMARK_STARTED", "bench-001", "benchmark_run")
        assert tracker.state == "RE_BENCHMARK_STARTED"

        tracker.advance("RE_BENCHMARK_COMPLETED", "bench-001", "benchmark_run")
        assert tracker.state == "RE_BENCHMARK_COMPLETED"

        tracker.advance("RECOVERY_REPORTED", "recovery-001", "recovery_report")
        assert tracker.state == "RECOVERY_REPORTED"

        assert tracker.is_complete
        assert len(tracker.audit) == 13

    def test_skip_transition_rejected(self):
        tracker = ClosedLoopTracker(loop_id="loop-002")
        with pytest.raises(ValueError, match="Invalid transition"):
            tracker.advance("BACKLOG_CREATED", "x", "x")

    def test_backward_transition_rejected(self):
        tracker = ClosedLoopTracker(loop_id="loop-003")
        tracker.advance("CLUSTER_FORMED", "c1", "cluster")
        tracker.advance("BACKLOG_CREATED", "b1", "backlog")
        with pytest.raises(ValueError, match="Invalid transition"):
            tracker.advance("CLUSTER_FORMED", "c2", "cluster")

    def test_terminal_state_is_stuck(self):
        tracker = ClosedLoopTracker(loop_id="loop-004")
        tracker.fail("test failure")
        assert tracker.is_complete
        with pytest.raises(ValueError, match="Invalid transition"):
            tracker.advance("CLUSTER_FORMED", "c1", "cluster")

    def test_fail_from_any_state(self):
        for initial_step in ["CLUSTER_FORMED", "BACKLOG_CREATED", "TRAINING_STARTED"]:
            tracker = ClosedLoopTracker(loop_id=f"loop-fail-{initial_step}")
            if initial_step == "CLUSTER_FORMED":
                tracker.advance("CLUSTER_FORMED", "c1", "cluster")
            elif initial_step == "BACKLOG_CREATED":
                tracker.advance("CLUSTER_FORMED", "c1", "cluster")
                tracker.advance("BACKLOG_CREATED", "b1", "backlog")
            elif initial_step == "TRAINING_STARTED":
                tracker.advance("CLUSTER_FORMED", "c1", "cluster")
                tracker.advance("BACKLOG_CREATED", "b1", "backlog")
                tracker.advance("BACKLOG_APPROVED", "b1", "backlog")
                tracker.advance("DEFENSE_PROFILED", "d1", "defense")
                tracker.advance("DATASET_MANIFEST_CREATED", "m1", "manifest")
                tracker.advance("TRAINING_STARTED", "t1", "training")
            tracker.fail(f"failed at {initial_step}")
            assert tracker.state == "FAILED"
            assert tracker.is_complete


class TestAuditTrail:
    """Verify audit entries capture full lineage."""

    def test_audit_entries_have_correct_steps(self):
        tracker = ClosedLoopTracker(loop_id="loop-audit")
        tracker.advance("CLUSTER_FORMED", "c1", "failure_cluster", artifact_hash="hash_c1")
        tracker.advance("BACKLOG_CREATED", "b1", "retraining_backlog", artifact_hash="hash_b1")

        assert tracker.audit[0].step == 0
        assert tracker.audit[0].parent_step is None
        assert tracker.audit[0].artifact_hash == "hash_c1"

        assert tracker.audit[1].step == 1
        assert tracker.audit[1].parent_step == 0
        assert tracker.audit[1].artifact_hash == "hash_b1"

    def test_lineage_chain(self):
        tracker = ClosedLoopTracker(loop_id="loop-lineage")
        tracker.advance("CLUSTER_FORMED", "c1", "cluster")
        tracker.advance("BACKLOG_CREATED", "b1", "backlog")
        tracker.advance("BACKLOG_APPROVED", "b1", "backlog")

        chain = tracker.lineage_chain()
        assert chain == ["c1", "b1", "b1"]

    def test_artifacts_tracked_by_type(self):
        tracker = ClosedLoopTracker(loop_id="loop-art")
        tracker.advance("CLUSTER_FORMED", "c1", "failure_cluster")
        tracker.advance("BACKLOG_CREATED", "b1", "retraining_backlog")
        assert tracker.artifacts["failure_cluster"] == "c1"
        assert tracker.artifacts["retraining_backlog"] == "b1"

    def test_metadata_is_preserved(self):
        tracker = ClosedLoopTracker(loop_id="loop-meta")
        tracker.advance(
            "CLUSTER_FORMED", "c1", "cluster",
            metadata={"failure_count": 42, "attack": "fog"},
        )
        assert tracker.audit[0].metadata["failure_count"] == 42


class TestTrainingDatasetManifest:
    """Verify training dataset manifest contract."""

    def test_valid_manifest(self):
        m = TrainingDatasetManifest(
            manifest_id="tdm-001",
            source_dataset_version_id="ds-v1",
            clean_sample_count=1000,
            generated_sample_count=500,
            clean_ratio=0.667,
            defense_profile_id="def-001",
            seed=42,
        )
        assert m.clean_ratio == 0.667
        assert m.locked_test_excluded is True

    def test_leakage_hash_optional(self):
        m = TrainingDatasetManifest(
            manifest_id="tdm-002",
            source_dataset_version_id="ds-v1",
            clean_sample_count=100,
            generated_sample_count=100,
            clean_ratio=0.5,
            defense_profile_id="def-001",
            seed=42,
            leakage_report_hash="sha256-leak-clean",
        )
        assert m.leakage_report_hash is not None


class TestRecoveryReport:
    """Verify recovery report contract."""

    def test_recovery_report_creation(self):
        r = RecoveryReport(
            report_id="rr-001",
            baseline_model_id="yolo-b0",
            candidate_model_id="yolo-r1",
            protocol_id="proto-001",
            baseline_run_id="run-b0",
            candidate_run_id="run-r1",
            recovery_rate=0.65,
            clean_delta=-0.01,
            robust_score_delta=12.0,
            degradation_delta=-0.15,
            failure_count_delta=-7,
            gate_passed=True,
        )
        assert r.gate_passed
        assert r.recovery_rate == 0.65
        assert r.clean_delta == -0.01

    def test_recovery_with_audit_trail(self):
        entries = (
            ClosedLoopAuditEntry(step=0, state="CLUSTER_FORMED", artifact_id="c1", artifact_type="cluster"),
            ClosedLoopAuditEntry(step=1, state="BACKLOG_CREATED", artifact_id="b1", artifact_type="backlog", parent_step=0),
        )
        r = RecoveryReport(
            report_id="rr-002",
            baseline_model_id="b0",
            candidate_model_id="r1",
            protocol_id="p1",
            baseline_run_id="run-b0",
            candidate_run_id="run-r1",
            audit_trail=entries,
        )
        assert len(r.audit_trail) == 2
        assert r.audit_trail[1].parent_step == 0


class TestTransitionValidation:
    """Unit tests for the transition validator."""

    def test_valid_forward(self):
        assert validate_transition("FAILURE_IDENTIFIED", "CLUSTER_FORMED") is True
        assert validate_transition("TRAINING_STARTED", "TRAINING_COMPLETED") is True

    def test_invalid_skip(self):
        assert validate_transition("FAILURE_IDENTIFIED", "TRAINING_STARTED") is False

    def test_fail_always_valid(self):
        for state in ["FAILURE_IDENTIFIED", "CLUSTER_FORMED", "TRAINING_STARTED", "GATE_EVALUATED"]:
            assert validate_transition(state, "FAILED") is True

    def test_cancel_valid_from_appropriate_states(self):
        assert validate_transition("BACKLOG_CREATED", "CANCELLED") is True
        assert validate_transition("TRAINING_STARTED", "CANCELLED") is True

    def test_cancel_invalid_after_training_complete(self):
        assert validate_transition("TRAINING_COMPLETED", "CANCELLED") is False
        assert validate_transition("GATE_EVALUATED", "CANCELLED") is False
