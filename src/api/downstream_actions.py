"""Automated downstream action chains triggered by reviewer decisions.

When a reviewer resolves a review item, this module orchestrates the
appropriate follow-up actions: creating retraining backlogs, defense
profiles, ODD constraints, or annotation tasks.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def trigger_downstream_actions(
    decision: str,
    review: dict[str, Any],
    store: Any,
    workflow_store: Any,
) -> dict[str, Any]:
    """Dispatch downstream actions based on the reviewer's decision.

    Args:
        decision: One of BLOCK_DEPLOY, REQUEST_RETRAIN, RESTRICT_ODD,
                  RELABEL_DATA, or ACCEPT_RISK.
        review: The resolved review item dict.
        store: SqliteRunStore instance for product records.
        workflow_store: WorkflowJobStore instance for backlogs.

    Returns:
        Dict summarizing all created downstream artifacts.
    """
    result: dict[str, Any] = {"decision": decision, "review_id": review.get("review_id", "")}

    if decision == "BLOCK_DEPLOY":
        retrain = _trigger_retrain_chain(review, store, workflow_store)
        gate = _trigger_deploy_block(review, store)
        result["retrain"] = retrain
        result["deploy_gate"] = gate
        return result

    if decision == "REQUEST_RETRAIN":
        retrain = _trigger_retrain_chain(review, store, workflow_store)
        result["retrain"] = retrain
        return result

    if decision == "RESTRICT_ODD":
        odd = _trigger_odd_restriction(review, store)
        result["odd_constraint"] = odd
        return result

    if decision == "RELABEL_DATA":
        relabel = _trigger_relabel(review, store)
        result["annotation_task"] = relabel
        return result

    if decision == "ACCEPT_RISK":
        audit = _record_risk_acceptance(review, store)
        result["audit_entry"] = audit
        return result

    result["warning"] = f"Unknown decision '{decision}', no downstream actions triggered."
    return result


def _trigger_retrain_chain(
    review: dict[str, Any],
    store: Any,
    workflow_store: Any,
) -> dict[str, Any]:
    """Create retraining backlog + add failure item + suggest defense profile."""
    review_id = review.get("review_id", "")
    attack = review.get("attack", "unknown")
    severity = review.get("severity", 3)
    risk_level = review.get("risk_level", "HIGH")

    backlog_name = f"HITL-Retrain: {attack} sev{severity} [{risk_level}]"
    backlog = workflow_store.create_backlog(backlog_name)
    backlog_id = backlog.get("id", backlog.get("backlog_id", ""))

    try:
        workflow_store.add_backlog_item(backlog_id, review_id)
    except (KeyError, ValueError):
        pass

    profile_id = f"defense-hitl-{uuid.uuid4().hex[:12]}"
    defense_profile = {
        "profile_id": profile_id,
        "name": f"Defense Profile: {attack} ({risk_level})",
        "source_review_id": review_id,
        "source_backlog_id": backlog_id,
        "attack_recipes": [
            {
                "attack": attack,
                "severity": severity,
                "strategy": "adversarial_training" if risk_level == "CRITICAL" else "augmentation_mix",
            }
        ],
        "mixing_ratio": 0.3 if risk_level == "CRITICAL" else 0.2,
        "created_at": _utcnow(),
        "auto_generated": True,
    }
    store.put_record("defense_profile", profile_id, defense_profile)

    return {
        "backlog_id": backlog_id,
        "backlog_name": backlog_name,
        "defense_profile_id": profile_id,
        "status": "created",
    }


def _trigger_deploy_block(
    review: dict[str, Any],
    store: Any,
) -> dict[str, Any]:
    """Record a deployment block gate for CI/CD integration."""
    gate_id = f"gate-block-{uuid.uuid4().hex[:12]}"
    gate_record = {
        "gate_id": gate_id,
        "gate_type": "DEPLOY_BLOCK",
        "source_review_id": review.get("review_id", ""),
        "run_id": review.get("run_id", ""),
        "attack": review.get("attack", ""),
        "severity": review.get("severity", 0),
        "risk_level": review.get("risk_level", "CRITICAL"),
        "reason": f"Reviewer blocked deployment due to {review.get('risk_level', 'CRITICAL')} risk in {review.get('attack', 'unknown')}",
        "created_at": _utcnow(),
        "resolved": False,
    }
    store.put_record("deploy_gate", gate_id, gate_record)
    return gate_record


def _trigger_odd_restriction(
    review: dict[str, Any],
    store: Any,
) -> dict[str, Any]:
    """Generate an Operational Design Domain constraint record."""
    constraint_id = f"odd-{uuid.uuid4().hex[:12]}"
    attack = review.get("attack", "unknown")
    severity = review.get("severity", 3)

    condition_map: dict[str, str] = {
        "fog": f"Sương mù vượt cấp {severity}",
        "depth_fog": f"Sương mù dày vượt cấp {severity}",
        "snow": f"Tuyết rơi vượt cấp {severity}",
        "rain": f"Mưa vượt cấp {severity}",
        "depth_rain": f"Mưa lớn vượt cấp {severity}",
        "frost": f"Sương giá vượt cấp {severity}",
        "brightness": f"Ánh sáng cực đoan vượt cấp {severity}",
        "contrast": f"Mất tương phản vượt cấp {severity}",
        "motion_blur": f"Rung lắc / nhòe chuyển động vượt cấp {severity}",
    }
    condition = condition_map.get(attack.lower(), f"Điều kiện {attack} vượt cấp {severity}")

    constraint_record = {
        "constraint_id": constraint_id,
        "constraint_type": "ODD_RESTRICTION",
        "source_review_id": review.get("review_id", ""),
        "run_id": review.get("run_id", ""),
        "condition": condition,
        "restricted_operation": f"Giảm tốc độ hoặc chuyển sang cảm biến dự phòng khi gặp: {condition}",
        "max_safe_severity": max(1, severity - 1),
        "attack": attack,
        "severity_threshold": severity,
        "affected_class": review.get("affected_class", "all"),
        "created_at": _utcnow(),
        "active": True,
    }
    store.put_record("odd_constraint", constraint_id, constraint_record)
    return constraint_record


def _trigger_relabel(
    review: dict[str, Any],
    store: Any,
) -> dict[str, Any]:
    """Create an annotation correction task for suspected labeling issues."""
    task_id = f"relabel-{uuid.uuid4().hex[:12]}"
    task_record = {
        "task_id": task_id,
        "task_type": "RELABEL_ANNOTATION",
        "source_review_id": review.get("review_id", ""),
        "run_id": review.get("run_id", ""),
        "dataset": review.get("dataset", ""),
        "attack": review.get("attack", ""),
        "severity": review.get("severity", 0),
        "reason": "Reviewer suspects labeling error, not attack-induced failure",
        "status": "PENDING",
        "created_at": _utcnow(),
    }
    store.put_record("annotation_task", task_id, task_record)
    return task_record


def _record_risk_acceptance(
    review: dict[str, Any],
    store: Any,
) -> dict[str, Any]:
    """Record an explicit risk acceptance with full audit trail."""
    entry_id = f"risk-accept-{uuid.uuid4().hex[:12]}"
    audit_entry = {
        "entry_id": entry_id,
        "entry_type": "RISK_ACCEPTANCE",
        "source_review_id": review.get("review_id", ""),
        "run_id": review.get("run_id", ""),
        "attack": review.get("attack", ""),
        "severity": review.get("severity", 0),
        "risk_level": review.get("risk_level", "LOW"),
        "degradation": review.get("degradation", 0.0),
        "decision_note": review.get("decision_note", ""),
        "resolved_by": review.get("resolved_by", "reviewer"),
        "created_at": _utcnow(),
        "immutable": True,
    }
    store.put_record("risk_acceptance_audit", entry_id, audit_entry)
    return audit_entry
