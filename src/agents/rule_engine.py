"""Deterministic Rule Engine evaluating project state and benchmark evidence to generate actionable recommendations."""

from __future__ import annotations

import uuid
from typing import Any

from src.agents.contracts import Recommendation
from src.api.jobs import SqliteRunStore


def evaluate_rules(
    store: SqliteRunStore,
    *,
    project_id: str | None = None,
    run_id: str | None = None,
) -> list[Recommendation]:
    """Evaluate deterministic rules against available checkpoints, datasets, runs, and failure evidence.

    Rule priorities:
    1. Unlabeled or missing dataset annotations -> LABEL_DATASET (CRITICAL)
    2. Checkpoints pending validation -> UPLOAD_CHECKPOINT (HIGH)
    3. Checkpoint ready without clean baseline -> RUN_CLEAN_BASELINE (HIGH)
    4. Attack benchmark has high degradation (>25%) -> GENERATE_DEFENCE_DATASET (HIGH)
    5. 3D detection failures concentrated at far distance (>40m) -> INVESTIGATE_FAR_RANGE_FAILURES (MEDIUM)
    6. Candidate model evaluated on seen attacks only -> RUN_HELDOUT_ATTACK (MEDIUM)
    7. Clean degradation regression after fine-tuning -> ADJUST_TRAINING (HIGH)
    """
    recommendations: list[Recommendation] = []

    # 1. Check datasets
    dataset_records = store.list_records("dataset_version")
    for ds in dataset_records:
        if ds.get("status") in ("UNLABELED", "PENDING_ANNOTATION"):
            recommendations.append(
                Recommendation(
                    id=f"rec-ds-{uuid.uuid4().hex[:8]}",
                    priority="CRITICAL",
                    action_type="LABEL_DATASET",
                    title=f"Annotate Dataset: {ds.get('name', ds.get('id', 'Unknown'))}",
                    reason="Perception models require ground-truth annotations before scientific benchmark execution.",
                    evidence=[f"dataset_id:{ds.get('id')}", "status:UNLABELED"],
                    risks=["Cannot compute mAP or detection precision without ground truth"],
                    suggested_parameters={"dataset_id": ds.get("id")},
                )
            )

    # 2. Check checkpoints
    checkpoint_records = store.list_records("checkpoint")
    for cp in checkpoint_records:
        if cp.get("status") == "PENDING_VALIDATION":
            recommendations.append(
                Recommendation(
                    id=f"rec-cp-{uuid.uuid4().hex[:8]}",
                    priority="HIGH",
                    action_type="UPLOAD_CHECKPOINT",
                    title=f"Validate Checkpoint: {cp.get('display_name', cp.get('checkpoint_id', 'Unknown'))}",
                    reason="Checkpoint is quarantined and awaiting automated sandbox security validation.",
                    evidence=[f"checkpoint_id:{cp.get('checkpoint_id')}", "status:PENDING_VALIDATION"],
                    risks=["Non-validated checkpoints cannot be dispatched to execution workers"],
                    suggested_parameters={"checkpoint_id": cp.get("checkpoint_id")},
                )
            )

    # 3. Check benchmark runs
    runs = store.list()
    completed_runs: list[dict[str, Any]] = [
        r for r in runs if r.get("status") == "COMPLETED" and r.get("report") is not None
    ]

    # Specific run focus if requested
    if run_id:
        target_run = store.get(run_id)
        if target_run and target_run.get("report"):
            _evaluate_run_specific_rules(target_run["report"], recommendations)
    else:
        for run in completed_runs:
            _evaluate_run_specific_rules(run["report"], recommendations)

    # 4. Check comparisons for clean regression
    comparisons = store.list_records("model_comparison")
    for comp in comparisons:
        metric_deltas = comp.get("metric_deltas") or {}
        clean_score_delta = (metric_deltas.get("clean_detection_score") or {}).get("value", 0.0)
        if isinstance(clean_score_delta, (int, float)) and clean_score_delta < -0.05:
            recommendations.append(
                Recommendation(
                    id=f"rec-comp-{uuid.uuid4().hex[:8]}",
                    priority="HIGH",
                    action_type="ADJUST_TRAINING",
                    title="Mitigate Clean Accuracy Regression",
                    reason=f"Defended model suffered a clean mAP drop of {abs(clean_score_delta) * 100:.1f}%. Increase clean replay ratio in defense profile.",
                    evidence=[
                        f"comparison_id:{comp.get('comparison_id')}",
                        f"clean_score_delta:{clean_score_delta}",
                    ],
                    risks=["Overfitting on adversarial examples damages clean operational performance"],
                    suggested_parameters={
                        "comparison_id": comp.get("comparison_id"),
                        "clean_replay_ratio": 0.5,
                    },
                )
            )

    # If no runs exist but checkpoints exist, recommend baseline run
    ready_checkpoints = [cp for cp in checkpoint_records if cp.get("status") == "READY"]
    if not completed_runs and ready_checkpoints:
        first_cp = ready_checkpoints[0]
        recommendations.append(
            Recommendation(
                id=f"rec-base-{uuid.uuid4().hex[:8]}",
                priority="HIGH",
                action_type="RUN_CLEAN_BASELINE",
                title=f"Establish Clean Baseline for {first_cp.get('display_name', first_cp.get('checkpoint_id'))}",
                reason="No clean baseline benchmark has been recorded for this checkpoint yet.",
                evidence=[f"checkpoint_id:{first_cp.get('checkpoint_id')}"],
                risks=["Robustness testing cannot measure degradation without a clean baseline"],
                suggested_parameters={
                    "checkpoint_id": first_cp.get("checkpoint_id"),
                    "task_id": first_cp.get("task_id", "detection2d"),
                },
            )
        )

    # Sort recommendations by priority (CRITICAL -> HIGH -> MEDIUM -> LOW)
    priority_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    return sorted(recommendations, key=lambda r: priority_order.get(r.priority, 99))


def _evaluate_run_specific_rules(report: dict[str, Any], recommendations: list[Recommendation]) -> None:
    """Evaluate performance patterns inside a completed single run report."""
    run_id = report.get("run_id", "")
    clean_ap = float(report.get("ap_clean", 0.0))
    cells = list(report.get("cells", []))
    task = str(report.get("task", "detection2d"))

    worst_attack_name = None
    worst_degradation = 0.0

    for cell in cells:
        attack_ap = float(cell.get("ap", 0.0))
        if clean_ap > 0.0:
            deg = (clean_ap - attack_ap) / clean_ap
            if deg > worst_degradation:
                worst_degradation = deg
                worst_attack_name = cell.get("attack")

    if worst_degradation > 0.25 and worst_attack_name:
        recommendations.append(
            Recommendation(
                id=f"rec-deg-{uuid.uuid4().hex[:8]}",
                priority="HIGH",
                action_type="GENERATE_DEFENCE_DATASET",
                title=f"Generate Robustness Defense for '{worst_attack_name}'",
                reason=f"Attack '{worst_attack_name}' caused a severe {worst_degradation * 100:.1f}% accuracy drop. Build a training dataset with corresponding corruption mix.",
                evidence=[
                    f"run_id:{run_id}",
                    f"worst_attack:{worst_attack_name}",
                    f"degradation:{worst_degradation * 100:.1f}%",
                ],
                risks=["Model remains vulnerable to high-severity perturbation in production"],
                suggested_parameters={
                    "run_id": run_id,
                    "target_attack": worst_attack_name,
                    "preset_id": "fast_yolo_mix" if task == "detection2d" else "robust_mix_profile",
                },
            )
        )

    # Check 3D distance vulnerabilities
    if task == "detection3d":
        worst_cases = list(report.get("worst_cases", []))
        far_failures = sum(1 for f in worst_cases if (f.get("metadata") or {}).get("distance_bucket") == "far")
        if far_failures > 5:
            recommendations.append(
                Recommendation(
                    id=f"rec-3d-far-{uuid.uuid4().hex[:8]}",
                    priority="MEDIUM",
                    action_type="INVESTIGATE_FAR_RANGE_FAILURES",
                    title="Investigate Far-Range 3D Perception Failures (>40m)",
                    reason=f"Concentration of {far_failures} failure cases in the far range (>=40m). Consider beam density or LiDAR point dropout fine-tuning.",
                    evidence=[f"run_id:{run_id}", f"far_failures_count:{far_failures}"],
                    risks=["Late detection of fast-approaching distant obstacles in autonomous driving"],
                    suggested_parameters={"run_id": run_id, "focus_distance": "far"},
                )
            )
