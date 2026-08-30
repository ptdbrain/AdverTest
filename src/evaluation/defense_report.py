"""Build one scientific, paired before/after Defence report from immutable runs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from src.api.schemas.defense import (
    DecisionOut,
    DefenseReportOut,
    EligibilityOut,
    FailureSummaryOut,
    MetricDeltaOut,
    RecoveryOut,
)
from src.core.hashing import stable_digest
from src.evaluation.metric_catalog import (
    MetricDefinition,
    build_pairing_signature,
    differing_signature_keys,
    metric_definitions_for,
    required_protocol_fields,
)
from src.evaluation.recovery_metrics import UndefinedMetric, recovery_rate


def build_defense_report(
    *,
    project_id: str,
    baseline_run: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
) -> DefenseReportOut:
    """Return a report that declines conclusions whenever evidence is incomplete."""
    baseline = _mapping(baseline_run.get("report"))
    candidate = _mapping(candidate_run.get("report"))
    baseline_signature = build_pairing_signature(baseline)
    candidate_signature = build_pairing_signature(candidate)
    incompatibilities = differing_signature_keys(baseline_signature, candidate_signature)
    definitions = metric_definitions_for(
        _text(baseline_signature.get("task_id")),
        _text(baseline_signature.get("dataset_version_id")),
        metric_protocol=_text(baseline_signature.get("metric_protocol")),
    )
    reasons = _rejection_reasons(baseline_run, candidate_run, baseline, candidate, definitions, incompatibilities)
    eligibility = EligibilityOut(
        status="ELIGIBLE" if not reasons else "NOT_ELIGIBLE",
        reasons=tuple(reasons),
        next_action=(
            "Review paired evidence and make a human promotion decision."
            if not reasons
            else "Resolve the listed evidence and protocol issues, then rerun the locked benchmark."
        ),
    )
    metric_deltas = _metric_deltas(baseline, candidate, definitions) if not reasons else ()
    recovery = _recovery(baseline, candidate, definitions) if not reasons else RecoveryOut(reason="NOT_ELIGIBLE")
    failures = _failure_summary(baseline, candidate) if not reasons else FailureSummaryOut(
        baseline_count=0,
        candidate_count=0,
        recovered_count=0,
        regressed_count=0,
    )
    decision = DecisionOut(
        status="READY_FOR_REVIEW" if not reasons else "NOT_ELIGIBLE",
        summary=(
            "Evidence is paired and complete; a human reviewer may assess deployment trade-offs."
            if not reasons
            else "No benchmark conclusion or promotion is allowed because evidence is incomplete or unpaired."
        ),
        next_action=eligibility.next_action,
    )
    comparison_id = f"comparison-{stable_digest({'project_id': project_id, 'baseline': baseline_run.get('run_id'), 'candidate': candidate_run.get('run_id')}, length=20)}"
    return DefenseReportOut(
        comparison_id=comparison_id,
        project_id=project_id,
        baseline_run_id=_text(baseline_run.get("run_id")),
        candidate_run_id=_text(candidate_run.get("run_id")),
        eligibility=eligibility,
        baseline_signature=baseline_signature,
        candidate_signature=candidate_signature,
        incompatibilities=tuple(incompatibilities),
        metric_deltas=metric_deltas,
        recovery=recovery,
        failures=failures,
        decision=decision,
    )


def _rejection_reasons(
    baseline_run: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    definitions: Sequence[MetricDefinition],
    incompatibilities: Sequence[str],
) -> list[str]:
    reasons: list[str] = []
    if baseline_run.get("status") != "COMPLETED" or candidate_run.get("status") != "COMPLETED":
        reasons.append("RUN_NOT_COMPLETED")
    if not baseline or not candidate:
        reasons.append("REPORT_MISSING")
    for report in (baseline, candidate):
        provenance = _mapping(report.get("provenance"))
        task = _text(provenance.get("task_id") or _mapping(provenance.get("run_config")).get("task_id"))
        dataset = _text(provenance.get("dataset_version_id") or report.get("dataset"))
        for field in sorted(required_protocol_fields(task, dataset)):
            if not provenance.get(field):
                reasons.append(f"{field.upper()}_MISSING")
        if report.get("simulation_only") is not False:
            reasons.append(
                "SIMULATION_ONLY" if report.get("simulation_only") is True else "SIMULATION_STATUS_UNDECLARED"
            )
        if _text(report.get("source_kind") or provenance.get("source_kind")).lower() == "demo":
            reasons.append("DEMO_SOURCE")
    if not definitions:
        reasons.append("UNSUPPORTED_METRIC_PROTOCOL")
    if incompatibilities:
        reasons.append("UNPAIRED_PROTOCOL")
    if definitions and not _all_metric_values_finite(baseline, candidate, definitions):
        reasons.append("METRIC_VALUES_MISSING_OR_NONFINITE")
    return sorted(set(reasons))


def _metric_deltas(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], definitions: Sequence[MetricDefinition]
) -> tuple[MetricDeltaOut, ...]:
    items: list[MetricDeltaOut] = []
    for definition in definitions:
        base_clean = _clean_metric(baseline, definition.key)
        cand_clean = _clean_metric(candidate, definition.key)
        base_attacked = _mean_attacked_metric(baseline, definition.key)
        cand_attacked = _mean_attacked_metric(candidate, definition.key)
        items.append(
            MetricDeltaOut(
                key=definition.key,
                label=definition.label,
                unit=definition.unit,
                higher_is_better=definition.higher_is_better,
                baseline_clean=base_clean,
                candidate_clean=cand_clean,
                baseline_attacked=base_attacked,
                candidate_attacked=cand_attacked,
                clean_delta=cand_clean - base_clean,
                attacked_delta=cand_attacked - base_attacked,
            )
        )
    return tuple(items)


def _recovery(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], definitions: Sequence[MetricDefinition]
) -> RecoveryOut:
    if not definitions:
        return RecoveryOut(reason="UNSUPPORTED_METRIC_PROTOCOL")
    definition = definitions[0]
    result = recovery_rate(
        _clean_metric(baseline, definition.key),
        _mean_attacked_metric(baseline, definition.key),
        _mean_attacked_metric(candidate, definition.key),
        higher_is_better=definition.higher_is_better,
    )
    if isinstance(result, UndefinedMetric):
        return RecoveryOut(metric_key=definition.key, reason=result.reason)
    return RecoveryOut(
        metric_key=definition.key,
        value=result.value,
        percent_value=result.percent_value,
        unit="ratio",
        unbounded=True,
    )


def _failure_summary(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> FailureSummaryOut:
    base = _explicit_failure_cases(baseline)
    cand = _explicit_failure_cases(candidate)
    recovered = sorted(set(base) - set(cand))
    regressed = sorted(set(cand) - set(base))
    transitions = tuple(
        [{"case_id": case_id, "transition": "RECOVERED"} for case_id in recovered]
        + [{"case_id": case_id, "transition": "REGRESSED"} for case_id in regressed]
    )
    return FailureSummaryOut(
        baseline_count=len(base),
        candidate_count=len(cand),
        recovered_count=len(recovered),
        regressed_count=len(regressed),
        transitions=transitions,
    )


def _explicit_failure_cases(report: Mapping[str, Any]) -> set[str]:
    cases = report.get("worst_cases") if isinstance(report.get("worst_cases"), list) else []
    return {
        _text(item.get("case_id") or item.get("failure_id") or item.get("sample_id"))
        for item in cases
        if isinstance(item, Mapping)
        and item.get("failed") is True
        and (item.get("case_id") or item.get("failure_id") or item.get("sample_id"))
    }


def _all_metric_values_finite(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any], definitions: Sequence[MetricDefinition]
) -> bool:
    return all(
        math.isfinite(value)
        for report in (baseline, candidate)
        for definition in definitions
        for value in (_clean_metric(report, definition.key), _mean_attacked_metric(report, definition.key))
    )


def _clean_metric(report: Mapping[str, Any], key: str) -> float:
    metrics = _mapping(_mapping(report.get("metrics")).get("clean"))
    return _finite_metric(metrics.get(key), key)


def _mean_attacked_metric(report: Mapping[str, Any], key: str) -> float:
    values = [
        _finite_metric(_mapping(cell.get("metrics")).get(key), key)
        for cell in report.get("cells", [])
        if isinstance(cell, Mapping)
    ]
    if not values:
        return math.nan
    return sum(values) / len(values)


def _finite_metric(value: Any, key: str) -> float:
    if isinstance(value, bool):
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    return str(value) if value is not None else ""
