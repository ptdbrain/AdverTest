"""Risk assessment engine for Human-in-the-Loop review decisions.

Classifies failures into risk levels based on object class priority,
proximity, degradation magnitude, and attack type, then recommends
standardized decisions with justifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── Risk Classification Constants ────────────────────────────────────────

SAFETY_CRITICAL_CLASSES = frozenset({"Pedestrian", "Cyclist", "Person", "Rider"})
HIGH_PRIORITY_CLASSES = frozenset({"Car", "Truck", "Bus", "Van", "Motorcycle"})

DECISION_LABELS: dict[str, str] = {
    "BLOCK_DEPLOY": "Chặn triển khai — Bắt buộc retrain",
    "REQUEST_RETRAIN": "Yêu cầu tôi luyện đối kháng",
    "RESTRICT_ODD": "Giới hạn vùng vận hành (ODD)",
    "RELABEL_DATA": "Chuyển sang sửa nhãn dữ liệu",
    "ACCEPT_RISK": "Chấp nhận rủi ro có giải trình",
}

RISK_LEVEL_ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "AUTO_PASS")


@dataclass(frozen=True)
class RiskAssessment:
    """Structured risk evaluation with recommended action."""

    risk_level: str
    risk_category: str
    recommended_decision: str
    recommended_decision_label: str
    justification: str
    downstream_actions: tuple[str, ...]
    confidence: float
    contributing_factors: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskSummary:
    """Aggregated risk profile for a session or run."""

    total_reviews: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    auto_pass_count: int
    dominant_risk_level: str
    overall_recommendation: str
    cluster_count: int
    top_affected_classes: tuple[str, ...]
    top_attack_groups: tuple[str, ...]


def classify_risk_level(
    degradation_percent: float,
    affected_class: str = "Car",
    distance_meters: float | None = None,
    has_false_negative: bool = False,
    attack_group: str = "corruption",
    severity: int = 3,
) -> str:
    """Classify a failure into one of 5 risk levels.

    Args:
        degradation_percent: Performance degradation as percentage (0-100).
        affected_class: Object class affected by the failure.
        distance_meters: Distance to the affected object in meters (if known).
        has_false_negative: Whether the failure involves a missed detection.
        attack_group: Category of attack that caused the failure.
        severity: Attack severity level (1-5).

    Returns:
        Risk level string: CRITICAL, HIGH, MEDIUM, LOW, or AUTO_PASS.
    """
    is_safety_critical = affected_class in SAFETY_CRITICAL_CLASSES
    is_near = distance_meters is not None and distance_meters < 20.0
    is_very_near = distance_meters is not None and distance_meters < 10.0

    if is_safety_critical and is_very_near and has_false_negative:
        return "CRITICAL"
    if is_safety_critical and is_near and degradation_percent >= 50.0:
        return "CRITICAL"
    if degradation_percent >= 80.0:
        return "CRITICAL"
    if is_safety_critical and has_false_negative:
        return "HIGH"
    if degradation_percent >= 50.0:
        return "HIGH"
    if is_near and degradation_percent >= 30.0:
        return "HIGH"
    if degradation_percent >= 30.0:
        return "MEDIUM"
    if degradation_percent >= 15.0:
        return "LOW"
    return "AUTO_PASS"


def assess_risk(
    review: dict[str, Any],
    report: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> RiskAssessment:
    """Evaluate a review item and produce a structured risk assessment.

    Args:
        review: Review item dict with attack, severity, degradation, etc.
        report: Optional run report for additional context.
        config: Optional run config for dataset/model context.

    Returns:
        RiskAssessment with recommended decision and justification.
    """
    degradation_pct = review.get(
        "degradation_percent",
        float(review.get("degradation_ratio", review.get("degradation", 0.0))) * 100.0,
    )
    affected_class = review.get("affected_class", review.get("model", "Car"))
    distance = review.get("distance_meters")
    has_fn = review.get("has_false_negative", False)
    attack = review.get("attack", "unknown")
    severity = review.get("severity", 3)
    attack_group = _infer_group(attack)

    risk_level = classify_risk_level(
        degradation_percent=degradation_pct,
        affected_class=affected_class,
        distance_meters=distance,
        has_false_negative=has_fn,
        attack_group=attack_group,
        severity=severity,
    )

    risk_category = _categorize_risk(affected_class, distance, attack_group)
    decision, actions = _recommend_decision(risk_level, attack_group, affected_class)
    justification = _generate_justification(
        risk_level,
        degradation_pct,
        affected_class,
        distance,
        attack,
        severity,
        has_fn,
    )
    factors = _contributing_factors(degradation_pct, affected_class, distance, has_fn, severity)
    confidence = _compute_confidence(degradation_pct, distance)

    return RiskAssessment(
        risk_level=risk_level,
        risk_category=risk_category,
        recommended_decision=decision,
        recommended_decision_label=DECISION_LABELS.get(decision, decision),
        justification=justification,
        downstream_actions=actions,
        confidence=confidence,
        contributing_factors=factors,
    )


def generate_risk_summary(reviews: list[dict[str, Any]]) -> RiskSummary:
    """Aggregate risk assessments across all reviews in a session."""
    if not reviews:
        return RiskSummary(
            total_reviews=0,
            critical_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            auto_pass_count=0,
            dominant_risk_level="AUTO_PASS",
            overall_recommendation="Không có ca lỗi cần xem xét",
            cluster_count=0,
            top_affected_classes=(),
            top_attack_groups=(),
        )

    level_counts = {level: 0 for level in RISK_LEVEL_ORDER}
    class_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}

    for review in reviews:
        level = review.get("risk_level", "MEDIUM")
        level_counts[level] = level_counts.get(level, 0) + 1

        cls = review.get("affected_class", review.get("model", "unknown"))
        class_counts[cls] = class_counts.get(cls, 0) + 1

        grp = _infer_group(review.get("attack", ""))
        group_counts[grp] = group_counts.get(grp, 0) + 1

    dominant = next(
        (level for level in RISK_LEVEL_ORDER if level_counts.get(level, 0) > 0),
        "AUTO_PASS",
    )

    recommendation = _overall_recommendation(dominant, level_counts)

    top_classes = tuple(cls for cls, _ in sorted(class_counts.items(), key=lambda p: p[1], reverse=True)[:3])
    top_groups = tuple(grp for grp, _ in sorted(group_counts.items(), key=lambda p: p[1], reverse=True)[:3])

    return RiskSummary(
        total_reviews=len(reviews),
        critical_count=level_counts.get("CRITICAL", 0),
        high_count=level_counts.get("HIGH", 0),
        medium_count=level_counts.get("MEDIUM", 0),
        low_count=level_counts.get("LOW", 0),
        auto_pass_count=level_counts.get("AUTO_PASS", 0),
        dominant_risk_level=dominant,
        overall_recommendation=recommendation,
        cluster_count=0,
        top_affected_classes=top_classes,
        top_attack_groups=top_groups,
    )


# ── Private helpers ──────────────────────────────────────────────────────


def _infer_group(attack_name: str) -> str:
    """Map attack name to its group category."""
    name = attack_name.lower().replace("-", "_")
    corruption = {
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "speckle_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "gaussian_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "spatter",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
        "saturate",
    }
    weather = {"depth_fog", "depth_rain", "depth_snow", "lidar_fog", "lidar_snow"}
    occlusion = {
        "random_erasing",
        "object_occlusion",
        "camera_dropout",
        "lidar_beam_drop",
        "lidar_sector_drop",
        "frame_freeze",
        "lidar_point_dropout",
        "sensor_fault",
    }
    adversarial = {"fgsm", "pgd", "mi_fgsm", "cw_l2", "tog", "dag", "sam2_pgd"}
    patch = {"dpatch", "thys_patch", "universal_patch", "printable_patch"}
    blackbox = {"square_attack", "random_noise_linf"}

    if name in corruption:
        return "corruption"
    if name in weather:
        return "weather"
    if name in occlusion:
        return "occlusion"
    if name in adversarial:
        return "adversarial"
    if name in patch:
        return "patch"
    if name in blackbox:
        return "blackbox"
    return "corruption"


def _categorize_risk(affected_class: str, distance: float | None, attack_group: str) -> str:
    is_safety = affected_class in SAFETY_CRITICAL_CLASSES
    is_near = distance is not None and distance < 20.0

    if is_safety and is_near:
        return "safety_critical_near"
    if is_safety:
        return "safety_critical_far"
    if is_near:
        return "high_priority_near"
    return f"{attack_group}_degradation"


def _recommend_decision(
    risk_level: str,
    attack_group: str,
    affected_class: str,
) -> tuple[str, tuple[str, ...]]:
    if risk_level == "CRITICAL":
        return "BLOCK_DEPLOY", (
            "Tạo Retraining Backlog",
            "Nạp vào Defense Profile adversarial_hardening",
            "Gắn cờ chặn triển khai trong CI/CD Gate",
        )
    if risk_level == "HIGH":
        if attack_group in ("weather", "occlusion"):
            return "REQUEST_RETRAIN", (
                "Tạo Retraining Backlog",
                f"Sinh thêm dữ liệu augmentation nhóm {attack_group}",
                "Nạp vào Defense Profile",
            )
        return "REQUEST_RETRAIN", (
            "Tạo Retraining Backlog",
            "Nạp vào Defense Profile",
        )
    if risk_level == "MEDIUM":
        if attack_group in ("weather", "occlusion"):
            return "RESTRICT_ODD", (
                f"Giới hạn vận hành khi {attack_group} vượt cấp 3",
                "Bổ sung cảnh báo vận hành",
            )
        return "REQUEST_RETRAIN", ("Tạo Retraining Backlog",)
    if risk_level == "LOW":
        return "ACCEPT_RISK", (
            "Ghi nhận vào Audit Log",
            "Theo dõi xu hướng trong các phiên tiếp theo",
        )
    return "ACCEPT_RISK", ("Ghi nhận vào Audit Log",)


def _generate_justification(
    risk_level: str,
    degradation_pct: float,
    affected_class: str,
    distance: float | None,
    attack: str,
    severity: int,
    has_fn: bool,
) -> str:
    parts: list[str] = []

    parts.append(f"Suy giảm {degradation_pct:.1f}% hiệu suất")

    if affected_class in SAFETY_CRITICAL_CLASSES:
        parts.append(f"trên đối tượng an toàn quan trọng ({affected_class})")
    else:
        parts.append(f"trên đối tượng {affected_class}")

    if distance is not None:
        parts.append(f"ở cự ly {distance:.0f}m")

    parts.append(f"do {attack} cấp {severity}")

    if has_fn:
        parts.append("— có phát hiện sai âm (False Negative)")

    risk_desc = {
        "CRITICAL": "Mức độ rủi ro tính mạng cực kỳ cao.",
        "HIGH": "Mức độ rủi ro cao, cần hành động khắc phục.",
        "MEDIUM": "Mức độ rủi ro trung bình, cần đánh giá thêm.",
        "LOW": "Mức độ rủi ro thấp, có thể chấp nhận.",
        "AUTO_PASS": "Mức độ rủi ro không đáng kể.",
    }
    suffix = risk_desc.get(risk_level, "")

    return f"{' '.join(parts)}. {suffix}"


def _contributing_factors(
    degradation_pct: float,
    affected_class: str,
    distance: float | None,
    has_fn: bool,
    severity: int,
) -> tuple[str, ...]:
    factors: list[str] = []
    if degradation_pct >= 50.0:
        factors.append(f"Suy giảm nghiêm trọng ({degradation_pct:.1f}%)")
    if affected_class in SAFETY_CRITICAL_CLASSES:
        factors.append(f"Đối tượng an toàn quan trọng ({affected_class})")
    if distance is not None and distance < 20.0:
        factors.append(f"Cự ly gần ({distance:.0f}m)")
    if has_fn:
        factors.append("Có False Negative (mất phát hiện)")
    if severity >= 4:
        factors.append(f"Mức tấn công cao (cấp {severity})")
    return tuple(factors)


def _compute_confidence(degradation_pct: float, distance: float | None) -> float:
    confidence = 0.7
    if degradation_pct >= 50.0:
        confidence += 0.15
    if distance is not None:
        confidence += 0.1
    return min(confidence, 1.0)


def _overall_recommendation(dominant: str, counts: dict[str, int]) -> str:
    critical = counts.get("CRITICAL", 0)
    high = counts.get("HIGH", 0)

    if critical > 0:
        return f"⛔ {critical} ca rủi ro tính mạng — BẮT BUỘC retrain và chặn triển khai trước khi tiếp tục."
    if high > 0:
        return f"⚠️ {high} ca rủi ro cao — Khuyến nghị tạo Defense Profile và tôi luyện đối kháng."
    return "✅ Không có ca rủi ro cao. Có thể tiếp tục với giám sát thường xuyên."


# ── Risk Rubric Table (for API/UI rendering) ─────────────────────────────

RISK_RUBRIC_TABLE: tuple[dict[str, Any], ...] = (
    {
        "risk_level": "CRITICAL",
        "color": "red",
        "icon": "shield-alert",
        "criteria": "Mất Pedestrian/Cyclist cự ly <20m HOẶC degradation ≥80%",
        "required_decision": "BLOCK_DEPLOY",
        "downstream": "Tự động tạo Retraining Backlog + Defense Profile + CI/CD Gate block",
    },
    {
        "risk_level": "HIGH",
        "color": "orange",
        "icon": "alert-triangle",
        "criteria": "Degradation ≥50% HOẶC mất đối tượng quan trọng",
        "required_decision": "REQUEST_RETRAIN",
        "downstream": "Tạo Retraining Backlog + Defense Profile",
    },
    {
        "risk_level": "MEDIUM",
        "color": "amber",
        "icon": "alert-circle",
        "criteria": "Degradation 30-50%",
        "required_decision": "REQUEST_RETRAIN hoặc RESTRICT_ODD",
        "downstream": "Tùy ngữ cảnh: retrain hoặc giới hạn ODD",
    },
    {
        "risk_level": "LOW",
        "color": "blue",
        "icon": "info",
        "criteria": "Degradation 15-30%",
        "required_decision": "ACCEPT_RISK",
        "downstream": "Ghi audit log, theo dõi xu hướng",
    },
    {
        "risk_level": "AUTO_PASS",
        "color": "green",
        "icon": "check-circle",
        "criteria": "Degradation <15%",
        "required_decision": "Tự động thông qua",
        "downstream": "Ghi audit log tự động, không cần reviewer",
    },
)
