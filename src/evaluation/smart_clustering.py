"""Smart failure clustering using feature-based agglomerative grouping.

Extends the deterministic rule-based FailureGrouper with distance-based
clustering on a multi-dimensional feature vector extracted from review items.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from src.core.hashing import stable_digest

# ── Feature extraction constants ─────────────────────────────────────────

ATTACK_GROUP_INDEX: dict[str, int] = {
    "corruption": 0,
    "weather": 1,
    "occlusion": 2,
    "adversarial": 3,
    "patch": 4,
    "blackbox": 5,
}

CLASS_PRIORITY: dict[str, int] = {
    "Pedestrian": 3,
    "Cyclist": 3,
    "Car": 2,
    "Truck": 1,
    "Bus": 1,
}

RISK_LEVEL_INDEX: dict[str, int] = {
    "CRITICAL": 4,
    "HIGH": 3,
    "MEDIUM": 2,
    "LOW": 1,
    "AUTO_PASS": 0,
}

SIZE_BUCKET_INDEX: dict[str, int] = {
    "tiny": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
    "unknown": 1,
}

DISTANCE_BAND_INDEX: dict[str, int] = {
    "near": 0,
    "mid": 1,
    "far": 2,
    "unknown": 1,
}


@dataclass(frozen=True)
class SmartCluster:
    """A semantically meaningful group of related failure cases."""

    cluster_id: str
    label: str
    member_review_ids: tuple[str, ...]
    member_count: int
    representative_review_id: str
    mean_degradation: float
    max_degradation: float
    severity_distribution: dict[str, int]
    affected_classes: tuple[str, ...]
    dominant_attack_group: str
    dominant_risk_level: str
    recommended_action: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _extract_feature_vector(review: dict[str, Any]) -> tuple[float, ...]:
    """Extract an 8-dimensional normalized feature vector from a review item."""
    attack_name = review.get("attack", "")
    attack_group_raw = _infer_attack_group(attack_name)
    attack_group_val = ATTACK_GROUP_INDEX.get(attack_group_raw, 2) / max(1, len(ATTACK_GROUP_INDEX) - 1)

    severity_val = min(review.get("severity", 3), 5) / 5.0

    affected_class = review.get("affected_class", "Car")
    class_priority_val = CLASS_PRIORITY.get(affected_class, 1) / 3.0

    size_bucket = review.get("object_size_bucket", "medium")
    size_val = SIZE_BUCKET_INDEX.get(size_bucket, 1) / max(1, len(SIZE_BUCKET_INDEX) - 1)

    degradation_raw = review.get("degradation_percent", review.get("degradation", 0.0) * 100.0)
    degradation_val = min(max(degradation_raw, 0.0), 100.0) / 100.0

    distance_band = review.get("distance_band", "unknown")
    distance_val = DISTANCE_BAND_INDEX.get(distance_band, 1) / max(1, len(DISTANCE_BAND_INDEX) - 1)

    risk_level = review.get("risk_level", "MEDIUM")
    risk_val = RISK_LEVEL_INDEX.get(risk_level, 2) / max(1, len(RISK_LEVEL_INDEX) - 1)

    has_false_negative = 1.0 if review.get("has_false_negative", False) else 0.0

    return (
        attack_group_val,
        severity_val,
        class_priority_val,
        size_val,
        degradation_val,
        distance_val,
        risk_val,
        has_false_negative,
    )


def _infer_attack_group(attack_name: str) -> str:
    """Map an attack name to its group category."""
    corruption_attacks = {
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
    weather_attacks = {"depth_fog", "depth_rain", "depth_snow", "lidar_fog", "lidar_snow"}
    occlusion_attacks = {
        "random_erasing",
        "object_occlusion",
        "camera_dropout",
        "lidar_beam_drop",
        "lidar_sector_drop",
        "frame_freeze",
        "lidar_point_dropout",
        "sensor_fault",
    }
    adversarial_attacks = {"fgsm", "pgd", "mi_fgsm", "cw_l2", "tog", "dag", "sam2_pgd"}
    patch_attacks = {"dpatch", "thys_patch", "universal_patch", "printable_patch"}
    blackbox_attacks = {"square_attack", "random_noise_linf"}

    name_lower = attack_name.lower().replace("-", "_")
    if name_lower in corruption_attacks:
        return "corruption"
    if name_lower in weather_attacks:
        return "weather"
    if name_lower in occlusion_attacks:
        return "occlusion"
    if name_lower in adversarial_attacks:
        return "adversarial"
    if name_lower in patch_attacks:
        return "patch"
    if name_lower in blackbox_attacks:
        return "blackbox"
    return "corruption"


def _euclidean_distance(vec_a: tuple[float, ...], vec_b: tuple[float, ...]) -> float:
    """Compute Euclidean distance between two feature vectors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_a, vec_b)))


def _severity_band_label(severity: int) -> str:
    if severity <= 2:
        return "low"
    if severity == 3:
        return "medium"
    return "high"


def _auto_label_cluster(members: list[dict[str, Any]], dominant_group: str, dominant_risk: str) -> str:
    """Generate a human-readable label for a cluster."""
    count = len(members)
    class_counter = Counter(m.get("affected_class", m.get("model", "object")) for m in members)
    top_class = class_counter.most_common(1)[0][0] if class_counter else "object"

    attack_counter = Counter(m.get("attack", "unknown") for m in members)
    top_attack = attack_counter.most_common(1)[0][0] if attack_counter else "unknown"

    severity_counter = Counter(m.get("severity", 3) for m in members)
    top_severity = severity_counter.most_common(1)[0][0] if severity_counter else 3

    group_labels = {
        "corruption": "nhiễu ảnh",
        "weather": "thời tiết",
        "occlusion": "che khuất",
        "adversarial": "tấn công gradient",
        "patch": "adversarial patch",
        "blackbox": "black-box",
    }
    group_label = group_labels.get(dominant_group, dominant_group)

    risk_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "AUTO_PASS": "🟢"}
    emoji = risk_emoji.get(dominant_risk, "⚪")

    return f"{emoji} {count} ca: {top_class} bị {group_label} ({top_attack} cấp {top_severity})"


def _select_representative(members: list[dict[str, Any]]) -> str:
    """Select the review closest to the median degradation as cluster representative."""
    if not members:
        return ""
    degradations = [
        (m.get("review_id", m.get("id", "")), m.get("degradation_percent", m.get("degradation", 0.0) * 100.0))
        for m in members
    ]
    degradations.sort(key=lambda pair: pair[1])
    median_idx = len(degradations) // 2
    return degradations[median_idx][0]


def _recommend_action(dominant_risk: str, dominant_group: str) -> str:
    """Suggest a downstream action based on cluster characteristics."""
    if dominant_risk == "CRITICAL":
        return "BLOCK_DEPLOY"
    if dominant_risk == "HIGH":
        return "REQUEST_RETRAIN"
    if dominant_group in ("weather", "occlusion") and dominant_risk == "MEDIUM":
        return "RESTRICT_ODD"
    if dominant_risk == "MEDIUM":
        return "REQUEST_RETRAIN"
    return "ACCEPT_RISK"


class SmartFailureClusterer:
    """Groups review items into semantically meaningful clusters.

    Uses agglomerative clustering on an 8-dimensional feature vector.
    Falls back to single-linkage merging when scipy is unavailable.

    Args:
        max_clusters: Upper bound on the number of clusters to produce.
        distance_threshold: Maximum inter-cluster distance for merging.
    """

    version = "2.0.0"

    def __init__(self, *, max_clusters: int = 8, distance_threshold: float = 0.35) -> None:
        self._max_clusters = max_clusters
        self._distance_threshold = distance_threshold

    def cluster(self, reviews: list[dict[str, Any]]) -> tuple[SmartCluster, ...]:
        """Group review items and return semantically labelled clusters."""
        if not reviews:
            return ()

        if len(reviews) == 1:
            return (self._singleton_cluster(reviews[0]),)

        feature_vectors = [_extract_feature_vector(r) for r in reviews]
        labels = self._agglomerative_cluster(feature_vectors)

        groups: dict[int, list[dict[str, Any]]] = {}
        for label, review in zip(labels, reviews):
            groups.setdefault(label, []).append(review)

        clusters: list[SmartCluster] = []
        for group_label in sorted(groups.keys()):
            members = groups[group_label]
            clusters.append(self._build_cluster(members))

        clusters.sort(key=lambda c: RISK_LEVEL_INDEX.get(c.dominant_risk_level, 0), reverse=True)
        return tuple(clusters)

    def _agglomerative_cluster(self, vectors: list[tuple[float, ...]]) -> list[int]:
        """Single-linkage agglomerative clustering without external dependencies."""
        n = len(vectors)
        cluster_assignments = list(range(n))

        distances: list[tuple[float, int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = _euclidean_distance(vectors[i], vectors[j])
                distances.append((dist, i, j))
        distances.sort(key=lambda t: t[0])

        active_clusters = set(range(n))

        for dist, idx_a, idx_b in distances:
            if dist > self._distance_threshold:
                break

            cluster_a = cluster_assignments[idx_a]
            cluster_b = cluster_assignments[idx_b]
            if cluster_a == cluster_b:
                continue

            unique_active = len(active_clusters)
            if unique_active <= self._max_clusters:
                current_unique = len(set(cluster_assignments))
                if current_unique <= self._max_clusters:
                    break

            merge_target = min(cluster_a, cluster_b)
            merge_source = max(cluster_a, cluster_b)
            for k in range(n):
                if cluster_assignments[k] == merge_source:
                    cluster_assignments[k] = merge_target
            active_clusters.discard(merge_source)

        label_remap: dict[int, int] = {}
        counter = 0
        for assignment in cluster_assignments:
            if assignment not in label_remap:
                label_remap[assignment] = counter
                counter += 1

        return [label_remap[a] for a in cluster_assignments]

    def _build_cluster(self, members: list[dict[str, Any]]) -> SmartCluster:
        """Construct a SmartCluster from a group of review items."""
        review_ids = tuple(sorted(m.get("review_id", m.get("id", "")) for m in members))

        degradations = [m.get("degradation_percent", m.get("degradation", 0.0) * 100.0) for m in members]
        mean_deg = sum(degradations) / max(len(degradations), 1)
        max_deg = max(degradations) if degradations else 0.0

        severity_dist: dict[str, int] = {}
        for m in members:
            band = _severity_band_label(m.get("severity", 3))
            severity_dist[band] = severity_dist.get(band, 0) + 1

        class_counter = Counter(m.get("affected_class", m.get("model", "unknown")) for m in members)
        affected = tuple(cls for cls, _ in class_counter.most_common())

        group_counter = Counter(_infer_attack_group(m.get("attack", "")) for m in members)
        dominant_group = group_counter.most_common(1)[0][0] if group_counter else "corruption"

        risk_counter = Counter(m.get("risk_level", "MEDIUM") for m in members)
        dominant_risk = max(risk_counter, key=lambda r: RISK_LEVEL_INDEX.get(r, 0))

        representative_id = _select_representative(members)
        label = _auto_label_cluster(members, dominant_group, dominant_risk)
        action = _recommend_action(dominant_risk, dominant_group)

        cluster_id = f"smart-cluster-{stable_digest({'ids': review_ids}, length=16)}"

        return SmartCluster(
            cluster_id=cluster_id,
            label=label,
            member_review_ids=review_ids,
            member_count=len(members),
            representative_review_id=representative_id,
            mean_degradation=round(mean_deg, 2),
            max_degradation=round(max_deg, 2),
            severity_distribution=severity_dist,
            affected_classes=affected,
            dominant_attack_group=dominant_group,
            dominant_risk_level=dominant_risk,
            recommended_action=action,
        )

    def _singleton_cluster(self, review: dict[str, Any]) -> SmartCluster:
        """Create a single-member cluster."""
        return self._build_cluster([review])
