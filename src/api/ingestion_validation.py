"""Central mode-specific validation for user-ingested datasets."""

from __future__ import annotations

import math
from typing import Any

from src.api.schemas import AnnotationDocument, ValidationIssue, ValidationSummary

_SCHEMAS = {
    "detection2d": (["image"], ["class_labels", "bbox_xyxy"]),
    "segmentation": (["image"], ["class_labels", "polygon_or_mask"]),
    "detection3d": (["camera", "calibration", "lidar"], ["class_labels", "box3d"]),
}


def validate_annotation(
    document: AnnotationDocument,
    *,
    sample_id: str,
    sample: dict[str, Any],
    class_map: dict[str, str],
) -> list[ValidationIssue]:
    """Validate a canonical annotation before it is retained or exported."""
    if document.task_id == "detection2d":
        return _validate_boxes(document, sample_id, sample, class_map)
    if document.task_id == "segmentation":
        return _validate_polygons(document, sample_id, sample, class_map)
    return _validate_boxes3d(document, sample_id, class_map)


def summarize_batch(batch: dict[str, Any]) -> ValidationSummary:
    """Return one reproducible readiness decision for a draft upload batch."""
    task_id = str(batch["task_id"])
    input_schema, annotation_schema = _SCHEMAS[task_id]
    class_map = dict(batch.get("class_map", {}))
    issues: list[ValidationIssue] = []
    samples = dict(batch.get("samples", {}))
    if not samples:
        issues.append(_issue("SAMPLES_REQUIRED", "samples", None, "At least one sample is required."))
    if not batch.get("anonymized", False):
        issues.append(_issue("ANONYMISATION_REQUIRED", "anonymized", None, "Benchmark datasets require an anonymisation manifest."))

    if batch.get("dataset_kind") == "attacked_paired":
        manifest = batch.get("attacked_manifest") or {}
        if not manifest.get("clean_dataset_version_id") or not manifest.get("pairs"):
            issues.append(_issue("PAIRED_MANIFEST_REQUIRED", "attacked_manifest", None, "Paired attacked data requires a clean dataset version and sample mapping."))
        for required in ("attack_name", "attack_version", "severity", "seed", "source_hash", "ground_truth_hash"):
            if manifest.get(required) is None:
                issues.append(_issue("PAIRED_MANIFEST_INCOMPLETE", f"attacked_manifest.{required}", None, "Paired attacked data requires complete attack provenance and hashes."))
        known_sample_ids = set(samples)
        mapped_attacked_ids = set()
        for index, pair in enumerate(manifest.get("pairs", [])):
            attacked_id = pair.get("attacked_sample_id") if isinstance(pair, dict) else None
            clean_id = pair.get("clean_sample_id") if isinstance(pair, dict) else None
            if not clean_id or not attacked_id or attacked_id not in known_sample_ids:
                issues.append(_issue("PAIRED_SAMPLE_UNMAPPED", f"attacked_manifest.pairs.{index}", None, "Every paired manifest entry must map one uploaded attacked sample to one clean sample ID."))
            elif attacked_id in mapped_attacked_ids:
                issues.append(_issue("PAIRED_SAMPLE_DUPLICATE", f"attacked_manifest.pairs.{index}", attacked_id, "An attacked sample may appear in only one clean pair."))
            mapped_attacked_ids.add(str(attacked_id))
        for sample_id in sorted(known_sample_ids - mapped_attacked_ids):
            issues.append(_issue("PAIRED_SAMPLE_UNMAPPED", "attacked_manifest.pairs", sample_id, "Each uploaded attacked sample must have a clean pair."))

    for sample_id, sample in samples.items():
        stored = sample.get("annotation")
        if not stored:
            issues.append(_issue("ANNOTATION_REQUIRED", "annotation", sample_id, "A benchmark sample needs a valid annotation."))
            continue
        try:
            document = AnnotationDocument.model_validate(stored)
        except ValueError:
            issues.append(_issue("ANNOTATION_DOCUMENT_INVALID", "annotation", sample_id, "Annotation document does not match the canonical schema."))
            continue
        if document.task_id != task_id:
            issues.append(_issue("TASK_ANNOTATION_MISMATCH", "annotation.task_id", sample_id, "Annotation task does not match the upload batch."))
            continue
        issues.extend(validate_annotation(document, sample_id=sample_id, sample=sample, class_map=class_map))

    errors = [issue for issue in issues if issue.severity == "error"]
    return ValidationSummary(
        state="VALID" if not errors else "INVALID",
        issues=issues,
        task_id=task_id,  # type: ignore[arg-type]
        input_schema=input_schema,
        annotation_schema=annotation_schema,
        benchmark_ready=not errors,
    )


def _validate_boxes(document: AnnotationDocument, sample_id: str, sample: dict[str, Any], class_map: dict[str, str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not document.annotations:
        return [_issue("BOXES_REQUIRED", "annotations", sample_id, "Detection requires at least one labelled box.")]
    width, height = sample.get("width"), sample.get("height")
    for index, annotation in enumerate(document.annotations):
        _validate_class(annotation, index, sample_id, class_map, issues)
        coords = annotation.get("bbox_xyxy")
        if not isinstance(coords, list) or len(coords) != 4 or not all(_finite(value) for value in coords):
            issues.append(_issue("BBOX_INVALID", f"annotations.{index}.bbox_xyxy", sample_id, "bbox_xyxy must contain four finite coordinates."))
            continue
        x1, y1, x2, y2 = (float(value) for value in coords)
        if x1 >= x2 or y1 >= y2:
            issues.append(_issue("BBOX_DEGENERATE", f"annotations.{index}.bbox_xyxy", sample_id, "Bounding boxes must have positive width and height."))
        elif width is None or height is None or x1 < 0 or y1 < 0 or x2 > float(width) or y2 > float(height):
            issues.append(_issue("BBOX_OUT_OF_BOUNDS", f"annotations.{index}.bbox_xyxy", sample_id, "Bounding box coordinates must be inside the original image."))
    return issues


def _validate_polygons(document: AnnotationDocument, sample_id: str, sample: dict[str, Any], class_map: dict[str, str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not document.annotations:
        return [_issue("POLYGONS_REQUIRED", "annotations", sample_id, "Segmentation requires at least one labelled polygon or mask.")]
    width, height = sample.get("width"), sample.get("height")
    for index, annotation in enumerate(document.annotations):
        _validate_class(annotation, index, sample_id, class_map, issues)
        polygon = annotation.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            issues.append(_issue("POLYGON_INVALID", f"annotations.{index}.polygon", sample_id, "A polygon must contain at least three points."))
            continue
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2 or not all(_finite(value) for value in point):
                issues.append(_issue("POLYGON_INVALID", f"annotations.{index}.polygon", sample_id, "Polygon points must be finite [x, y] pairs."))
                break
            x, y = (float(value) for value in point)
            if width is None or height is None or x < 0 or y < 0 or x > float(width) or y > float(height):
                issues.append(_issue("POLYGON_OUT_OF_BOUNDS", f"annotations.{index}.polygon", sample_id, "Polygon points must remain inside the original image."))
                break
    return issues


def _validate_boxes3d(document: AnnotationDocument, sample_id: str, class_map: dict[str, str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not document.calibration:
        issues.append(_issue("CALIBRATION_REQUIRED", "calibration", sample_id, "3D detection requires camera calibration."))
    if not document.lidar_path or ".." in document.lidar_path.replace("\\", "/").split("/"):
        issues.append(_issue("LIDAR_REQUIRED", "lidar_path", sample_id, "3D detection requires a dataset-relative LiDAR file."))
    if not document.annotations:
        issues.append(_issue("BOX3D_REQUIRED", "annotations", sample_id, "3D detection requires labelled 3D boxes."))
    for index, annotation in enumerate(document.annotations):
        _validate_class(annotation, index, sample_id, class_map, issues)
        box = annotation.get("box3d")
        if not isinstance(box, list) or len(box) != 7 or not all(_finite(value) for value in box):
            issues.append(_issue("BOX3D_INVALID", f"annotations.{index}.box3d", sample_id, "box3d must contain seven finite values."))
    return issues


def _validate_class(annotation: dict[str, Any], index: int, sample_id: str, class_map: dict[str, str], issues: list[ValidationIssue]) -> None:
    class_id = annotation.get("class_id")
    if not isinstance(class_id, str) or class_id not in class_map:
        issues.append(_issue("CLASS_NOT_IN_MAP", f"annotations.{index}.class_id", sample_id, "Annotation class must exist in the batch class map."))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _issue(code: str, field: str, sample_id: str | None, message: str) -> ValidationIssue:
    return ValidationIssue(code=code, field=field, sample_id=sample_id, message=message)
