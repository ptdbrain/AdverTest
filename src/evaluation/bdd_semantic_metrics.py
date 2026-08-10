"""Non-paired BDD100K semantic external evaluation for SAM outputs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.core.types import ModelPrediction, Sample, SegmentationPrediction
from src.evaluation.base import EvaluationResult
from src.evaluation.contracts import MetricEnvelope
from src.evaluation.segmentation_metrics import binary_iou
from src.pipeline.protocol import BenchmarkProtocol

METRIC_VERSION = "advertest-bdd-semantic-external-v1"


class BDD100KSemanticEvaluator:
    task = "segmentation"
    metric_versions = {"bdd_semantic_iou": METRIC_VERSION}

    def evaluate(self, predictions: Sequence[ModelPrediction], samples: Sequence[Sample], protocol: BenchmarkProtocol) -> EvaluationResult:
        by_id = {item.sample_id: item for item in predictions if isinstance(item, SegmentationPrediction)}
        values: dict[str, float] = {}
        for sample in samples:
            target = sample.meta.get("semantic_mask")
            prediction = by_id.get(sample.sample_id)
            if not isinstance(target, np.ndarray) or prediction is None:
                continue
            union = np.zeros(target.shape, dtype=bool)
            for instance in prediction.instances:
                if instance.mask.shape != target.shape:
                    raise ValueError("BDD semantic prediction shape does not match label map")
                union |= instance.mask
            values[sample.sample_id] = binary_iou(union, target)
        score = float(np.mean(list(values.values()))) if values else 0.0
        metric = MetricEnvelope(name="bdd_semantic_iou", value=score, unit="ratio", percent_value=score * 100, version=METRIC_VERSION, higher_is_better=True, metadata={"comparison_scope": "external_semantic_nonpaired"})
        return EvaluationResult(task="segmentation", protocol_id=protocol.protocol_id, headline=metric, per_sample_metrics={sample_id: (MetricEnvelope(name="bdd_semantic_iou", value=value, unit="ratio", percent_value=value * 100, version=METRIC_VERSION, higher_is_better=True, metadata={"comparison_scope": "external_semantic_nonpaired"}),) for sample_id, value in values.items()}, validation_warnings=("BDD100K semantic evaluation is external and non-paired; do not use it for checkpoint selection or Recovery.",))
