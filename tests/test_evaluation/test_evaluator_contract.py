from collections.abc import Sequence

from src.core.types import DetectionPrediction, ModelPrediction, Sample
from src.evaluation.base import EvaluationResult, TaskEvaluator
from src.evaluation.contracts import MetricEnvelope
from src.pipeline.protocol import BenchmarkProtocol


class FakeEvaluator:
    task = "detection2d"
    metric_versions = {"score": "1.0.0"}

    def evaluate(
        self,
        predictions: Sequence[ModelPrediction],
        samples: Sequence[Sample],
        protocol: BenchmarkProtocol,
    ) -> EvaluationResult:
        metric = MetricEnvelope(
            name="score",
            value=1.0,
            unit="ratio",
            percent_value=100.0,
            version="1.0.0",
            higher_is_better=True,
        )
        return EvaluationResult(
            task=self.task,
            protocol_id=protocol.protocol_id,
            headline=metric,
            per_sample_metrics={samples[0].sample_id: (metric,)},
        )


def test_detection_and_segmentation_evaluators_share_one_boundary(sample: Sample) -> None:
    protocol = BenchmarkProtocol.minimal(
        name="fixture",
        dataset_version_id="dataset-1",
        sample_ids=(sample.sample_id,),
    )
    evaluator: TaskEvaluator = FakeEvaluator()
    result = evaluator.evaluate(
        [DetectionPrediction(sample_id=sample.sample_id)],
        [sample],
        protocol,
    )
    assert result.protocol_id == protocol.protocol_id
    assert result.headline.percent_value == 100.0
