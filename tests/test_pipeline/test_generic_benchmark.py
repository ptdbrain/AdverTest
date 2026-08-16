from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np

from src.adapters.base import ModelAdapter
from src.core.types import (
    DetectionPrediction,
    MaskPrediction,
    ModelInfo,
    ModelPrediction,
    Sample,
    SegmentationPrediction,
)
from src.evaluation.base import EvaluationResult
from src.evaluation.contracts import MetricEnvelope
from src.pipeline.generic_benchmark import BenchmarkRunner
from src.pipeline.protocol import BenchmarkProtocol


class FakeAdapter(ModelAdapter):
    def __init__(self, name: str, task: str):
        super().__init__()
        self.name = name
        self.task = task
        self.calls = 0

    def metadata(self) -> ModelInfo:
        return ModelInfo(name=self.name, task=self.task, version=f"{self.name}-v1")

    def predict(self, samples: Sequence[Sample]) -> list[ModelPrediction]:
        self.calls += 1
        if self.task == "segmentation":
            return [
                SegmentationPrediction(
                    sample_id=sample.sample_id,
                    instances=(
                        MaskPrediction(
                            instance_id="mask-1",
                            mask=np.ones(sample.image.shape[:2], dtype=np.bool_),
                        ),
                    ),
                )
                for sample in samples
            ]
        return [DetectionPrediction(sample_id=sample.sample_id) for sample in samples]


class FakeEvaluator:
    def __init__(self, task: str):
        self.task = task
        self.metric_versions = {"score": "1.0.0"}

    def evaluate(self, predictions, samples, protocol):
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
            per_sample_metrics={sample.sample_id: (metric,) for sample in samples},
        )


def test_generic_runner_pairs_detection_and_segmentation_without_heavy_imports() -> None:
    sample = Sample(
        sample_id="s1",
        image=np.zeros((4, 5, 3), dtype=np.float32),
        mask=np.ones((4, 5), dtype=np.bool_),
        anonymized=True,
    )
    protocol = BenchmarkProtocol(
        name="fixture",
        dataset_version_id="dataset-1",
        sample_ids=("s1",),
        sample_hashes={"s1": "source-1"},
        ground_truth_hashes={"s1": "gt-1"},
        recipe_hashes=("recipe-1",),
        seeds=(195,),
        metric_versions={"score": "1.0.0"},
        created_at=datetime.now(UTC),
    ).transition("VALIDATED").transition("LOCKED")
    models = [FakeAdapter("detector", "detection2d"), FakeAdapter("segmenter", "segmentation")]
    events = []
    runner = BenchmarkRunner(
        sample_provider=lambda _: [sample],
        variant_provider=lambda recipe_hash, samples: [
            item.with_image(np.full_like(item.image, 0.1)) for item in samples
        ],
    )

    first = runner.run(
        protocol,
        models,
        {"detection2d": FakeEvaluator("detection2d"), "segmentation": FakeEvaluator("segmentation")},
        events.append,
    )
    second = runner.run(
        protocol,
        models,
        {"detection2d": FakeEvaluator("detection2d"), "segmentation": FakeEvaluator("segmentation")},
        events.append,
    )

    assert len(first.models) == 2
    assert all(result.clean.headline.unit == "ratio" for result in first.models)
    assert all(len(result.cells) == 1 for result in first.models)
    assert all(result.paired_sample_ids == ("s1",) for result in first.models)
    assert second.resumed_cells == 2
    assert [event.sequence for event in events] == sorted(event.sequence for event in events)
    assert all(model.calls == 2 for model in models)


def test_generic_runner_reports_incompatibility_and_valid_partial_cancellation() -> None:
    sample = Sample(
        sample_id="s1",
        image=np.zeros((2, 2, 3), dtype=np.float32),
        anonymized=True,
    )
    protocol = BenchmarkProtocol(
        name="fixture",
        dataset_version_id="dataset-1",
        sample_ids=("s1",),
        sample_hashes={"s1": "source-1"},
        ground_truth_hashes={"s1": "gt-1"},
        recipe_hashes=("recipe-1", "recipe-2"),
        metric_versions={"score": "1.0.0"},
    ).transition("VALIDATED").transition("LOCKED")
    cancel_calls = 0

    def cancel_after_first_cell() -> bool:
        nonlocal cancel_calls
        cancel_calls += 1
        return cancel_calls >= 3

    runner = BenchmarkRunner(
        sample_provider=lambda _: [sample],
        variant_provider=lambda _recipe_hash, samples: samples,
        cancel_check=cancel_after_first_cell,
    )
    report = runner.run(
        protocol,
        [FakeAdapter("detector", "detection2d"), FakeAdapter("segmenter", "segmentation")],
        {"detection2d": FakeEvaluator("detection2d")},
    )

    assert report.complete is False
    assert report.cancellation_reason
    assert len(report.models) == 1
    assert len(report.models[0].cells) == 1


def test_generic_runner_skips_metric_version_mismatch_explicitly() -> None:
    sample = Sample(
        sample_id="s1",
        image=np.zeros((2, 2, 3), dtype=np.float32),
        anonymized=True,
    )
    protocol = BenchmarkProtocol(
        name="fixture",
        dataset_version_id="dataset-1",
        sample_ids=("s1",),
        sample_hashes={"s1": "source-1"},
        ground_truth_hashes={"s1": "gt-1"},
        metric_versions={"score": "2.0.0"},
    ).transition("VALIDATED").transition("LOCKED")
    runner = BenchmarkRunner(
        sample_provider=lambda _: [sample],
        variant_provider=lambda _recipe_hash, samples: samples,
    )

    report = runner.run(
        protocol,
        [FakeAdapter("detector", "detection2d")],
        {"detection2d": FakeEvaluator("detection2d")},
    )

    assert report.models == ()
    assert "metric version mismatch" in report.skipped[0].reason
