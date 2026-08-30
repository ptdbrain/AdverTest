from __future__ import annotations

import numpy as np

from src.core.types import MaskPrediction, Sample, SegmentationPrediction
from src.evaluation.bdd_semantic_metrics import BDD100KSemanticEvaluator
from src.pipeline.protocol import BenchmarkProtocol
from src.segmentation.evidence import SamSegmentationEvidenceSerializer


def test_bdd_metric_is_nonpaired_and_evidence_writes(tmp_path) -> None:
    image = np.zeros((4, 4, 3), dtype=np.float32)
    target = np.zeros((4, 4), dtype=np.uint8)
    target[1:3, 1:3] = 1
    sample = Sample("s", image, mask=target, meta={"semantic_mask": target.astype(bool)})
    prediction = SegmentationPrediction(
        sample_id="s", prompt_id="p", instances=(MaskPrediction(instance_id="1", mask=target.astype(bool)),)
    )
    protocol = (
        BenchmarkProtocol.minimal(name="bdd", dataset_version_id="d", sample_ids=("s",))
        .transition("VALIDATED")
        .transition("LOCKED")
    )
    result = BDD100KSemanticEvaluator().evaluate([prediction], [sample], protocol)
    assert result.headline.metadata["comparison_scope"] == "external_semantic_nonpaired"
    paths = SamSegmentationEvidenceSerializer().write(
        root=tmp_path, clean=sample, attacked=sample, clean_prediction=prediction, attacked_prediction=prediction
    )
    assert all(__import__("pathlib").Path(path).is_file() for path in paths.values())
