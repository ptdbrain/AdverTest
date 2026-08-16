from src.evaluation.bootstrap import paired_bootstrap
from src.pipeline.protocol import BenchmarkProtocol


def test_paired_bootstrap_resamples_ids_deterministically() -> None:
    protocol = BenchmarkProtocol(
        name="bootstrap",
        dataset_version_id="d1",
        sample_ids=("a", "b", "c"),
        sample_hashes={key: key for key in ("a", "b", "c")},
        ground_truth_hashes={key: key for key in ("a", "b", "c")},
        bootstrap_iterations=100,
        bootstrap_seed=7,
    ).transition("VALIDATED").transition("LOCKED")
    values = {"a": 1.0, "b": 2.0, "c": 3.0}

    first = paired_bootstrap(
        protocol.sample_ids,
        lambda ids: sum(values[item] for item in ids) / len(ids),
        protocol,
        name="mean",
        unit="points",
    )
    second = paired_bootstrap(
        protocol.sample_ids,
        lambda ids: sum(values[item] for item in ids) / len(ids),
        protocol,
        name="mean",
        unit="points",
    )

    assert first == second
    assert first.value == 2.0
    assert first.ci95 is not None
    assert first.metadata["resampling_unit"] == "sample_id"
