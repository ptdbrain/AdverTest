# Person D completion matrix

Snapshot: 2026-08-06, branch `feat/person-d-platform`.

| Requirement | Implementation | Direct evidence | Status / dependency |
|---|---|---|---|
| Shared prediction, mask, job, objective contracts | `src/core/types.py`, `contracts.py`, `events.py`, `objectives.py` | `tests/test_core` | COMPLETE (CPU) |
| Dataset identity and deterministic locked splits | `src/datasets/versioning.py`, `splits.py` | dataset version/split tests | COMPLETE (CPU) |
| Leakage and lineage gates | `src/datasets/leakage.py` | leakage and training-builder tests | COMPLETE (CPU) |
| Versioned catalog and deterministic recipes | `src/attacks/catalog.py`, `recipes.py` | catalog/recipe tests, reproducibility integration test | COMPLETE (CPU) |
| Ordered composition and annotation transforms | `src/attacks/composition.py`, `transforms.py` | composition/transform tests | COMPLETE (CPU) |
| Generated dataset, resume, cache separation | `src/pipeline/generator.py`, `cache.py` | generator/cache tests | COMPLETE (CPU) |
| Hard-example bank and grouping | `src/evaluation/hard_examples.py` | hard-example tests | COMPLETE (CPU) |
| Defense training manifest | `src/training/dataset_builder.py` | training dataset builder tests | COMPLETE (CPU) |
| Locked benchmark protocol and generic multi-model runner | `src/pipeline/protocol.py`, `generic_benchmark.py` | protocol/generic benchmark tests | COMPLETE (CPU) |
| Recovery, paired bootstrap, comparison, promotion gate | `src/evaluation/recovery_metrics.py`, `bootstrap.py`, `model_comparison.py` | evaluation and recovery integration tests | COMPLETE (CPU) |
| Generic trainer/worker lifecycle | `src/training/base.py`, `registry.py`, `worker.py` | `tests/test_training_orchestration.py` | COMPLETE (CPU fakes) |
| Stable service facade and JSON CLI | `src/services/person_d.py`, `src/cli.py` | service/CLI tests | COMPLETE (CPU) |
| Detection owner integration | B-owned runnable adapter/evaluator/checkpoint | `tests/gpu/test_person_d_yolo11.py` | WAITING_FOR_OWNER unless checkpoint env is supplied |
| Segmentation owner integration | C-owned runnable adapter/evaluator/prompt/checkpoint | `tests/gpu/test_person_d_sam2.py` | WAITING_FOR_OWNER unless checkpoint/config env are supplied |
| A persistence/API integration | A-owned persistence and callbacks | handoff contract | WAITING_FOR_OWNER |

## Recorded quality commands

The final run must record fresh results for:

```powershell
uv run --no-sync ruff check --no-cache src tests
uv run --no-sync pytest -q
uv run --no-sync pytest tests/integration -q
uv run --no-sync pytest -m gpu tests/gpu -q
```

Skipped GPU tests are evidence of an unavailable external handoff, not evidence
that real-model integration passed.
