# Integration Completion Matrix

Recorded locally on 2026-08-10. “CPU contract” means deterministic repository verification, not a scientific result from external model artefacts.

| Area | Evidence command or test | Status | Evidence class |
| --- | --- | --- | --- |
| Durable generated data | `tests/test_api/test_generated_datasets.py` | Complete | CPU contract |
| Backlog persistence | `tests/test_api/test_retraining_backlogs.py` | Complete | CPU contract |
| Training resources | `tests/test_api/test_training_resources.py` | Complete | CPU contract |
| Paired comparison/export | `tests/test_api/test_comparison_reports.py` | Complete | CPU contract |
| Closed-loop HTTP acceptance | `tests/integration/test_closed_loop_e2e.py` | Complete | CPU contract; training gate asserted |
| Frontend recovery UI | `npm --prefix frontend run test -- --run` | Complete | UI contract |
| Frontend production build | `npm --prefix frontend run lint`; `npm --prefix frontend run build` | Complete | Static quality gate |
| Full Python quality | `uv run --no-sync ruff check src tests`; `uv run --no-sync pytest -q` | Complete: 743 passed, 7 skipped | Repository gate |
| Integration suite | `uv run --no-sync pytest tests/integration -q` | Complete: 5 passed | CPU contract |
| Model-marked suite | `uv run --no-sync pytest -m models -q` | 750 deselected; no verified model test selected | External validation still required |
| Real YOLO retraining/evaluation | Verified dataset, checkpoint, and external compute | `WAITING_FOR_ARTIFACTS` | External validation |
| SAM2 handoff/evaluation | Person C compatible artefacts | `WAITING_FOR_ARTIFACTS` | External dependency |
