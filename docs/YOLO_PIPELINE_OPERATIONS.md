# YOLO Closed-Loop Operations

This guide operates the repository-backed CPU workflow. It does not claim that a real YOLO checkpoint, external dataset, or SAM2 artefact is present.

## Start locally

```powershell
uv sync
uv run uvicorn src.main:app --reload --port 8000
npm --prefix frontend run dev
```

Use an isolated database and artifact location for verification:

```powershell
$env:DATABASE_URL = 'sqlite:///./.verify/app.db'
$env:ARTIFACT_ROOT = '.verify/artifacts'
```

## Evidence workflow

1. Start a detection benchmark with `POST /api/v1/runs` and wait for its durable status to become `COMPLETED`.
2. Start a second run under the same dataset/sample protocol, then create `POST /api/v1/model-comparisons`. Read pairedness before using `/metric-deltas`; download `/export?format=json`, `csv`, or `html` for retained evidence.
3. In the detection workspace, create a retraining backlog from measured degradation cells. The UI persists every failure ID and approves the backlog only after it contains items.
4. Estimate a registered trainer with `POST /api/v1/training-runs/estimate`. Start requests are asynchronous and inspectable at `/training-runs/{id}`, `/events`, `/checkpoints`, and `/cancel`.
5. A missing/non-runnable model version returns `409 WAITING_FOR_ARTIFACTS`. Supply a verified checkpoint and the required dataset/defense inputs before scheduling real training; do not bypass this gate.

## Intentional external gates

- Real YOLO retraining/evaluation remains `WAITING_FOR_ARTIFACTS` until a runnable checkpoint, approved source data, and external compute are supplied.
- SAM2 UI/API remains blocked until Person C's artefacts satisfy the installed handoff contract. Detection metrics must never be presented as segmentation results.
