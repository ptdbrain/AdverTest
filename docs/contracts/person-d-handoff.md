# Person D compute handoff

Version: `1.0.0`

Person D owns deterministic dataset identity/splits, attack recipes and generated
manifests, leakage-safe training manifests, generic benchmark orchestration,
paired comparison, and guarded training-worker execution. The service boundary
is `PersonDServices`; it imports no FastAPI or frontend code.

## Shared rules

- JSON contracts are immutable Pydantic models with `extra="forbid"`.
- Ratios use unit `ratio` and carry the equivalent `percent_value`; points,
  counts, seconds, and bytes are never silently converted.
- Sample IDs and their order must exactly match a `LOCKED` benchmark protocol.
- Progress events use `JobRequest`/`ProgressEvent` version `1.0.0`; sequence
  numbers increase monotonically within a worker instance.
- Validation failures return a non-zero CLI status and a JSON `error` or typed
  validation report. No Person D command downloads model weights.

## Owner callbacks

- Owner A supplies persistence/API bindings around `PersonDServices` and stores
  emitted progress events without renumbering them.
- Owner B registers detection adapters, evaluators, evidence serializers, and a
  `YoloTrainer` implementation of `ModelTrainer`.
- Owner C registers segmentation adapters, evaluators, mask/boundary evidence
  serializers, and a `Sam2Trainer` implementation of `ModelTrainer`.
- Person D does not fabricate those real integrations. Missing plugins or local
  checkpoints are reported as `WAITING_FOR_OWNER`/skipped in acceptance evidence.

## Lifecycle and errors

Training follows `DRAFT -> VALIDATING -> ESTIMATING -> QUEUED ->
PREPARING_DATA -> TRAINING -> VALIDATING_CHECKPOINT -> EXPORTING ->
REGISTERING_MODEL -> COMPLETED`. Active states may end in `FAILED`, `CANCELLED`,
or `BUDGET_EXCEEDED`. Registration metadata is emitted only after lineage,
checkpoint hash, evaluation, export-load, and export-hash checks pass.

Common machine-readable reasons include `metric version mismatch`,
`storage_budget_exceeded`, `checkpoint hash validation failed`,
`cancellation requested`, and explicit protocol comparison field names.

## CLI

All service commands accept `--config <json>`: `dataset-ingest`,
`dataset-split`, `dataset-validate`, `recipe-validate`, `recipe-sample`,
`generate-recipe`, `inspect-attack-dataset`, `build-training-dataset`,
`benchmark-run`, `compare-models`, `training-estimate`, and `training-run`.
The JSON schemas are the corresponding model schemas in `src/datasets`,
`src/attacks/recipes.py`, `src/pipeline`, `src/evaluation/model_comparison.py`,
and `src/training`.
