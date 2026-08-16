# AdverTest YOLO Integration Design

## Goal

Deliver a truthful, runnable YOLO11 robustness workflow using Person B's
existing B0, R1, and R2 training artefacts: ingest an image or a valid
versioned dataset, measure clean performance, generate an attack recipe,
measure the attacked performance on the same samples, and present/export an
auditable comparison.  The shared segmentation path is prepared for Person
C's SAM2 handoff, but never fabricates masks, metrics, or checkpoint results.

## Scope and boundaries

The in-scope product covers two perception modes, `detection` and
`segmentation`. Detection is fully operational with B's checkpoints. The
segmentation selector, contracts, validation, viewer states, and report schema
are implemented now; an attempt to infer, benchmark, or compare segmentation
returns a machine-readable `WAITING_FOR_ARTIFACTS` state until a runnable SAM2
adapter, real masks, prompt protocol, and model artefacts are supplied.

The implementation excludes 3D/LiDAR, automatic red-team search, synthetic
ground-truth masks, and web-triggered distributed GPU training. Local queued
training orchestration may register and expose its state, but real training is
only dispatched through installed model-specific trainers.

## Artifact registration and model lineage

An artefact scanner reads only known text metadata and explicitly selected
weight files below `runs/`. It registers immutable model versions instead of
hard-coded model names:

| Logical version | Source | Parent / intended role |
| --- | --- | --- |
| `yolo11s-kitti-clean-b0` | `runs/train/yolo_b0/...` | clean baseline |
| `yolo11s-kitti-robust-r1` | `runs/train/yolo_r1/...` | robust-mix child of B0 |
| `yolo11s-kitti-repaired-r2-fog` | `runs/train/yolo_r2_fog/...` | targeted fog repair |
| `yolo11s-kitti-repaired-r2-sensor-fault` | `runs/train/yolo_r2_sensor/...` | targeted sensor-fault repair |

Every version stores the canonical checkpoint path, SHA-256 digest, Ultralytics
arguments/result metadata, training dataset reference if known, parent ID,
task, class map, and registration timestamp. A missing, unreadable, or
digest-mismatched checkpoint remains visible but is not runnable.

## Data, recipes, and benchmark contracts

Users can upload one image for an evidence-only single-image run or ingest a
folder/dataset into a versioned dataset. A benchmark requires an explicit
dataset version, locked split, ground-truth boxes, anonymization approval, and
a deterministic seed. The API rejects a benchmark dataset used as a training
source and rejects incompatible attack/task/dataset combinations before jobs
are queued.

Recipes are ordered, immutable records with an ID, implementation versions,
per-step severity/parameters, seed, estimates, and compatibility outcome.
They support single attack, manual composition, random N attacks, random by
group, scenario preset, and bounded auto sweep. Every generated variant stores
source/variant hashes, transformed annotation provenance, recipe, seed, and
evidence paths. Compatibility adapters preserve the existing single-attack
`RunConfig` API while the new API exposes recipe-first endpoints.

A locked benchmark protocol pins the dataset/split hashes, model preprocessing,
recipe, thresholds, class map, metric versions, seed, and environment version.
Baseline and defended comparisons are paired only when these fields match.

## Execution and measurements

The asynchronous worker runs this sequence:

```text
validate dataset and recipe
-> load registered model version
-> clean prediction + ground-truth evaluation
-> generate deterministic attacked variants
-> attacked prediction + evaluation
-> persist sample/cell evidence and progress events
-> derive failures, robustness metrics, and report
```

For detection, reports emit clean mAP/AP50/AP75, attacked score, absolute and
relative degradation, objects broken, attack success rate where the attack
defines it, robustness score, severity results, per-class/per-size results,
and explicit metric units. The report records unavailable values rather than
substituting an unrelated metric. A model comparison additionally emits
paired/unpaired compatibility, clean trade-off, deltas, recovery rate,
confidence intervals when sample-level data permits, gate outcome, lineage,
and residual-risk narrative.

## API and persistence

The existing queued `/runs` endpoints remain compatible. New resource routes
model attack recipes, generated datasets, benchmark protocols/runs, model
versions/lineage, and comparisons/recovery reports. Long-running operations
return `202`, persist ordered events in SQLite, support cancellation and
resume-safe checkpoints, and expose a WebSocket event stream. Errors carry a
stable code, user message, and remediation context.

SQLite remains the development persistence layer. Its repository boundaries
must keep PostgreSQL/Redis/Celery/S3 substitutions possible without changing
domain contracts or frontend behaviour.

## Product interface

The application uses the plan's two-region layout: a left configuration panel
and a center workspace. It deliberately removes the narrow right metric panel.

The left panel selects perception mode, dataset/version/split, model version,
attack recipe mode, severity/intensity, and preflight estimate. Switching mode
filters datasets, model versions, attacks, metric labels, prompt controls, and
viewer capabilities. Controls show why an unavailable or incompatible option
is blocked.

The center workspace has five ordered sections: (1) input and ground truth,
(2) attacked input with ordered recipe/provenance, (3) clean prediction, (4)
attacked prediction and failure status, and (5) five headline metrics. An
advanced drawer provides detailed metrics, failure cases, lineage, progress,
and export. Comparison mode selects baseline, defended version, and locked
protocol and renders the recovery narrative, metric deltas, clean trade-off,
and pairedness warning.

Status always has text and icons in addition to colour. Metric values carry a
unit; ratios and percentages use separate fields; keyboard controls and
clickable explanatory popovers are supported. SAM2 is visibly marked as
waiting for C until its gate passes.

## Acceptance criteria

1. B0, R1, and both R2 artefacts are registered with verified hashes and
   visible lineage; the model selector contains no `blob_detector` default.
2. A valid YOLO image/dataset run performs clean and attacked inference on the
   same samples, persists evidence, and exposes progress/cancellation through
   the API and UI.
3. Recipe validation supports all six required modes deterministically and
   rejects incompatible requests before enqueueing.
4. A locked protocol produces a report with the stated five headline metrics,
   detail tables, provenance, and exportable artefacts.
5. B0-versus-R1/R2 comparison rejects unpaired inputs and otherwise reports
   recovery, clean trade-off, lineage, and residual risk without invented
   scientific claims.
6. The UI matches the two-region/five-workspace-section layout and passes
   lint/build checks plus focused browser/API tests.
7. The SAM2 path has shared contracts and an honest waiting state. It becomes
   runnable only after C's adapter, checkpoint, masks, prompt protocol, and
   metric evaluator pass contract tests.

## Verification strategy

Unit tests cover artefact discovery, hashing, model registration, recipe
validation/randomness, protocol locking, paired comparisons, metric envelopes,
and error codes. Integration tests use a small annotated detection fixture to
run the complete asynchronous pipeline and assert repeatable manifests,
evidence, results, cancellation, and resume. Optional real-checkpoint tests
run only when B's local weights and Ultralytics dependencies are available;
they must never claim real detection validation on synthetic fixtures.

Frontend checks lint and build the Next.js application, verify mode and model
filtering, upload/preflight/run progress, five-section evidence rendering,
comparison warnings, and the SAM waiting state. A final recorded acceptance
run will document exact commands, passed tests, skipped optional checks, and
external dependencies still waiting for C.
