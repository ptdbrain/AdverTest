# Evidence-first product flow and curated datasets

## Purpose

Make AdverTest scientifically honest and project-safe: a visual inference result
is not a benchmark, promotion is only possible from verified evidence, all run
artifacts are isolated by project and run, and the local datasets are reduced
to deterministic, internally useful subsets without inventing missing data.

## Scope and invariants

- A report, metric, or promotion is **verified** only when its dataset version,
  official/local split declaration, ground truth, checkpoint hash, resolved
  config hash, artifact hashes, and metric protocol are all present and valid.
- Missing evidence produces `NOT_ELIGIBLE`; it must not produce a substitute
  score, a promotion decision, or a benchmark conclusion.
- Live/visual inference may expose predictions, confidence distribution and
  image artifacts. It must never manufacture mAP, mIoU, NDS, AP, or robustness
  drop from confidence values.
- All persisted user-visible artifacts use immutable logical keys beginning
  `projects/{project_id}/runs/{run_id}/`. Static frontend directories are not
  an artifact store.
- Authorization is enforced by the backend from the authenticated user and
  active membership; `project_id` supplied in a URL or request is a scope to
  verify, not authority.
- CPU-only execution may validate loaders, data integrity, contracts, and UI.
  PointPillars remains `WAITING_FOR_GPU_VALIDATION` until a compatible CUDA
  runtime returns non-empty finite `pred_instances_3d` with aligned shapes.

## Dataset curation policy

The user authorized reducing source datasets. Before deletion, the curator
creates a manifest and an integrity report, then uses an atomic staging
directory and a rollback manifest. It never deletes `data/artifacts`,
`data/checkpoints`, databases, benchmark reports, source code, or unrelated
user content.

### Retained datasets

| Logical task | Retained source | Size | Required paired evidence |
| --- | --- | ---: | --- |
| KITTI 2D detection | `data/kitti/Kitti/raw/training` | 100 labelled frames | `image_2/{id}.png`, `label_2/{id}.txt` |
| KITTI semantic segmentation | `data/datasets/kitti_semantics` | 100 frames | image, semantic mask, instance mask, RGB mask, matching id |
| nuScenes 3D/multimodal | `data/nuscenes/v1.0-mini` | 100 keyframes across 10 scenes | six camera keyframes, `LIDAR_TOP`, five radar keyframes, calibration, ego pose, annotations |

The 100 ids are deterministic (fixed seed recorded in the manifest) and
stratified by available class/scene evidence. The nuScenes subset uses
single-sweep (`num_sweeps=1`) evidence only; it does not retain unrelated
historical sweeps and declares that limitation in the protocol. Its metadata
tables are rewritten so every retained token and every file reference resolves.

There are only five complete KITTI 3D fixture samples in `data/training`. They
are retained as a development fixture, explicitly marked
`INSUFFICIENT_FOR_REQUESTED_SUBSET`, and cannot enable KITTI 3D benchmark or
promotion. nuScenes is the retained 3D dataset.

### Deletion targets after validation

- Unselected files from the three retained dataset roots above.
- Unlabelled KITTI testing images and duplicate KITTI download ZIP files once
  their selected training data has passed checksum and pairing validation.
- Unselected Cityscapes/demo segmentation source material, because the
  retained KITTI semantic set is the declared segmentation source.
- NuScenes sweeps and metadata/file references not needed by the selected
  100 single-sweep keyframes.

Deletion happens only after a dry-run prints exact file count, bytes, retained
ids, and checksum manifest; after staging passes all integrity tests. The
operation uses a recoverable quarantine directory inside the same data volume
until final verification completes, then removes only the listed targets.

## Evidence and report contract

Introduce an evidence evaluator shared by run creation, report export,
promotion, dashboard analytics, and API serialization.

`EvidenceStatus` is one of `VERIFIED`, `NOT_ELIGIBLE`,
`WAITING_FOR_GPU_VALIDATION`, or `INVALID`. It returns structured missing
requirements and a human action. A report stores an immutable evidence
snapshot and hashes it. CSV, JSON, and PDF serialize the same snapshot.

Metric records have explicit name, unit, value, protocol, sample count,
ground-truth source, dataset-version id, and calculation provenance. A
visual-only run cannot contain benchmark metric names. Its UI says “Visual
inference — not a benchmark” and only renders valid prediction-derived fields.

Promotion endpoints reject non-`VERIFIED` evidence with 409 and do not emit a
promotion artifact. Download endpoints may provide a diagnostic report for
`NOT_ELIGIBLE`, but its title and body must state that no benchmark conclusion
was produced.

## Project context, deep links, and artifacts

Use a single frontend project-context provider backed by the authenticated
project list. The provider updates a canonical `project_id` URL parameter and
the request scope; project changes clear run-specific state.

Run pages have canonical links containing both `project_id` and `run_id`.
Server routes load the run only through project membership and return 403 for a
foreign project and 404 for absent runs. A stale/deleted project produces a
clear selection state rather than silently falling back to another project.

Live inference creates or attaches to a run and writes clean, attacked, diff,
perturbation, prediction, and visual-summary artifacts through the artifact
repository. Artifact keys are project/run scoped and response URLs are API
authorized; no request writes under `frontend/public`.

## Docker and UX

Compose receives the host curated dataset root only through
`DATA_ROOT_HOST`, mounted read-only at `/data`. Compose fails configuration if
the variable is absent. Health checks cover database, queue, object storage,
backend and frontend.

The Docker E2E flow creates a project, creates a scoped run, uploads or
generates an authorized artifact, reads the run through a deep link, and
downloads JSON/CSV/PDF. It verifies that an incomplete evidence run cannot be
promoted. CUDA is not part of this E2E gate.

Dashboard and run screens consume API evidence fields. Missing verified values
render `—` / `No verified data`, a status badge, limitation text, and the next
valid action. Mobile navigation has a 44px minimum target, a drawer sidebar,
and preserves the selected project in deep links.

## Tests and acceptance criteria

1. Tests first demonstrate that a confidence-only live inference response has
   no benchmark metric fields and cannot be promoted.
2. Tests reject reports missing every individual required evidence field and
   accept one complete immutable evidence bundle.
3. Tests prove cross-project artifact/run deep links return 403 and valid
   project/run links restore the correct context.
4. Tests prove live inference stores project/run artifact keys and never writes
   frontend public assets.
5. Dataset tests validate 100 exact paired KITTI 2D samples, 100 exact paired
   KITTI semantic samples, 100 nuScenes keyframes with all selected sensor,
   pose, calibration, and annotation references, and no dangling rewritten
   nuScenes metadata tokens.
6. The KITTI 3D registry test remains not-ready with an explicit missing-count
   reason; no CPU execution upgrades it to ready.
7. Docker Compose configuration and E2E pass with a read-only data mount;
   production healthchecks are green.
8. Backend tests, frontend tests, lint, type/build gates, and `git diff --check`
   pass. Actual CUDA validation remains reported as an external blocked gate.
