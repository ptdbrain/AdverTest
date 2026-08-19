# Platform contract V1

Status: ready for A/B/C/D review. This document is the complete Sprint 0
proposal; it becomes accepted only when the contract PR is approved and merged.

## Identity and tenancy

- All persistent IDs are server-generated UUIDs.
- `project_id` is the tenancy boundary. Every platform resource is scoped to a
  project; authorization checks use the authenticated actor plus project
  membership.
- Clients never submit `owner_user_id`, `created_by_user_id`, or `storage_key`.
- `Dataset` is a logical collection. `DatasetVersion` is an immutable snapshot
  used by training and benchmarks; both IDs must be persisted in its record.
- Object storage keys use `projects/{project_id}/artifacts/{artifact_id}/...`.
  They must not depend on a user's ID because projects can have many members.

## Resource boundaries

- `ArtifactV1` describes bytes in object storage and their integrity metadata.
  Signed URLs are short-lived responses, never persisted as artifact metadata.
- `CheckpointArtifactV1` describes model semantics and references exactly one
  artifact through `artifact_id`.
- `JobV1` tracks durable asynchronous work. `JobProgressEventV1` is append-only
  and uses a monotonically increasing sequence per job.
- `ArtifactReferenceV1` is the only generic provenance edge. It records every
  platform entity's input, output, evidence, report, or export artifact.
- Existing domain contracts such as `src.attacks.recipes.AttackRecipe` and
  `src.evaluation.ModelComparison` remain the source of scientific content.
  Their `*V1` platform records add project ownership, persistence IDs, jobs,
  artifacts, and lifecycle state; they do not replace the domain contracts.

## Complete shared entity list

| Shared entity | Canonical Sprint 0 platform contract | Owner of domain behavior |
|---|---|---|
| TaskDefinition | `TaskDefinitionV1` | A |
| ModelFamily | `ModelFamilyV1` | A / B |
| CheckpointArtifact | `CheckpointArtifactV1` | B |
| DatasetVersion | `DatasetV1`, `DatasetVersionV1` | A / C / B |
| AttackRecipe | `AttackRecipeV1` | A / C |
| BenchmarkRun | `BenchmarkRunV1` | A / C |
| Job | `JobV1`, `JobProgressEventV1` | B |
| TrainingRun | `TrainingRunV1` | C |
| DefenceRun | `DefenceRunV1` | C |
| ModelComparison | `ModelComparisonV1` | C |
| User | `UserV1` | D |
| Project | `ProjectV1`, `ProjectMembershipV1` | D |
| Artifact | `ArtifactV1`, `ArtifactReferenceV1` | B |

`ClassMappingV1` is also required: a scientific `BenchmarkRunV1` cannot be
created without it. Qualitative inference may set `scientific=False`.

## Lifecycles

```text
Artifact: UPLOADING -> QUARANTINED -> READY
                              \-> FAILED
Checkpoint: UPLOADING -> UPLOADED -> QUARANTINED -> INTEGRITY_VALIDATING
            -> SANDBOX_LOADING -> METADATA_EXTRACTING -> SMOKE_TESTING -> READY
Job: QUEUED -> RUNNING -> COMPLETED | FAILED | CANCELLED
Training: DRAFT -> VALIDATING -> ESTIMATING -> QUEUED -> PREPARING_DATA
          -> TRAINING -> VALIDATING_CHECKPOINT -> EXPORTING
          -> REGISTERING_MODEL -> COMPLETED
Defence: DRAFT -> GENERATING_DATA -> TRAINING -> EVALUATING -> COMPLETED
```

Only a sandbox validation worker may put a checkpoint into `READY`. A client
cannot set a resource state through a request payload.

## Platform API V1

All routes are mounted under `/api/v1`. Routes are project-scoped and use
plural nouns. Actions that cannot be expressed as a resource use a final verb.

```text
POST /projects/{project_id}/artifact-upload-sessions
POST /projects/{project_id}/artifact-upload-sessions/{session_id}/complete
GET  /projects/{project_id}/artifacts/{artifact_id}
POST /projects/{project_id}/artifacts/{artifact_id}/download-url

POST /projects/{project_id}/checkpoints
GET  /projects/{project_id}/checkpoints/{checkpoint_id}
POST /projects/{project_id}/checkpoints/imports/ultralytics
GET  /projects/{project_id}/checkpoints/{checkpoint_id}/validation

GET  /projects/{project_id}/jobs/{job_id}
GET  /projects/{project_id}/jobs/{job_id}/events
POST /projects/{project_id}/jobs/{job_id}/cancel
```

The feature routers owned by A and C use the same project scope:

```text
POST /projects/{project_id}/benchmark-runs
GET  /projects/{project_id}/benchmark-runs/{run_id}
POST /projects/{project_id}/training-runs
GET  /projects/{project_id}/training-runs/{run_id}
POST /projects/{project_id}/defence-runs
GET  /projects/{project_id}/model-comparisons/{comparison_id}
```

Long-running mutations return `202 Accepted` and a job resource. Errors use
`{"error": {"code": "...", "message": "...", "details": {}}}`.

## Migration and branch rules

- PostgreSQL is the production source of truth; SQLAlchemy and Alembic are
  introduced by the platform-foundation PR, not this contract PR.
- B serializes migration merges so `alembic heads` always has one head.
- Migrations are additive first; backfill and constraint tightening happen in
  later revisions. The application never runs migrations at startup.
- Platform work uses `feature/platform-deploy`; cross-domain shared contracts
  use the exception branch prefix `contract/`.

## Review gate

Before merging, each owner must explicitly approve its rows in the table above:

- A validates task/model-family IDs, 3D prediction contract names, and attack
  recipe compatibility.
- B validates artifact, checkpoint, job, provenance, and migration boundaries.
- C validates training/defence/comparison lineage and scientific mapping gate.
- D validates user, project membership, and project-scoped authorization.
