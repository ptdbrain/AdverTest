# P0-P3 Hardening Design

## Goal

Make the verified P0 security exploits impossible, make run exports truthful and usable, remove runtime dashboard fabrication, and make CI reject the defects that previously passed.

## Scope and ordering

1. P0 authorization and identity: sessions, legacy runs, exports, settings, and Google claims.
2. P1 reliability: PDF generation, PostgreSQL behavior, migration drift, and authenticated PDF download.
3. P2 evidence UX: dashboard values must come from completed run reports or remain `No data`.
4. P3 gates: negative exploit tests, real Alembic drift detection, and provenance validation.

CUDA PointPillars execution, Docker-host health, and full nuScenes NDS remain external validation gates. The code must surface those as waiting states; it must not claim them as complete on this CPU-only host.

## Authorization design

Every mutable or artifact-bearing request receives a required `project_id` and a JWT actor. A shared dependency loads the actor and verifies that actor is an active project member or owner. Repositories receive the already-authorized project identifier and include it in every query.

Legacy run records do not currently carry `project_id`; they will be treated as local-development runs and will require authentication but cannot be advertised as project-isolated evidence until migrated. New browser-facing report exports will use an authenticated fetch-to-Blob flow rather than unauthenticated `window.open`.

Settings profile updates are limited to display name and avatar. Role changes are removed from self-service input. W&B settings become actor-authenticated and project-scoped, or are rejected until project scope is supplied.

Google login accepts only the credential. It requires issuer, configured audience, expiry, subject, verified email, and derives identity fields only from verified claims.

## Export and provenance design

The PDF exporter must never invent protocol values, evaluator status, simulation state, or metrics. Missing values render as `No data`. It uses a Unicode-capable PDF text encoding strategy, safely handles nullable provenance, and paginates all attack cells. The report schema rejects an official-evidence state without a non-empty protocol hash.

## Data and CI design

Dashboard KPI and charts derive from completed run reports. Any absent metric remains `—` and charts render an explicit empty state. CI executes negative authorization and claim-validation tests, creates a fresh Alembic database then runs `alembic check`, and rejects runtime hard-coded dashboard metrics.

## Acceptance criteria

- Anonymous session CRUD, W&B read/write, run report/ZIP/PDF, and cross-project reads return 401/403.
- A researcher cannot change their own role; missing Google `exp` or `aud` returns 401.
- PDF endpoint returns a valid `%PDF-1.4` payload for reports with nullable and Unicode fields, without fabricated provenance.
- Fresh migration upgrade and `alembic check` both pass.
- Dashboard displays only calculated values or `—/No data`; no fixed ASR/robustness chart values remain in runtime source.
- Existing suites plus new negative tests pass; CUDA/Docker remain explicit external gates.
