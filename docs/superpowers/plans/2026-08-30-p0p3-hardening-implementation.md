# P0-P3 Hardening Implementation Plan

**Goal:** close the verified security, export/report truthfulness, workflow-cost, dashboard-evidence, and migration-drift gaps from the approved design.

**Scope:** backend FastAPI, persistence migrations/models, report/PDF export, frontend API/dashboard/attack flow, and regression tests. CUDA, Docker daemon, and external nuScenes assets remain environment gates and must be reported honestly.

## 1. Establish red regression tests

**Files:** tests/test_sessions_authorization.py, tests/test_settings_security.py, tests/test_google_auth_contract.py, tests/test_run_artifact_authorization.py, tests/test_pdf_export.py, tests/test_reports_provenance.py, frontend/src/**/*.test.*

- Add direct HTTP tests that show unauthenticated and cross-project session/run/report/PDF/ZIP requests return 401 or 403.
- Add tests that a researcher cannot set their own role, W&B settings require authentication, Google tokens without aud or exp are rejected, and profile fields cannot be sourced from the Google payload.
- Add a direct ZIP-response test that downloads, opens, and validates the archive; add PDF tests for nullable checkpoint values and Unicode text.
- Add report tests requiring provenance when a report claims non-simulated evidence, dashboard no-data tests, and attack submission tests requiring a model selection plus estimate confirmation.
- Run each targeted group and record its expected pre-fix failure.

## 2. Enforce actor and project authorization

**Files:** src/api/platform_dependencies.py, src/api/routers/sessions.py, src/api/routers/runs.py, src/api/jobs.py, src/api/workflow_store.py, tests above.

- Refactor project membership into an injected current-user dependency and require an explicit project_id on every evidence/session route.
- Require authentication on settings and legacy run artifact routes. Persist creator/project scope on newly created runs, reject legacy unscoped artifacts rather than silently exposing them.
- Preserve correct 401 for missing/invalid credentials and 403 for valid users without active membership.
- Run authorization regressions, then the router suite.

## 3. Harden SSO and mutable settings

**Files:** src/auth/contracts.py, src/auth/service.py, src/api/routers/settings.py, tests above.

- Remove Google identity/profile/role fallback fields from the input contract.
- Require configured Google client id, aud and exp claims, validate issuer/audience/expiry, and take profile identity only from verified claims.
- Limit self-service profile edits to display name and avatar. Keep role changes in an explicitly privileged admin path only.
- Run auth/settings regression tests.

## 4. Repair exports and report provenance

**Files:** src/evaluation/pdf_export.py, src/evaluation/models.py, src/api/routers/runs.py, tests/test_pdf_export.py, tests/test_reports_provenance.py.

- Generate valid PDF streams without encoding crashes; use a Unicode-safe conversion path and nullable-safe provenance fields.
- Label simulation and real evidence truthfully. A non-simulation report must include protocol/config/checkpoint/split provenance before it can claim scientific metrics.
- Ensure ZIP uses a concrete repository method and its HTTP result is a valid archive.
- Run export/provenance tests and inspect artifacts programmatically.

## 5. Align migration metadata and persistence behavior

**Files:** src/persistence/models.py, alembic/versions/20260830_0005_*.py, tests/test_p3_ci_quality_gates.py.

- Make ORM metadata accurately represent the migrated auth/project/session schema and add a forward migration for intentional indexes/schema changes.
- Make unsupported non-SQLite legacy store configuration fail clearly instead of creating an unrelated SQLite database.
- Run an isolated fresh upgrade and alembic check.

## 6. Remove fabricated runtime UI values and gate expensive work

**Files:** frontend/src/components/dashboard/DashboardView.jsx, frontend/src/components/charts/ComparisonBarChart.jsx, frontend/src/lib/api.js, frontend/src/app/experiments/[id]/attack/page.jsx, relevant Vitest tests.

- Render API-backed metric values only; show —/No data when evidence is absent.
- Require an explicitly selected backend model. Fetch an estimate and require a confirmation token before creating an expensive attack run.
- Preserve loading/error/empty states and update unit tests.
- Run Vitest and production build.

## 7. Full verification and handoff

- Run Ruff, targeted backend tests, full pytest with an isolated Windows base temp, frontend test suite, Next production build, fresh Alembic upgrade/check, and git diff checks.
- Document any blocked Docker/CUDA/nuScenes verification as environment constraints rather than claiming success.
- Review the final diff and report exact evidence and remaining external gates.
