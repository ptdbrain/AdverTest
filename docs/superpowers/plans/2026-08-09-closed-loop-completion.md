# AdverTest Closed-Loop Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the remaining durable generated-dataset, retraining, training, recovery/export, and frontend orchestration paths so the existing YOLO workflow can be operated end to end while SAM remains truthfully blocked.

**Architecture:** Reuse the existing immutable dataset, recipe, benchmark, training, and comparison domain contracts. Add a small durable workflow-job repository and injected background services so HTTP routes only validate, persist, enqueue, and stream events. Extend the current two-region frontend instead of replacing it again.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, SQLite, concurrent futures, pytest, Ruff, Next.js 16, React 19, Vitest, Testing Library, ESLint.

## Global Constraints

- Work only under `D:\Project\AIthucchien\P-195`; never read, inspect, or modify `D:\Project\AIthucchien\DAY01_2A202601138_PhanTrongDat`.
- Preserve untracked `PRDs.md`, `runs.zip`, caches, checkpoints, datasets, and all unrelated user files.
- Do not fabricate SAM masks, metrics, prompt results, or runnable checkpoints. Return `WAITING_FOR_ARTIFACTS` until Person C's complete contract is installed.
- Every new public Pydantic request model uses `ConfigDict(extra="forbid")`; every stochastic operation receives an explicit seed.
- Long-running creation/training operations return HTTP `202`, persist ordered events, expose cancellation, and never execute compute in a route handler.
- Scientific deltas are emitted only for paired runs. Metric values always include explicit units; ratio and percent fields are distinct.
- Generated data and checkpoints are stored below the configured application artifact root, never committed to Git.
- Use TDD for every behavior change: observe the focused test fail for the expected missing behavior before implementation.
- Run focused tests before each task commit. Record the final backend/frontend quality gate without converting skips into passes.

---

### Task 1: Durable generated-dataset jobs and APIs

**Files:**
- Create: `src/api/workflow_store.py`
- Create: `src/api/generated_dataset_service.py`
- Modify: `src/api/schemas.py`
- Modify: `src/api/routes.py`
- Modify: `src/config.py`
- Create: `tests/test_api/test_generated_datasets.py`
- Create: `tests/test_api/test_workflow_store.py`

**Interfaces:**
- Produce `WorkflowJobStore.create_job`, `append_event`, `request_cancel`, `complete_job`, `fail_job`, `get_job`, `events`, and `checkpoints` using durable SQLite tables.
- Produce `GeneratedDatasetService.enqueue(request) -> job_id`; the service resolves persisted DatasetVersion and AttackRecipe records, executes `AttackDatasetGenerator` outside the request thread, and persists the resulting descriptor, manifest, variants, validation result, and lineage.
- Expose `POST /api/v1/generated-datasets` as `202`, plus `GET /api/v1/generated-datasets/{id}`, `/manifest`, `/variants`, `/events`, and `POST /validate`.

- [ ] **Step 1: Write failing store and API tests**

```python
def test_workflow_events_remain_ordered_after_store_reopen(tmp_path):
    store = WorkflowJobStore(f"sqlite:///{tmp_path / 'jobs.db'}")
    job_id = store.create_job("generated_dataset", {"seed": 17})
    store.append_event(job_id, "VALIDATING", {"progress_ratio": 0.1})
    store.append_event(job_id, "GENERATING", {"progress_ratio": 0.5})
    reopened = WorkflowJobStore(f"sqlite:///{tmp_path / 'jobs.db'}")
    assert [event["sequence"] for event in reopened.events(job_id)] == [0, 1, 2]

@pytest.mark.asyncio
async def test_generated_dataset_creation_is_queued_and_exposes_manifest(client, generated_job_fixture):
    created = await client.post("/api/v1/generated-datasets", json=generated_job_fixture.request)
    assert created.status_code == 202
    item = await generated_job_fixture.wait_terminal(created.json()["id"])
    assert item["status"] == "COMPLETED"
    assert (await client.get(f"/api/v1/generated-datasets/{item['id']}/manifest")).status_code == 200
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api/test_workflow_store.py tests/test_api/test_generated_datasets.py -q`

Expected: FAIL because the workflow store/service and routes do not exist.

- [ ] **Step 3: Implement the minimal durable job path**

Use one SQLite transaction for each state transition and event append. Allocate event sequence numbers with `MAX(sequence) + 1` inside the same transaction. Validate source dataset readiness, recipe compatibility, artifact-root containment, and cancellation before each generation cell. Persist only JSON metadata and artifact-relative paths in SQLite.

- [ ] **Step 4: Verify focused behavior and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api/test_workflow_store.py tests/test_api/test_generated_datasets.py tests/test_pipeline/test_generator.py -q`

Commit: `feat: expose durable generated dataset jobs`

---

### Task 2: Retraining backlog and queued training orchestration

**Files:**
- Create: `src/api/training_service.py`
- Modify: `src/api/workflow_store.py`
- Modify: `src/api/schemas.py`
- Modify: `src/api/routes.py`
- Modify: `src/models/catalog.py`
- Create: `tests/test_api/test_retraining_backlogs.py`
- Create: `tests/test_api/test_training_runs.py`

**Interfaces:**
- Expose `POST/GET /api/v1/retraining-backlogs`, `POST /{id}/items`, and `POST /{id}/approve`.
- Expose `POST /api/v1/training-runs/estimate`, `POST /api/v1/training-runs` as `202`, `GET /api/v1/training-runs`, `GET /{id}`, `POST /{id}/cancel`, `GET /{id}/checkpoints`, and `GET /{id}/events/ws`.
- `TrainingJobService` resolves approved backlog, DefenseProfile, DatasetVersion, split manifest, generated sources, and trainer. It builds `TrainingDatasetManifest`, invokes `ComputeWorker` in a background executor, persists every `ProgressEvent`, persists checkpoint metadata, and registers a child ModelVersion only after checkpoint hash/load validation succeeds.

- [ ] **Step 1: Write failing backlog and training API tests**

```python
@pytest.mark.asyncio
async def test_backlog_requires_items_before_approval(client):
    created = await client.post("/api/v1/retraining-backlogs", json={"name": "fog failures"})
    approved = await client.post(f"/api/v1/retraining-backlogs/{created.json()['id']}/approve")
    assert approved.status_code == 409
    assert approved.json()["detail"]["code"] == "BACKLOG_EMPTY"

@pytest.mark.asyncio
async def test_training_run_persists_events_checkpoint_and_child_lineage(client, training_fixture):
    response = await client.post("/api/v1/training-runs", json=training_fixture.request)
    assert response.status_code == 202
    run = await training_fixture.wait_terminal(response.json()["id"])
    assert run["status"] == "COMPLETED"
    checkpoints = (await client.get(f"/api/v1/training-runs/{run['id']}/checkpoints")).json()
    assert checkpoints[0]["sha256"] == training_fixture.checkpoint_sha256
    assert run["model_version"]["parent_id"] == training_fixture.parent_version_id
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api/test_retraining_backlogs.py tests/test_api/test_training_runs.py -q`

Expected: FAIL because backlog and training routes/services do not exist.

- [ ] **Step 3: Implement state machines and worker bridge**

Backlogs transition `DRAFT -> APPROVED`; approval is immutable and rejects duplicate or unknown failure IDs. Training transitions follow the existing `TrainingStateMachine`. Return `TRAINER_NOT_AVAILABLE`, `BACKLOG_NOT_APPROVED`, `TRAINING_DATASET_INVALID`, and `WAITING_FOR_ARTIFACTS` before enqueue where applicable. Cancellation is cooperative and produces terminal `CANCELLED` without registering a model version.

- [ ] **Step 4: Verify focused behavior and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_api/test_retraining_backlogs.py tests/test_api/test_training_runs.py tests/test_training_orchestration.py tests/test_training_dataset_builder.py -q`

Commit: `feat: orchestrate durable defense training jobs`

---

### Task 3: Rich paired comparison, recovery, failures, and exports

**Files:**
- Create: `src/evaluation/export.py`
- Modify: `src/evaluation/model_comparison.py`
- Modify: `src/evaluation/recovery_metrics.py`
- Modify: `src/api/schemas.py`
- Modify: `src/api/routes.py`
- Create: `tests/test_evaluation/test_export.py`
- Create: `tests/test_api/test_comparison_reports.py`

**Interfaces:**
- Persist `ModelComparison` with pairedness, incompatibilities, metric envelopes, clean trade-off, failure deltas, lineage, recovery report, gate outcome, and residual-risk narrative.
- Expose `GET /api/v1/model-comparisons/{id}/metric-deltas`, `/failures`, `/recovery-report`, and `/export?format=json|csv|html`.
- `export_comparison(comparison, format) -> ExportArtifact` returns media type, filename, SHA-256, and content; HTML is self-contained and escapes all user-controlled text.

- [ ] **Step 1: Write failing comparison/export tests**

```python
@pytest.mark.asyncio
async def test_paired_comparison_exposes_recovery_units_and_all_export_formats(client, paired_runs):
    created = await client.post("/api/v1/model-comparisons", json=paired_runs)
    assert created.status_code == 201
    recovery = (await client.get(f"/api/v1/model-comparisons/{created.json()['id']}/recovery-report")).json()
    assert recovery["recovery_rate"]["unit"] == "percent"
    assert "ratio_value" in recovery["recovery_rate"]
    for format_name in ("json", "csv", "html"):
        exported = await client.get(f"/api/v1/model-comparisons/{created.json()['id']}/export?format={format_name}")
        assert exported.status_code == 200

@pytest.mark.asyncio
async def test_unpaired_comparison_never_exposes_scientific_deltas(client, unpaired_runs):
    created = await client.post("/api/v1/model-comparisons", json=unpaired_runs)
    assert created.json()["paired"] is False
    assert (await client.get(f"/api/v1/model-comparisons/{created.json()['id']}/metric-deltas")).json() == {}
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evaluation/test_export.py tests/test_api/test_comparison_reports.py -q`

Expected: FAIL because the detailed resources and export engine do not exist.

- [ ] **Step 3: Implement scientific envelopes and durable exports**

Use the locked protocol compatibility fields from the design spec. Recovery denominator zero returns an explicit undefined metric. Percent metrics store both `ratio_value` and `percent_value`; score metrics store `value` and `unit="score"`. Never infer segmentation values from detection reports.

- [ ] **Step 4: Verify focused behavior and commit**

Run: `.venv\Scripts\python.exe -m pytest tests/test_evaluation tests/test_api/test_comparison_reports.py tests/test_api/test_product_workflow.py -q`

Commit: `feat: publish paired recovery comparison exports`

---

### Task 4: Complete the frontend closed-loop orchestration

**Files:**
- Modify: `frontend/src/lib/api.js`
- Modify: `frontend/src/hooks/useAdverTest.js`
- Modify: `frontend/src/app/page.js`
- Modify: `frontend/src/app/globals.css`
- Modify: `frontend/src/components/ConfigPanel.jsx`
- Modify: `frontend/src/components/Workspace.jsx`
- Create: `frontend/src/components/AdvancedMetricsDrawer.jsx`
- Create: `frontend/src/components/ModelComparison.jsx`
- Create: `frontend/src/components/TrainingWorkflow.jsx`
- Create: `frontend/src/components/__tests__/closed-loop.test.jsx`
- Modify: `frontend/src/components/__tests__/workspace.test.jsx`

**Interfaces:**
- `useAdverTest` owns dataset version, recipe, generated-dataset job, benchmark protocol/run, selected failures, backlog, defense profile, training run, child model version, comparison, export, ordered events, loading/error/cancel states.
- It exposes `importDataset`, `createRecipe`, `generateDataset`, `startBenchmark`, `createBacklog`, `approveBacklog`, `startTraining`, `cancelTraining`, `rerunBenchmark`, `createComparison`, and `exportComparison`.
- The UI keeps the two-region layout. The left panel makes the next valid action explicit; the center keeps five evidence stages and adds progressive comparison/training detail without nested-card grids or modal-first flows.

- [ ] **Step 1: Read project design and Next.js references**

Read `PRODUCT.md`, `frontend/AGENTS.md`, and the installed Next.js App Router/client-component guides under `frontend/node_modules/next/dist/docs/`. Record any convention that affects client boundaries in the task report.

- [ ] **Step 2: Write failing user-flow tests**

```jsx
it("runs the closed loop from dataset through defended comparison", async () => {
  render(<HomePage />)
  await user.click(screen.getByRole("button", { name: "Generate attacked dataset" }))
  await user.click(await screen.findByRole("button", { name: "Create defense backlog" }))
  await user.click(screen.getByRole("button", { name: "Start training" }))
  expect(await screen.findByText("Checkpoint verified")).toBeVisible()
  await user.click(screen.getByRole("button", { name: "Re-run locked benchmark" }))
  expect(await screen.findByRole("region", { name: "Recovery report" })).toBeVisible()
})

it("shows pairedness before metric deltas and keeps SAM blocked", async () => {
  render(<HomePage />)
  await user.selectOptions(screen.getByLabelText("Perception mode"), "segmentation")
  expect(screen.getByText("Waiting for Person C artefacts")).toBeVisible()
  expect(screen.queryByRole("button", { name: "Start training" })).toBeDisabled()
})
```

- [ ] **Step 3: Run tests and verify RED**

Run: `npm --prefix frontend run test -- --run`

Expected: FAIL because the training/comparison orchestration and components are absent.

- [ ] **Step 4: Implement the progressive workflow UI**

Use restrained tinted neutrals and one semantic accent. Status must include text and icon, not color alone. Every asynchronous action has loading, disabled, error, retry, and cancellation states. Render skeletons for initial resource loading. Comparison renders pairedness before deltas; export remains disabled when comparison is unpaired. Preserve keyboard navigation and WCAG 2.1 AA focus/contrast.

- [ ] **Step 5: Verify focused frontend behavior and commit**

Run: `npm --prefix frontend run test -- --run; npm --prefix frontend run lint; npm --prefix frontend run build`

Commit: `feat: connect the full robustness recovery workspace`

---

### Task 5: End-to-end acceptance, quality gates, and operations evidence

**Files:**
- Create: `tests/integration/test_closed_loop_e2e.py`
- Create: `tests/integration/test_training_cancel_resume.py`
- Create: `tests/integration/test_sam_waiting_gate.py`
- Modify: `README.md`
- Create: `docs/YOLO_PIPELINE_OPERATIONS.md`
- Create: `docs/INTEGRATION_COMPLETION_MATRIX.md`
- Modify: `.gitignore`

**Interfaces:**
- Provide a deterministic CPU fake trainer/model fixture that exercises the real HTTP, SQLite, background worker, event, cancellation, checkpoint, model-lineage, comparison, recovery, and export boundaries.
- Keep optional real B-checkpoint tests marked `models`; skip with the exact missing dependency or runtime reason.

- [ ] **Step 1: Write failing E2E tests**

```python
@pytest.mark.asyncio
async def test_complete_closed_loop_persists_recovery_after_app_restart(closed_loop_client):
    ids = await closed_loop_client.run_dataset_to_recovery()
    restarted = closed_loop_client.restart()
    comparison = await restarted.get(f"/api/v1/model-comparisons/{ids.comparison_id}")
    assert comparison.status_code == 200
    assert comparison.json()["paired"] is True

@pytest.mark.asyncio
async def test_cancelled_training_keeps_events_but_registers_no_child_model(cancel_client):
    run_id = await cancel_client.start_and_cancel_training()
    run = (await cancel_client.get(f"/api/v1/training-runs/{run_id}")).json()
    assert run["status"] == "CANCELLED"
    assert run["model_version"] is None
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv\Scripts\python.exe -m pytest tests/integration/test_closed_loop_e2e.py tests/integration/test_training_cancel_resume.py tests/integration/test_sam_waiting_gate.py -q`

- [ ] **Step 3: Complete acceptance harness and documentation**

Document exact startup, dataset import, recipe/generation, benchmark, backlog/defense, training, re-benchmark, comparison, export, and SAM handoff commands. The completion matrix records command, result, timestamp, and whether evidence is CPU contract, real model, skipped, or blocked by Person C.

- [ ] **Step 4: Run final quality gates**

Run:

```powershell
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pytest tests/integration -q
.venv\Scripts\python.exe -m pytest -m models -q
npm --prefix frontend run test -- --run
npm --prefix frontend run lint
npm --prefix frontend run build
```

Resolve the known pre-existing Ruff import/unused issues and the React `set-state-in-effect` lint failure as part of this quality-gate task, with focused regression checks. Do not inspect DAY01 as a boundary check; use `git status --short` only inside this P-195 worktree.

- [ ] **Step 5: Verify Git boundaries and commit**

Confirm no weights, `runs/`, archives, databases, uploads, caches, `PRDs.md`, `runs.zip`, or unrelated files are staged. Commit only E2E, docs, ignore rules, and quality-gate fixes.

Commit: `docs: verify the complete robustness closed loop`
