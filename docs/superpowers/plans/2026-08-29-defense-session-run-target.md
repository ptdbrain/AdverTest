# Defense Session and Attack Run Targeting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require a real experiment session and one attack run before configuring or evaluating a defense, and derive the defended model, dataset, attack, severity, and locked protocol from that evidence.

**Architecture:** `DefensePage` loads sessions and owns selection and API state. A focused `DefenseTargetSelector` renders the dependent session/run controls and provenance summary. The existing checkpoint and defense-run APIs upload a candidate and start evaluation against the selected run's `backend_run_id`; mock comparison data is removed.

**Tech Stack:** Next.js 16.3, React 19, Tailwind CSS, Vitest, Testing Library, existing AdverTest REST client.

## Global Constraints

- Do not modify, stage, stash, reset, or commit unrelated dirty-worktree changes.
- Never infer successful defense evaluation from timers or `DEFENSE_COMPARISON` mock data.
- A display session run ID is not a backend baseline ID; evaluation must use non-empty `backend_run_id`.
- Changing sessions clears the selected run and all downstream checkpoint/evaluation state.
- Preserve responsive single-column behavior and visible labels.

---

### Task 1: Dependent Session and Attack Run Selector

**Files:**
- Create: `frontend/src/components/DefenseTargetSelector.jsx`
- Create: `frontend/src/components/__tests__/defense-target-selector.test.jsx`

**Interfaces:**
- Consumes: `sessions: SessionRecord[]`, `selectedSessionId: string`, `selectedRunId: string`, `loading: boolean`, `error: string`, `onSessionChange(id)`, `onRunChange(id)`, `onRetry()`.
- Produces: accessible session/run selects and read-only provenance based only on the selected session and run.

- [ ] **Step 1: Write failing tests for dependent selection and provenance**

```jsx
render(<DefenseTargetSelector sessions={sessions} selectedSessionId="" selectedRunId="" {...handlers} />);
expect(screen.getByRole("combobox", { name: /phiên thử nghiệm/i })).toBeInTheDocument();
expect(screen.getByText(/chọn phiên và attack run/i)).toBeInTheDocument();

await user.selectOptions(screen.getByRole("combobox", { name: /attack run/i }), "run-visible-1");
expect(screen.getByText("YOLO11s Production")).toBeInTheDocument();
expect(screen.getByText(/Depth Fog/i)).toBeInTheDocument();
expect(screen.getByText("45.1%")).toBeInTheDocument();
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `npm test -- --run src/components/__tests__/defense-target-selector.test.jsx`

Expected: FAIL because `DefenseTargetSelector.jsx` does not exist.

- [ ] **Step 3: Implement the minimal selector and summary**

```jsx
const selectedSession = sessions.find((item) => item.id === selectedSessionId) ?? null;
const selectedRun = selectedSession?.runs?.find((item) => item.id === selectedRunId) ?? null;

<select aria-label="Phiên thử nghiệm" value={selectedSessionId} onChange={(event) => onSessionChange(event.target.value)} />
<select aria-label="Attack run" disabled={!selectedSession} value={selectedRunId} onChange={(event) => onRunChange(event.target.value)} />
```

The provenance block displays `model_name`, `dataset_name`, `attack_name`, `severity`, clean and attacked AP/mIoU, `map_drop_pct`/`miou_drop_pct`, `backend_run_id`, `seed`, and `run_config_hash`. Missing values render `Không có dữ liệu`, not defaults.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `npm test -- --run src/components/__tests__/defense-target-selector.test.jsx`

Expected: all selector tests pass.

---

### Task 2: Wire Real Session/Run Context into Defense Page

**Files:**
- Modify: `frontend/src/app/defense/page.jsx`
- Modify: `frontend/src/lib/api.js`
- Create: `frontend/src/components/__tests__/defense.page.test.jsx`

**Interfaces:**
- Consumes: `listSessions()`, `getRun(runId)`, `uploadCheckpoint(file, metadata, onProgress)`, `getCheckpoint(id)`, `createDefenceRun(baselineRunId, checkpointId)`.
- Produces: `getRunDefenceCandidates(runId)` helper and a defense page whose model/dataset/recipe are derived from the selected run.

- [ ] **Step 1: Write failing page tests**

```jsx
expect(screen.getByRole("button", { name: /chạy đánh giá/i })).toBeDisabled();
await user.selectOptions(screen.getByRole("combobox", { name: /phiên thử nghiệm/i }), "session-a");
await user.selectOptions(screen.getByRole("combobox", { name: /attack run/i }), "display-run-a");
expect(screen.getByDisplayValue("YOLO11s Production")).toHaveAttribute("readOnly");
expect(screen.queryByText("Phục hồi +68.4%")).not.toBeInTheDocument();
```

Add cases proving that switching to `session-b` clears the selected run and that a run without `backend_run_id` shows `Không có baseline backend có thể đối chiếu`.

- [ ] **Step 2: Run the page test and verify RED**

Run: `npm test -- --run src/components/__tests__/defense.page.test.jsx`

Expected: FAIL because the current page has independent mock defaults and no session selector.

- [ ] **Step 3: Add the missing API helper**

```js
export function getRunDefenceCandidates(runId) {
  return apiFetch(`/api/v1/runs/${encodeURIComponent(runId)}/defence-candidates`);
}
```

- [ ] **Step 4: Load sessions and reset dependent state**

```jsx
const handleSessionChange = (id) => {
  setSelectedSessionId(id);
  setSelectedRunId("");
  setBaselineRun(null);
  setCandidateId("");
  setEvaluationJob(null);
  setEvaluationError("");
};
```

On mount, call `listSessions()` and expose loading, retry, empty, and error states. Do not auto-select a session or run.

- [ ] **Step 5: Resolve the real backend baseline**

When a selected run has `backend_run_id`, call `getRun(backend_run_id)` and `getRunDefenceCandidates(backend_run_id)`. Derive:

```jsx
const lockedModel = baselineRun?.config?.checkpoint_id ?? baselineRun?.config?.model_version_id ?? selectedSession?.model_id ?? "";
const lockedDataset = baselineRun?.config?.dataset ?? selectedSession?.dataset_id ?? "";
const lockedRecipe = selectedRun?.attack_components?.length
  ? selectedRun.attack_components.join(",")
  : selectedRun?.attack_type ?? "";
```

Model and recipe inputs are read-only. Training controls, command copy/download, checkpoint upload, and evaluation remain disabled until a run is selected; paired evaluation additionally requires `backend_run_id`.

- [ ] **Step 6: Replace fake upload/evaluation with backend state**

Upload the file with task/family/parent metadata from `baselineRun.config`, poll `getCheckpoint()` until `READY`/`REJECTED`, then set the returned checkpoint ID. Start evaluation with:

```jsx
const job = await createDefenceRun(selectedRun.backend_run_id, candidateId);
setEvaluationJob(job);
```

Render the real job ID/status. Remove `setTimeout`, `evaluationDone`, the `DEFENSE_COMPARISON` import, the hard-coded KITTI/seed/Depth Fog + PGD banner, and recovery table.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run: `npm test -- --run src/components/__tests__/defense-target-selector.test.jsx src/components/__tests__/defense.page.test.jsx`

Expected: all focused tests pass with no unhandled promise warnings.

---

### Task 3: Regression, Lint, and Production Build Verification

**Files:**
- Verify only; do not edit unrelated files.

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: evidence for frontend behavior and reports any pre-existing build blocker separately.

- [ ] **Step 1: Run full frontend tests**

Run: `npm test -- --run`

Expected: all frontend tests pass.

- [ ] **Step 2: Run scoped ESLint**

Run: `npx eslint src/app/defense/page.jsx src/components/DefenseTargetSelector.jsx src/components/__tests__/defense-target-selector.test.jsx src/components/__tests__/defense.page.test.jsx src/lib/api.js`

Expected: no new errors from changed files; pre-existing warnings in `api.js` are reported without unrelated cleanup.

- [ ] **Step 3: Run production build**

Run: `npm run build -- --webpack`

Expected: record the actual exit status. If the known `/attack` Suspense prerender blocker remains, report it as pre-existing and do not claim a successful production build.

- [ ] **Step 4: Inspect final diff and workspace state**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors introduced; all unrelated user changes remain untouched. Do not stage or commit.
