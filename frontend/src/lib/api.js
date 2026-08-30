const BUILD_TIME_API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// `NEXT_PUBLIC_*` is normally inlined by Next.js at build time.  The Render
// Docker service is configured at runtime, so let its entrypoint provide an
// optional public runtime value instead.  This also keeps local development
// working without a runtime-config.js file.
export function getApiBase() {
  if (typeof window !== "undefined") {
    const runtimeBase = window.__ADVERTEST_RUNTIME_CONFIG__?.apiUrl;
    if (runtimeBase) return runtimeBase.replace(/\/$/, "");
  }
  return BUILD_TIME_API_BASE.replace(/\/$/, "");
}

export function artifactUrl(pathValue) {
  if (!pathValue) return "";
  if (/^https?:\/\//i.test(pathValue)) return pathValue;
  const base = getApiBase();
  const normalized = pathValue.startsWith("/") ? pathValue : `/${pathValue}`;
  return `${base}${normalized}`;
}

export async function apiFetch(path, options = {}) {
  const url = `${getApiBase()}${path}`;
  const headers = { "Content-Type": "application/json", ...options.headers };

  const res = await fetch(url, {
    ...options,
    credentials: "include",
    headers,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : body.detail
          ? JSON.stringify(body.detail)
          : `API error ${res.status}`;
    throw new Error(detail);
  }
  return res.json();
}

export function loginUser(credentialsOrEmail, passwordArg) {
  const payload =
    typeof credentialsOrEmail === "object"
      ? credentialsOrEmail
      : { email: credentialsOrEmail, password: passwordArg };
  return apiFetch("/api/v1/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function registerUser(payloadOrEmail, passwordArg, displayNameArg) {
  const payload =
    typeof payloadOrEmail === "object"
      ? payloadOrEmail
      : { email: payloadOrEmail, password: passwordArg, display_name: displayNameArg };
  return apiFetch("/api/v1/auth/register", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getCurrentUser() {
  return apiFetch("/api/v1/auth/me");
}

export function logoutUser() {
  return apiFetch("/api/v1/auth/logout", { method: "POST" });
}

export function getCatalogAttacks(params = {}) {
  const qs = new URLSearchParams(params).toString();
  return apiFetch(`/api/v1/catalog/attacks${qs ? `?${qs}` : ""}`);
}

export function getCatalogModels() {
  return apiFetch("/api/v1/catalog/models");
}

export function getCatalogDatasets(params = {}) {
  const qs = new URLSearchParams(params).toString();
  return apiFetch(`/api/v1/catalog/datasets${qs ? `?${qs}` : ""}`);
}

export function getModelVersions() { return apiFetch("/api/v1/model-versions"); }
export function getPerceptionModes() { return apiFetch("/api/v1/perception-modes"); }
export function getModelFamilies(taskId) { return apiFetch(`/api/v1/model-families?task_id=${encodeURIComponent(taskId)}`); }
export function getBaseCheckpoints(taskId, familyId) { return apiFetch(`/api/v1/base-checkpoints?task_id=${encodeURIComponent(taskId)}&model_family_id=${encodeURIComponent(familyId)}`); }
export function getDefenceCheckpoints(taskId) { return apiFetch(`/api/v1/defence-checkpoints?task_id=${encodeURIComponent(taskId)}`); }
export function getRunDefenceCandidates(runId) { return apiFetch(`/api/v1/runs/${encodeURIComponent(runId)}/defence-candidates`); }
export function createDefenceRun(baselineRunId, checkpointId) {
  return apiFetch("/api/v1/defence-runs", { method: "POST", body: JSON.stringify({ baseline_run_id: baselineRunId, checkpoint_id: checkpointId }) });
}

export function estimateRun(config) {
  return apiFetch("/api/v1/runs/estimate", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export function preflightRun(config) {
  return apiFetch("/api/v1/runs/preflight", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export function createRun(config) {
  return apiFetch("/api/v1/runs", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export function createInferenceExperiment(payload) {
  return apiFetch("/api/v1/inference-experiments", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function listRuns() {
  return apiFetch("/api/v1/runs");
}

export function getRun(runId) {
  return apiFetch(`/api/v1/runs/${runId}`);
}

export function getRunReport(runId) {
  return apiFetch(`/api/v1/runs/${runId}/report`);
}

export function getRunSamples(runId, params = {}) {
  const qs = new URLSearchParams(params).toString();
  return apiFetch(`/api/v1/runs/${runId}/samples${qs ? `?${qs}` : ""}`);
}

export function getRunAnalyticsSummary(runId) {
  return apiFetch(`/api/v1/analytics/runs/${encodeURIComponent(runId)}/summary`);
}

export function getRunAnalyticsAttacks(runId) {
  return apiFetch(`/api/v1/analytics/runs/${encodeURIComponent(runId)}/attacks`);
}

export function getRunAnalyticsClasses(runId) {
  return apiFetch(`/api/v1/analytics/runs/${encodeURIComponent(runId)}/classes`);
}

export function getRunAnalyticsDistance(runId) {
  return apiFetch(`/api/v1/analytics/runs/${encodeURIComponent(runId)}/distance`);
}

export function cancelRun(runId) {
  return apiFetch(`/api/v1/runs/${runId}/cancel`, { method: "POST" });
}

export function connectRunWebSocket(runId, onEvent) {
  const wsBase = getApiBase().replace(/^http/, "ws");
  const ws = new WebSocket(`${wsBase}/api/v1/runs/${runId}/events/ws`);
  ws.onmessage = (e) => {
    try {
      onEvent(JSON.parse(e.data));
    } catch {}
  };
  return ws;
}

export function getReviews(params = {}) {
  const qs = new URLSearchParams(params).toString();
  return apiFetch(`/api/v1/reviews${qs ? `?${qs}` : ""}`);
}

export function resolveReview(reviewId, decision, decisionNote, resolvedBy, batchClusterId = null) {
  const payload = {
    decision,
    decision_note: decisionNote,
    resolved_by: resolvedBy,
  };
  if (batchClusterId) {
    payload.batch_cluster_id = batchClusterId;
  }
  return apiFetch(`/api/v1/reviews/${reviewId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function triggerAutoFlag(runId, threshold = 30) {
  return apiFetch(`/api/v1/runs/${runId}/flag-reviews?threshold=${threshold}`, {
    method: "POST"
  });
}

/* ---- Risk Rubric & HITL Triage ---- */
export function getRiskRubric() {
  return apiFetch("/api/v1/risk-rubric");
}

export function assessReviewRisk(reviewId) {
  return apiFetch(`/api/v1/risk-rubric/assess?review_id=${encodeURIComponent(reviewId)}`, {
    method: "POST",
  });
}

export function getRiskSessionSummary(params = {}) {
  const qs = new URLSearchParams(params).toString();
  return apiFetch(`/api/v1/risk-rubric/session-summary${qs ? `?${qs}` : ""}`);
}

export function autoGroupFailureClusters(runId = null) {
  const url = runId
    ? `/api/v1/failure-clusters/auto-group?run_id=${encodeURIComponent(runId)}`
    : "/api/v1/failure-clusters/auto-group";
  return apiFetch(url, { method: "POST" });
}

export function createRetrainingBacklog(name) {
  return apiFetch("/api/v1/retraining-backlogs", {
    method: "POST",
    body: JSON.stringify({ name }),
  });
}

export function addRetrainingBacklogItem(backlogId, failureId) {
  return apiFetch(`/api/v1/retraining-backlogs/${backlogId}/items`, {
    method: "POST",
    body: JSON.stringify({ failure_id: failureId }),
  });
}

export function approveRetrainingBacklog(backlogId) {
  return apiFetch(`/api/v1/retraining-backlogs/${backlogId}/approve`, {
    method: "POST",
  });
}

// ── Session Management ──────────────────────────────────
export function listSessions() {
  return apiFetch("/api/v1/sessions");
}

export function getSession(sessionId) {
  return apiFetch(`/api/v1/sessions/${encodeURIComponent(sessionId)}`);
}

export function createOrUpdateSession(sessionData) {
  return apiFetch("/api/v1/sessions", {
    method: "POST",
    body: JSON.stringify(sessionData),
  });
}

export function addRunToSession(sessionId, runRecord) {
  return apiFetch(
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/runs`,
    { method: "POST", body: JSON.stringify(runRecord) }
  );
}

export function deleteRunFromSession(sessionId, runId) {
  return apiFetch(
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/runs/${encodeURIComponent(runId)}`,
    { method: "DELETE" }
  );
}

export function updateRunNote(sessionId, runId, note) {
  return apiFetch(
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/runs/${encodeURIComponent(runId)}/note`,
    { method: "PATCH", body: JSON.stringify({ note }) }
  );
}

export function endSession(sessionId) {
  return apiFetch(
    `/api/v1/sessions/${encodeURIComponent(sessionId)}/end`,
    { method: "POST" }
  );
}

/* ---- Recipe API ---- */
export function getRecipePresets() {
  return apiFetch("/api/v1/catalog/recipes/presets");
}

export function randomizeRecipe(nSteps = 3, group = null, seed = 42) {
  return apiFetch("/api/v1/attack-recipes/randomize", {
    method: "POST",
    body: JSON.stringify({ n_steps: nSteps, group, seed }),
  });
}

export function sweepRecipe(attackId, severityRange = [1, 2, 3, 4, 5]) {
  return apiFetch("/api/v1/attack-recipes/sweep", {
    method: "POST",
    body: JSON.stringify({ attack_id: attackId, severity_range: severityRange }),
  });
}

export function previewRecipe(payload) {
  return apiFetch("/api/v1/attack-recipes/preview", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function validateRecipe(recipePayload) {
  return apiFetch("/api/v1/attack-recipes/validate", {
    method: "POST",
    body: JSON.stringify(recipePayload),
  });
}

/* ---- Failure Clusters ---- */
export function getFailureClusters() {
  return apiFetch("/api/v1/failure-clusters");
}

export function createFailureCluster(name, memberIds, defenseProfileId = null) {
  return apiFetch("/api/v1/failure-clusters", {
    method: "POST",
    body: JSON.stringify({ name, member_ids: memberIds, defense_profile_id: defenseProfileId }),
  });
}

export function getFailureCluster(clusterId) {
  return apiFetch(`/api/v1/failure-clusters/${clusterId}`);
}

/* ---- Model Lineage & Gate Evidence ---- */
export function getModelVersionLineage(versionId) {
  return apiFetch(`/api/v1/model-versions/${versionId}/lineage`);
}

export function getModelVersionBenchmarkHistory(versionId) {
  return apiFetch(`/api/v1/model-versions/${versionId}/benchmark-history`);
}

export function getModelVersionGateEvidence(versionId) {
  return apiFetch(`/api/v1/model-versions/${versionId}/gate-evidence`);
}

/* ---- Model Comparisons ---- */
export function createModelComparison(baselineRunId, candidateRunId) {
  return apiFetch("/api/v1/model-comparisons", {
    method: "POST",
    body: JSON.stringify({ baseline_run_id: baselineRunId, candidate_run_id: candidateRunId }),
  });
}

export function getModelComparison(comparisonId) {
  return apiFetch(`/api/v1/model-comparisons/${comparisonId}`);
}

export function getModelComparisonFailures(comparisonId) {
  return apiFetch(`/api/v1/model-comparisons/${comparisonId}/failures`);
}

export function exportModelComparison(comparisonId, format = "json") {
  return apiFetch(`/api/v1/model-comparisons/${comparisonId}/export?format=${format}`);
}

/* ---- Closed-Loop ---- */
export function startClosedLoop(runId) {
  return apiFetch("/api/v1/closed-loop/start", {
    method: "POST",
    body: JSON.stringify({ run_id: runId }),
  });
}

export function advanceClosedLoop(loopId, target, artifactId) {
  return apiFetch(`/api/v1/closed-loop/${loopId}/advance`, {
    method: "POST",
    body: JSON.stringify({ target, artifact_id: artifactId }),
  });
}

export function getClosedLoop(loopId) {
  return apiFetch(`/api/v1/closed-loop/${loopId}`);
}

/* ---- Defense Profiles ---- */
export function createDefenseProfile(profile) {
  return apiFetch("/api/v1/defense-profiles", {
    method: "POST",
    body: JSON.stringify(profile),
  });
}

export function getDefenseProfile(profileId) {
  return apiFetch(`/api/v1/defense-profiles/${profileId}`);
}

/* ---- Status / Evidence ---- */
export function getStatusEvidence() {
  return apiFetch("/api/v1/status/evidence");
}

/* ---- Upload & Dataset Import ---- */
export function createUploadBatch(payload) {
  return apiFetch("/api/v1/uploads/batches", { method: "POST", body: JSON.stringify(payload) });
}

export function getUploadBatch(batchId) { return apiFetch(`/api/v1/uploads/batches/${encodeURIComponent(batchId)}`); }

export function saveBatchAnnotation(batchId, sampleId, payload) {
  return apiFetch(`/api/v1/uploads/batches/${encodeURIComponent(batchId)}/annotations/${encodeURIComponent(sampleId)}`, {
    method: "PUT", body: JSON.stringify(payload),
  });
}

export function finalizeUploadBatch(batchId) {
  return apiFetch(`/api/v1/uploads/batches/${encodeURIComponent(batchId)}/finalize`, { method: "POST" });
}

export function uploadImage(file, batchId = null, taskId = "detection2d", sampleId = null, onProgress = null) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${getApiBase()}/api/v1/uploads/images`);
    request.setRequestHeader("Content-Type", "application/octet-stream");
    request.setRequestHeader("x-filename", file.name);
    request.setRequestHeader("x-task-id", taskId);
    if (batchId) request.setRequestHeader("x-upload-batch-id", batchId);
    if (sampleId) request.setRequestHeader("x-sample-id", sampleId);
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress?.(event.loaded / event.total);
    };
    request.onerror = () => reject(new Error("Network error while uploading image."));
    request.onload = () => {
      let payload = {};
      try { payload = JSON.parse(request.responseText || "{}"); } catch {}
      if (request.status >= 200 && request.status < 300) resolve(payload);
      else reject(new Error(typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail || `Upload failed with status ${request.status}`)));
    };
    request.send(file);
  });
}

export function uploadCheckpoint(file, { taskId, familyId, displayName, role = "base", parentCheckpointId = null, trainingDatasetVersionId = null } = {}, onProgress = null) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${getApiBase()}/api/v1/checkpoints/uploads`);
    request.setRequestHeader("Content-Type", "application/octet-stream");
    request.setRequestHeader("x-filename", file.name);
    request.setRequestHeader("x-task-id", taskId || "");
    request.setRequestHeader("x-model-family-id", familyId || "");
    request.setRequestHeader("x-display-name", displayName || file.name);
    request.setRequestHeader("x-checkpoint-role", role);
    if (parentCheckpointId) request.setRequestHeader("x-parent-checkpoint-id", parentCheckpointId);
    if (trainingDatasetVersionId) request.setRequestHeader("x-training-dataset-version-id", trainingDatasetVersionId);
    request.upload.onprogress = (event) => { if (event.lengthComputable) onProgress?.(event.loaded / event.total); };
    request.onerror = () => reject(new Error("Network error while uploading checkpoint."));
    request.onload = () => {
      let payload = {};
      try { payload = JSON.parse(request.responseText || "{}"); } catch {}
      if (request.status >= 200 && request.status < 300) resolve(payload);
      else reject(new Error(typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail || `Upload failed with status ${request.status}`)));
    };
    request.send(file);
  });
}

export function getCheckpoint(checkpointId) { return apiFetch(`/api/v1/checkpoints/${encodeURIComponent(checkpointId)}`); }
export function getCheckpointValidationEvents(checkpointId) { return apiFetch(`/api/v1/checkpoints/${encodeURIComponent(checkpointId)}/validation-events`); }

export function importFolderDataset({ root, name, logicalSourceId, inputFormat = "advertest", anonymizationManifest = "manifest.jsonl", maxSamples = 50, taskId = "detection2d" }) {
  return apiFetch("/api/v1/datasets/import", {
    method: "POST",
    body: JSON.stringify({
      root,
      name,
      logical_source_id: logicalSourceId,
      input_format: inputFormat,
      anonymization_manifest: anonymizationManifest,
      max_samples: maxSamples,
      task_id: taskId,
    }),
  });
}

export function startFolderDatasetImport({ root, name, logicalSourceId, inputFormat = "advertest", anonymizationManifest = "manifest.jsonl", maxSamples = 50, taskId = "detection2d" }) {
  return apiFetch("/api/v1/datasets/import-jobs", {
    method: "POST",
    body: JSON.stringify({
      root,
      name,
      logical_source_id: logicalSourceId,
      input_format: inputFormat,
      anonymization_manifest: anonymizationManifest,
      max_samples: maxSamples,
      task_id: taskId,
    }),
  });
}

export function getFolderDatasetImportJob(jobId) { return apiFetch(`/api/v1/datasets/import-jobs/${encodeURIComponent(jobId)}`); }
 
/* ---- Authentication & Google SSO ---- */
export function loginGoogleSSO(payload) {
  return apiFetch("/api/v1/auth/google", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getGoogleAuthConfig() {
  return apiFetch("/api/v1/auth/google/config");
}

export function getAuthMe() {
  return apiFetch("/api/v1/auth/me");
}

/* ---- Settings & W&B Integration ---- */
export function getWandbSettings() {
  return apiFetch("/api/v1/settings/wandb");
}

export function saveWandbSettings(payload) {
  return apiFetch("/api/v1/settings/wandb", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function testWandbConnection(payload) {
  return apiFetch("/api/v1/settings/wandb/test", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateUserProfile(payload) {
  return apiFetch("/api/v1/settings/profile", {
    method: "PUT",
    body: JSON.stringify(payload),
  });
}
