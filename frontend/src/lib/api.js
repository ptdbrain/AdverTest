const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function apiFetch(path, options = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `API error ${res.status}`);
  }
  return res.json();
}

export function getCatalogAttacks(params = {}) {
  const qs = new URLSearchParams(params).toString();
  return apiFetch(`/api/v1/catalog/attacks${qs ? `?${qs}` : ""}`);
}

export function getCatalogModels() {
  return apiFetch("/api/v1/catalog/models");
}

export function getCatalogDatasets() {
  return apiFetch("/api/v1/catalog/datasets");
}

export function getModelVersions() { return apiFetch("/api/v1/model-versions"); }
export function getPerceptionModes() { return apiFetch("/api/v1/perception-modes"); }

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

export function cancelRun(runId) {
  return apiFetch(`/api/v1/runs/${runId}/cancel`, { method: "POST" });
}

export function connectRunWebSocket(runId, onEvent) {
  const wsBase = API_BASE.replace(/^http/, "ws");
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

export function resolveReview(reviewId, decision, decisionNote, resolvedBy) {
  return apiFetch(`/api/v1/reviews/${reviewId}`, {
    method: "PATCH",
    body: JSON.stringify({
      decision,
      decision_note: decisionNote,
      resolved_by: resolvedBy,
    }),
  });
}

export function triggerAutoFlag(runId, threshold = 30) {
  return apiFetch(`/api/v1/runs/${runId}/flag-reviews?threshold=${threshold}`, {
    method: "POST"
  });
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

/* ---- Recipe API ---- */
export function getRecipePresets() {
  return apiFetch("/api/v1/catalog/recipes/presets");
}

export function randomizeRecipe(nSteps = 3, group = null) {
  return apiFetch("/api/v1/attack-recipes/randomize", {
    method: "POST",
    body: JSON.stringify({ n_steps: nSteps, group }),
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
export async function uploadImage(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const res = await fetch(`${API_BASE}/api/v1/uploads/images`, {
    method: "POST",
    headers: {
      "Content-Type": "application/octet-stream",
      "x-filename": file.name,
    },
    body: bytes,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Upload failed with status ${res.status}`);
  }
  return res.json();
}

export function importFolderDataset({ root, name, inputFormat = "yolo", anonymizationManifest = "manifest.jsonl", maxSamples = 50 }) {
  return apiFetch("/api/v1/datasets/import", {
    method: "POST",
    body: JSON.stringify({
      root,
      name,
      input_format: inputFormat,
      anonymization_manifest: anonymizationManifest,
      max_samples: maxSamples,
    }),
  });
}


