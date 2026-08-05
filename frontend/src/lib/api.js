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

