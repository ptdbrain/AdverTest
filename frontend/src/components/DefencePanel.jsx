"use client";

import React, { useRef, useState } from "react";
import { getCheckpoint, uploadCheckpoint } from "@/lib/api";

export default function DefencePanel({ baselineRunId, baseCheckpoints, candidates, comparison, running, progress, onEvaluate, onRefreshCandidates }) {
  const [candidateId, setCandidateId] = useState("");
  const [parentId, setParentId] = useState("");
  const [uploadState, setUploadState] = useState("");
  const fileInputRef = useRef(null);

  const uploadCandidate = async (event) => {
    const file = event.target.files?.[0];
    if (!file || !parentId) return;
    const parent = baseCheckpoints.find((item) => item.id === parentId);
    setUploadState("Uploading: 0%");
    try {
      const created = await uploadCheckpoint(file, {
        taskId: parent.task, familyId: parent.model_family_id, displayName: file.name,
        role: "fine_tuned", parentCheckpointId: parentId,
      }, (ratio) => setUploadState(`Uploading: ${Math.round(ratio * 100)}%`));
      for (let attempt = 0; attempt < 60; attempt += 1) {
        const status = await getCheckpoint(created.checkpoint_id);
        if (status.status === "READY") { setUploadState("Candidate ready."); await onRefreshCandidates?.(); return; }
        if (status.status === "REJECTED") { setUploadState(`Rejected: ${status.validation_reason || "failed"}`); return; }
        setUploadState("Validating..."); await new Promise((resolve) => setTimeout(resolve, 500));
      }
      setUploadState("Still validating — refresh later.");
    } catch (error) { setUploadState(error.message || "Upload failed."); }
    finally { event.target.value = ""; }
  };

  return (
    <section className="glass-panel" aria-label="Defence workspace" style={{ padding: 16, margin: "var(--space-sm)", borderRadius: "var(--radius-md)" }}>
      <p className="evidence-stage__eyebrow">Defence</p>
      <h2 style={{ fontSize: "0.85rem", fontWeight: 600, margin: "4px 0 8px", color: "var(--text-primary)" }}>Evaluate fine-tuned candidate</h2>
      <p className="text-xs text-secondary" style={{ marginBottom: 12 }}>
        Candidate inherits the Attack run&apos;s dataset, recipe, seed, and thresholds.
      </p>

      {!baselineRunId ? (
        <p className="text-xs text-tertiary" style={{ padding: "var(--space-md) 0" }}>Complete an Attack run first.</p>
      ) : (
        <div style={{ display: "grid", gap: 8 }}>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "end" }}>
            <label className="config-panel__label" style={{ flex: 1 }}>
              Parent checkpoint
              <select aria-label="Parent checkpoint" className="select-field" value={parentId} onChange={(event) => setParentId(event.target.value)}>
                <option value="">Select base parent</option>
                {baseCheckpoints.map((item) => <option key={item.id} value={item.id}>{item.model_name} — {item.id}</option>)}
              </select>
            </label>
            <button type="button" className="action-button action-button--secondary" disabled={!parentId} onClick={() => fileInputRef.current?.click()} style={{ padding: "6px 10px", fontSize: "0.62rem" }}>
              Upload candidate
            </button>
            <input ref={fileInputRef} aria-label="Upload defence candidate" type="file" accept=".pt,.pth" style={{ display: "none" }} onChange={uploadCandidate} />
          </div>

          {uploadState && <p className="text-xs text-tertiary" role="status">{uploadState}</p>}

          <label className="config-panel__label">
            Candidate checkpoint
            <select aria-label="Defence candidate" className="select-field" value={candidateId} onChange={(event) => setCandidateId(event.target.value)}>
              <option value="">Choose validated checkpoint</option>
              {candidates.map((item) => <option key={item.id} value={item.id} disabled={!item.runnable}>{item.model_name} — {item.checkpoint_role}{item.blocked_reason ? ` (${item.blocked_reason})` : ""}</option>)}
            </select>
          </label>

          <button type="button" className="run-button" disabled={running || !candidateId} onClick={() => onEvaluate(candidateId)}>
            {running ? `Evaluating ${progress}%` : "Run Defence Evaluation"}
          </button>

          {comparison && (
            <section aria-label="Base vs fine-tuned" className="glass-panel" style={{ padding: 10, marginTop: 4 }}>
              <strong style={{ fontSize: "0.72rem", color: "var(--text-primary)" }}>Base / fine-tuned delta</strong>
              {!comparison.paired ? (
                <p className="text-xs" style={{ color: "var(--danger)", marginTop: 4 }}>Recovery unavailable: {comparison.incompatibilities?.join(", ")}</p>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 6, marginTop: 6 }}>
                  <div><span className="text-xs text-secondary">Base</span><p className="text-mono" style={{ fontSize: "0.78rem" }}>{comparison.recovery_report?.baseline_clean?.toFixed?.(3) ?? "—"}</p></div>
                  <div><span className="text-xs text-secondary">Fine-tuned</span><p className="text-mono" style={{ fontSize: "0.78rem" }}>{comparison.recovery_report?.candidate_clean?.toFixed?.(3) ?? "—"}</p></div>
                  <div><span className="text-xs text-secondary">Recovery</span><p className="text-mono" style={{ fontSize: "0.78rem" }}>{comparison.recovery_report?.recovery_rate?.percent_value == null ? "—" : `${comparison.recovery_report.recovery_rate.percent_value.toFixed(1)}%`}</p></div>
                </div>
              )}
              {comparison.paired && (
                <div className="text-xs text-secondary" style={{ marginTop: 6 }}>
                  {Object.entries(comparison.metric_deltas || {}).filter(([name]) => name !== "clean_detection_score").map(([name, metric]) => (
                    <div key={name}>{name.replaceAll("_", " ")}: {metric.value >= 0 ? "+" : ""}{Number(metric.value).toFixed(3)}</div>
                  ))}
                </div>
              )}
            </section>
          )}
        </div>
      )}
    </section>
  );
}
