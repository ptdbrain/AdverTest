"use client";

import React, { useRef, useState } from "react";
import { getCheckpoint, uploadCheckpoint } from "@/lib/api";

/** Candidate workspace deliberately separate from the Attack recipe selector. */
export default function DefencePanel({ baselineRunId, baseCheckpoints, candidates, comparison, running, progress, onEvaluate, onRefreshCandidates }) {
  const [candidateId, setCandidateId] = useState("");
  const [parentId, setParentId] = useState("");
  const [uploadState, setUploadState] = useState("");
  const fileInputRef = useRef(null);
  const uploadCandidate = async (event) => {
    const file = event.target.files?.[0];
    if (!file || !parentId) return;
    const parent = baseCheckpoints.find((item) => item.id === parentId);
    setUploadState("Uploading candidate: 0%");
    try {
      const created = await uploadCheckpoint(file, {
        taskId: parent.task, familyId: parent.model_family_id, displayName: file.name,
        role: "fine_tuned", parentCheckpointId: parentId,
      }, (ratio) => setUploadState(`Uploading candidate: ${Math.round(ratio * 100)}%`));
      for (let attempt = 0; attempt < 60; attempt += 1) {
        const status = await getCheckpoint(created.checkpoint_id);
        if (status.status === "READY") { setUploadState("Candidate ready for Defence."); await onRefreshCandidates?.(); return; }
        if (status.status === "REJECTED") { setUploadState(`Candidate rejected: ${status.validation_reason || "validation failed"}`); return; }
        setUploadState("Validating candidate…"); await new Promise((resolve) => setTimeout(resolve, 500));
      }
      setUploadState("Candidate remains in validation; refresh this tab later.");
    } catch (error) { setUploadState(error.message || "Candidate upload failed."); }
    finally { event.target.value = ""; }
  };
  return <section className="glass-panel" aria-label="Defence workspace" style={{ padding: 20, marginTop: 14 }}>
    <p className="evidence-stage__eyebrow">Defence</p>
    <h2>Evaluate a fine-tuned or repaired candidate</h2>
    <p className="text-secondary">The candidate inherits the selected Attack run&apos;s dataset version, samples, recipe, seed, thresholds and preprocessing. It cannot be used in Attack.</p>
    {!baselineRunId ? <p className="empty-state">Complete an Attack run before starting a paired Defence evaluation.</p> : <div style={{ display: "grid", gap: 10 }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "end" }}>
        <label className="config-panel__label" style={{ flex: 1 }}>Parent base checkpoint
          <select aria-label="Candidate parent base checkpoint" className="select-field" value={parentId} onChange={(event) => setParentId(event.target.value)}>
            <option value="">Select base parent</option>{baseCheckpoints.map((item) => <option key={item.id} value={item.id}>{item.model_name} — {item.id}</option>)}
          </select>
        </label>
        <button type="button" className="tab-bar__item" disabled={!parentId} onClick={() => fileInputRef.current?.click()}>Upload candidate</button>
        <input ref={fileInputRef} aria-label="Upload defence candidate" type="file" accept=".pt,.pth" style={{ display: "none" }} onChange={uploadCandidate} />
      </div>
      {uploadState && <p className="text-xs text-secondary" role="status">{uploadState}</p>}
      <label className="config-panel__label">Candidate checkpoint
        <select aria-label="Defence candidate checkpoint" className="select-field" value={candidateId} onChange={(event) => setCandidateId(event.target.value)}>
          <option value="">Choose a validated fine-tuned or repaired checkpoint</option>
          {candidates.map((item) => <option key={item.id} value={item.id} disabled={!item.runnable}>{item.model_name} — {item.checkpoint_role}{item.blocked_reason ? ` (${item.blocked_reason})` : ""}</option>)}
        </select>
      </label>
      <button type="button" className="run-button" disabled={running || !candidateId} onClick={() => onEvaluate(candidateId)}>
        {running ? `Evaluating ${progress}%` : "Run paired Defence evaluation"}
      </button>
      {comparison && <section aria-label="Base versus fine-tuned metrics" className="glass-panel" style={{ padding: 12 }}>
        <strong>Base / fine-tuned delta</strong>
        {!comparison.paired ? <p className="text-xs text-danger">Recovery is unavailable: {comparison.incompatibilities?.join(", ")}</p> : <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 8, marginTop: 8 }}>
          <div><span className="text-xs text-secondary">Base</span><p>{comparison.recovery_report?.baseline_clean?.toFixed?.(3) ?? "—"}</p></div>
          <div><span className="text-xs text-secondary">Fine-tuned</span><p>{comparison.recovery_report?.candidate_clean?.toFixed?.(3) ?? "—"}</p></div>
          <div><span className="text-xs text-secondary">Recovery</span><p>{comparison.recovery_report?.recovery_rate?.percent_value == null ? "—" : `${comparison.recovery_report.recovery_rate.percent_value.toFixed(1)}%`}</p></div>
        </div>}
        {comparison.paired && <div className="text-xs text-secondary" style={{ marginTop: 8 }}>{Object.entries(comparison.metric_deltas || {}).filter(([name]) => name !== "clean_detection_score").map(([name, metric]) => <div key={name}>{name.replaceAll("_", " ")}: {metric.value >= 0 ? "+" : ""}{Number(metric.value).toFixed(3)}</div>)}</div>}
      </section>}
    </div>}
  </section>;
}
