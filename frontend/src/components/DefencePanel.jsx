"use client";

import React, { useState } from "react";

/** Candidate workspace deliberately separate from the Attack recipe selector. */
export default function DefencePanel({ baselineRunId, candidates, running, progress, onEvaluate }) {
  const [candidateId, setCandidateId] = useState("");
  return <section className="glass-panel" aria-label="Defence workspace" style={{ padding: 20, marginTop: 14 }}>
    <p className="evidence-stage__eyebrow">Defence</p>
    <h2>Evaluate a fine-tuned or repaired candidate</h2>
    <p className="text-secondary">The candidate inherits the selected Attack run&apos;s dataset version, samples, recipe, seed, thresholds and preprocessing. It cannot be used in Attack.</p>
    {!baselineRunId ? <p className="empty-state">Complete an Attack run before starting a paired Defence evaluation.</p> : <div style={{ display: "grid", gap: 10 }}>
      <label className="config-panel__label">Candidate checkpoint
        <select aria-label="Defence candidate checkpoint" className="select-field" value={candidateId} onChange={(event) => setCandidateId(event.target.value)}>
          <option value="">Choose a validated fine-tuned or repaired checkpoint</option>
          {candidates.map((item) => <option key={item.id} value={item.id} disabled={!item.runnable}>{item.model_name} — {item.checkpoint_role}{item.blocked_reason ? ` (${item.blocked_reason})` : ""}</option>)}
        </select>
      </label>
      <button type="button" className="run-button" disabled={running || !candidateId} onClick={() => onEvaluate(candidateId)}>
        {running ? `Evaluating ${progress}%` : "Run paired Defence evaluation"}
      </button>
    </div>}
  </section>;
}
