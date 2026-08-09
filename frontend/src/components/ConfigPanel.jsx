import React from "react";

const GROUP_LABELS = { A: "Corruption", B: "Weather", C: "Occlusion", D: "Adversarial", E: "Patch", F: "Blackbox" };

export default function ConfigPanel({ datasets, attacks, modes = [], modelVersions = [], mode, selectedModelVersion, selectedDataset, selectedAttacks, severity, isRunning, actions }) {
  const { setSelectedDataset, setMode, setSelectedModelVersion, toggleAttack, setSeverity, handleRun } = actions;
  const selectedMode = modes.find((item) => item.id === mode);
  const versions = modelVersions.filter((item) => item.task === mode);
  const blocked = !selectedMode?.runnable || !modelVersions.find((item) => item.id === selectedModelVersion)?.runnable;
  return <aside className="config-panel">
    <div className="config-panel__section"><label className="config-panel__label" htmlFor="perception-mode">Perception mode</label><div className="select-wrapper"><select id="perception-mode" className="select-field" value={mode} onChange={(e) => setMode(e.target.value)}>{modes.map((item) => <option key={item.id} value={item.id}>{item.title}{item.runnable ? "" : " (waiting)"}</option>)}</select></div></div>
    <div className="config-panel__section"><label className="config-panel__label" htmlFor="model-version">Model version</label><div className="select-wrapper"><select id="model-version" className="select-field" value={selectedModelVersion || ""} onChange={(e) => setSelectedModelVersion(e.target.value)}>{versions.map((item) => <option key={item.id} value={item.id} disabled={!item.runnable}>{item.id}{item.runnable ? "" : ` (${item.blocked_reason})`}</option>)}</select></div>{blocked && <small>Model/evaluation artefacts are waiting for handoff.</small>}</div>
    <div className="config-panel__section"><label className="config-panel__label">Dataset</label><div className="select-wrapper"><select className="select-field" value={selectedDataset} onChange={(e) => setSelectedDataset(e.target.value)}>{datasets.map((ds) => <option key={ds.name} value={ds.name} disabled={!ds.anonymized}>{ds.title || ds.name}{ds.anonymized ? "" : " (Private)"}</option>)}</select></div></div>
    <div className="config-panel__section"><label className="config-panel__label">Attack Type</label><div className="attack-grid">{attacks.map((atk) => <button key={atk.name} className={`attack-card ${selectedAttacks.includes(atk.name) ? "attack-card--selected" : ""}`} onClick={() => toggleAttack(atk.name)}><span className="attack-card__name">{atk.name.replace(/_/g, " ")}</span><span className="attack-card__group">{GROUP_LABELS[atk.group] || atk.group}</span></button>)}</div></div>
    <div className="config-panel__section"><label className="config-panel__label">Severity</label><div className="severity-value">{severity}</div><input aria-label="Severity" type="range" className="severity-slider__input" min="1" max="5" value={severity} onChange={(e) => setSeverity(Number(e.target.value))} /></div>
    <button className={`run-button ${isRunning ? "run-button--loading" : ""}`} onClick={handleRun} disabled={isRunning || blocked || selectedAttacks.length === 0}>{isRunning ? "Running..." : blocked ? "Waiting for model artefacts" : "Run Attack"}</button>
  </aside>;
}
