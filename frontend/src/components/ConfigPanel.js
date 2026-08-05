import React from "react";

const GROUP_LABELS = {
  A: "Corruption",
  B: "Weather",
  C: "Occlusion",
  D: "Adversarial",
  E: "Patch",
  F: "Blackbox",
};

/**
 * ConfigPanel Component
 * Handles dataset selection, attack type toggling, and severity slider.
 */
export default function ConfigPanel({
  datasets,
  attacks,
  selectedDataset,
  selectedAttacks,
  severity,
  isRunning,
  actions,
}) {
  const { setSelectedDataset, toggleAttack, setSeverity, handleRun } = actions;

  return (
    <aside className="config-panel">
      {/* Dataset selector */}
      <div className="config-panel__section">
        <label className="config-panel__label">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <ellipse cx="12" cy="5" rx="9" ry="3" />
            <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
            <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
          </svg>
          Dataset
        </label>
        <div className="select-wrapper">
          <select
            className="select-field"
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
          >
            {datasets.map((ds) => {
              const requiresParams = ds.name === "generated_dataset";
              const isDisabled = !ds.anonymized || requiresParams;
              return (
                <option key={ds.name} value={ds.name} disabled={isDisabled}>
                  {ds.title || ds.name} {ds.anonymized ? "" : "(Private)"} {requiresParams ? "(Requires Params)" : ""}
                </option>
              );
            })}
          </select>
        </div>
      </div>

      {/* Attack selector */}
      <div className="config-panel__section">
        <label className="config-panel__label">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
          </svg>
          Attack Type
        </label>
        <div className="attack-grid">
          {attacks.map((atk) => (
            <button
              key={atk.name}
              className={`attack-card ${selectedAttacks.includes(atk.name) ? "attack-card--selected" : ""}`}
              onClick={() => toggleAttack(atk.name)}
              title={`${atk.name} (${atk.cost_class})`}
            >
              <span className="attack-card__name">{atk.name.replace(/_/g, " ")}</span>
              <span className="attack-card__group">{GROUP_LABELS[atk.group] || atk.group}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Severity slider */}
      <div className="config-panel__section">
        <label className="config-panel__label">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2v20M2 12h20" />
          </svg>
          Severity
        </label>
        <div className="severity-value">{severity}</div>
        <div className="severity-slider">
          <div className="severity-slider__track-wrapper">
            <input
              type="range"
              className="severity-slider__input"
              min="1"
              max="5"
              step="1"
              value={severity}
              onChange={(e) => setSeverity(Number(e.target.value))}
            />
          </div>
          <div className="severity-slider__labels">
            {[1, 2, 3, 4, 5].map((s) => (
              <span key={s} className={s === severity ? "active" : ""}>
                {s}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Run button */}
      <button
        className={`run-button ${isRunning ? "run-button--loading" : ""}`}
        onClick={handleRun}
        disabled={isRunning || selectedAttacks.length === 0}
      >
        {isRunning && <span className="run-button__ripple" />}
        {isRunning ? "Running..." : "Run Attack"}
      </button>
    </aside>
  );
}
