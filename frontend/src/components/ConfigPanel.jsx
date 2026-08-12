import React, { useState } from "react";
import UploadModal from "./UploadModal";

const GROUP_LABELS = { A: "Corruption", B: "Weather", C: "Occlusion", D: "Adversarial", E: "Patch", F: "Blackbox" };

export default function ConfigPanel({
  datasets,
  attacks,
  modes = [],
  modelVersions = [],
  mode,
  selectedModelVersion,
  selectedDataset,
  selectedAttacks,
  severity,
  isRunning,
  actions,
}) {
  const {
    setSelectedDataset,
    addDataset,
    setMode,
    setSelectedModelVersion,
    toggleAttack,
    setSeverity,
    handleRun,
    loadPreset,
    randomizeRecipe,
    sweepRecipe,
    previewRecipe,
  } = actions;

  const [recipeMode, setRecipeMode] = useState("manual"); // "manual" | "preset" | "random_n" | "sweep"
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [estimatedCost, setEstimatedCost] = useState(null);
  const [warningMessage, setWarningMessage] = useState("");
  const [randomN, setRandomN] = useState(3);
  const [sweepAttack, setSweepAttack] = useState("gaussian_noise");
  const [showUploadModal, setShowUploadModal] = useState(false);

  const selectedMode = modes.find((item) => item.id === mode);
  const versions = modelVersions.filter((item) => item.task === mode);
  const blocked = !selectedMode?.runnable || !modelVersions.find((item) => item.id === selectedModelVersion)?.runnable;

  const handlePreviewCost = async () => {
    setWarningMessage("");
    try {
      if (previewRecipe) {
        const est = await previewRecipe({
          recipe: {
            id: `recipe-preview-${Date.now()}`,
            task: mode,
            steps: selectedAttacks.map((atk) => ({ attack_id: atk, severity })),
          },
          n_samples: 8,
        });
        setEstimatedCost(est);
        if (est.estimated_seconds > 30) {
          setWarningMessage("High compute cost: Estimated run time > 30 seconds.");
        }
      } else {
        const seconds = selectedAttacks.length * severity * 0.5;
        setEstimatedCost({ estimated_seconds: seconds, total_samples: 8 * selectedAttacks.length });
      }
    } catch (err) {
      setWarningMessage(err.message || "Failed to preview resource cost");
    }
  };

  return (
    <aside className="config-panel" aria-label="Configuration Panel">
      {/* Recipe Strategy Selector */}
      <div className="config-panel__section">
        <label className="config-panel__label">Recipe Strategy Mode</label>
        <div className="tab-bar" style={{ gap: "4px" }}>
          {["manual", "preset", "random_n", "sweep"].map((m) => (
            <button
              key={m}
              type="button"
              className={`tab-bar__item ${recipeMode === m ? "tab-bar__item--active" : ""}`}
              onClick={() => setRecipeMode(m)}
              style={{ fontSize: "0.75rem", padding: "6px 8px" }}
            >
              {m.replace("_", " ").toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Preset Selector if in Preset mode */}
      {recipeMode === "preset" && (
        <div className="config-panel__section">
          <label className="config-panel__label">Load Preset Suite</label>
          <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
            <button
              type="button"
              className="attack-card"
              onClick={() => loadPreset && loadPreset("weather_robustness")}
              style={{ flex: 1, minWidth: "120px" }}
            >
              Weather Robustness
            </button>
            <button
              type="button"
              className="attack-card"
              onClick={() => loadPreset && loadPreset("sensor_fault_suite")}
              style={{ flex: 1, minWidth: "120px" }}
            >
              Sensor Fault Suite
            </button>
          </div>
        </div>
      )}

      {/* Random N Controls if in Random N mode */}
      {recipeMode === "random_n" && (
        <div className="config-panel__section">
          <label className="config-panel__label">Random N Steps</label>
          <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
            <input
              type="number"
              min="1"
              max="5"
              value={randomN}
              onChange={(e) => setRandomN(Number(e.target.value))}
              className="select-field"
              style={{ width: "70px" }}
            />
            <button
              type="button"
              className="attack-card"
              onClick={() => randomizeRecipe && randomizeRecipe(randomN)}
              style={{ flex: 1 }}
            >
              🎲 Randomize Recipe
            </button>
          </div>
        </div>
      )}

      {/* Sweep Controls if in Sweep mode */}
      {recipeMode === "sweep" && (
        <div className="config-panel__section">
          <label className="config-panel__label">Parameter Severity Sweep</label>
          <div className="select-wrapper" style={{ marginBottom: "8px" }}>
            <select
              className="select-field"
              value={sweepAttack}
              onChange={(e) => setSweepAttack(e.target.value)}
            >
              {attacks.map((atk) => (
                <option key={atk.name} value={atk.name}>
                  {atk.name}
                </option>
              ))}
            </select>
          </div>
          <button
            type="button"
            className="attack-card"
            onClick={() => sweepRecipe && sweepRecipe(sweepAttack)}
            style={{ width: "100%" }}
          >
            📊 Generate Severity Sweep (1..5)
          </button>
        </div>
      )}

      {/* Perception Mode */}
      <div className="config-panel__section">
        <label className="config-panel__label" htmlFor="perception-mode">
          Perception Mode
        </label>
        <div className="select-wrapper">
          <select
            id="perception-mode"
            className="select-field"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
          >
            {modes.map((item) => (
              <option key={item.id} value={item.id}>
                {item.title}
                {item.runnable ? "" : " (waiting)"}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Model Version */}
      <div className="config-panel__section">
        <label className="config-panel__label" htmlFor="model-version">
          Model Version
        </label>
        <div className="select-wrapper">
          <select
            id="model-version"
            className="select-field"
            value={selectedModelVersion || ""}
            onChange={(e) => setSelectedModelVersion(e.target.value)}
          >
            {versions.map((item) => (
              <option key={item.id} value={item.id} disabled={!item.runnable}>
                {item.id}
                {item.runnable ? "" : ` (${item.blocked_reason})`}
              </option>
            ))}
          </select>
        </div>
        {blocked && (
          <small style={{ color: "var(--warning)", marginTop: "4px", display: "block" }}>
            Model/evaluation artifacts are waiting for handoff (WAITING_FOR_ARTIFACTS).
          </small>
        )}
      </div>

      {/* Dataset & Ingestion Selector */}
      <div className="config-panel__section">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
          <label className="config-panel__label" style={{ margin: 0 }}>Dataset Source</label>
          <button
            type="button"
            className="tab-bar__item"
            onClick={() => setShowUploadModal(true)}
            style={{ fontSize: "0.75rem", padding: "4px 8px", background: "rgba(59, 130, 246, 0.15)", color: "var(--info)", border: "1px solid rgba(59, 130, 246, 0.3)", borderRadius: "var(--radius-sm)" }}
          >
            📤 Upload / Import
          </button>
        </div>
        <div className="select-wrapper">
          <select
            className="select-field"
            value={selectedDataset}
            onChange={(e) => setSelectedDataset(e.target.value)}
          >
            {datasets.map((ds) => (
              <option key={ds.id || ds.name} value={ds.id || ds.name} disabled={ds.anonymized === false}>
                {ds.title || ds.name || ds.id}
                {ds.anonymized === false ? " (Private)" : ""}
              </option>
            ))}
          </select>
        </div>
      </div>

      <UploadModal
        isOpen={showUploadModal}
        onClose={() => setShowUploadModal(false)}
        onDatasetCreated={(newDs) => {
          if (newDs) {
            if (addDataset) {
              addDataset(newDs);
            } else {
              setSelectedDataset(newDs.id || newDs.name);
            }
          }
        }}
        datasets={datasets}
      />

      {/* Attacks Grid (Manual Selection) */}
      <div className="config-panel__section">
        <label className="config-panel__label">Selected Attacks ({selectedAttacks.length})</label>
        <div className="attack-grid">
          {attacks.map((atk) => (
            <button
              key={atk.name}
              type="button"
              className={`attack-card ${selectedAttacks.includes(atk.name) ? "attack-card--selected" : ""}`}
              onClick={() => toggleAttack(atk.name)}
            >
              <span className="attack-card__name">{atk.name.replace(/_/g, " ")}</span>
              <span className="attack-card__group">{GROUP_LABELS[atk.group] || atk.group}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Severity Slider */}
      <div className="config-panel__section">
        <label className="config-panel__label">Severity</label>
        <div className="severity-value">{severity}</div>
        <input
          aria-label="Severity"
          type="range"
          className="severity-slider__input"
          min="1"
          max="5"
          value={severity}
          onChange={(e) => setSeverity(Number(e.target.value))}
        />
      </div>

      {/* Resource Estimate & Cost Preview Button */}
      <div className="config-panel__section" style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "12px" }}>
        <button
          type="button"
          className="tab-bar__item"
          onClick={handlePreviewCost}
          style={{ width: "100%", justifyContent: "center", marginBottom: "8px" }}
        >
          ⚡ Preview Resource Cost & Estimate
        </button>

        {estimatedCost && (
          <div className="glass-panel" style={{ padding: "8px 12px", fontSize: "0.8rem" }}>
            <div>Est. Duration: <strong>{estimatedCost.estimated_seconds || estimatedCost.total_cost_class || 1.5}s</strong></div>
            <div>Total Samples: <strong>{estimatedCost.total_samples || (8 * selectedAttacks.length)}</strong></div>
          </div>
        )}

        {warningMessage && (
          <div className="status-badge status-badge--failed" style={{ marginTop: "8px", width: "100%", textAlign: "center" }}>
            {warningMessage}
          </div>
        )}
      </div>

      {/* Advanced Settings Drawer Toggle */}
      <div className="config-panel__section">
        <button
          type="button"
          className="text-xs text-secondary"
          onClick={() => setShowAdvanced(!showAdvanced)}
          style={{ background: "none", border: "none", cursor: "pointer", textDecoration: "underline" }}
        >
          {showAdvanced ? "▼ Hide Advanced Drawer" : "▶ Show Advanced Settings Drawer"}
        </button>

        {showAdvanced && (
          <div className="glass-panel mt-2" style={{ padding: "10px", fontSize: "0.8rem" }}>
            <label style={{ display: "block", marginBottom: "4px" }}>Seed: 42</label>
            <label style={{ display: "block", marginBottom: "4px" }}>Sample Limit: 8</label>
            <label style={{ display: "block", marginBottom: "4px" }}>IoU Threshold: 0.50</label>
            <label style={{ display: "block" }}>Confidence Threshold: 0.25</label>
          </div>
        )}
      </div>

      {/* Run Action */}
      <button
        className={`run-button ${isRunning ? "run-button--loading" : ""}`}
        onClick={handleRun}
        disabled={isRunning || blocked || selectedAttacks.length === 0}
      >
        {isRunning ? "Running..." : blocked ? "Waiting for model artifacts" : "Run Attack Recipe"}
      </button>
    </aside>
  );
}

