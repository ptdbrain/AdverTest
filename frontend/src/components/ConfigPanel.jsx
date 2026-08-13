import React, { useRef, useState } from "react";
import UploadModal from "./UploadModal";
import { getCheckpoint, uploadCheckpoint } from "@/lib/api";

const GROUP_LABELS = { A: "Corruption", B: "Weather", C: "Occlusion", D: "Adversarial", E: "Patch", F: "Blackbox" };

function checkpointLabel(item) {
  const path = item.checkpoint_path || "";
  const filename = path.split(/[\\/]/).pop();
  return filename ? `${item.id} — ${filename}` : item.id;
}

export default function ConfigPanel({
  datasets,
  attacks,
  modes = [],
  modelFamilies = [],
  baseCheckpoints = [],
  recipePresets = [],
  mode,
  selectedModelFamily,
  selectedModelVersion,
  selectedDataset,
  selectedAttacks,
  recipe = { steps: [] },
  runOptions,
  isRunning,
  actions,
}) {
  const {
    setSelectedDataset,
    addDataset,
    setMode,
    setModelFamily,
    setSelectedModelVersion,
    toggleAttack,
    updateAttackSeverity,
    handleRun,
    loadPreset,
    randomizeRecipe,
    sweepRecipe,
    previewRecipe,
    setRunOptions,
    refreshBaseCheckpoints,
  } = actions;

  const [recipeMode, setRecipeMode] = useState("manual"); // "manual" | "preset" | "random_n" | "sweep"
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [estimatedCost, setEstimatedCost] = useState(null);
  const [warningMessage, setWarningMessage] = useState("");
  const [randomN, setRandomN] = useState(3);
  const [sweepAttack, setSweepAttack] = useState("gaussian_noise");
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [threatFilter, setThreatFilter] = useState("");
  const [typeFilter, setTypeFilter] = useState("");
  const [scenarioFilter, setScenarioFilter] = useState("");
  const [checkpointUpload, setCheckpointUpload] = useState("");
  const checkpointInputRef = useRef(null);

  const checkpoints = baseCheckpoints.filter((item) => item.task === mode && item.model_family_id === selectedModelFamily);
  const blocked = !checkpoints.find((item) => item.id === selectedModelVersion)?.runnable;
  const visibleAttacks = attacks.filter((item) => (!threatFilter || item.threat_model === threatFilter) && (!typeFilter || item.attack_type === typeFilter) && (!scenarioFilter || item.scenario_kind === scenarioFilter));

  const handlePreviewCost = async () => {
    setWarningMessage("");
    try {
      const est = await previewRecipe({
          recipe: {
            name: "preview",
            steps: recipe.steps,
          },
          dataset_version_id: selectedDataset,
        });
        setEstimatedCost(est);
      if (est.estimated_seconds > 30) {
        setWarningMessage("High compute cost: Estimated run time > 30 seconds.");
      }
    } catch (err) {
      setWarningMessage(err.message || "Failed to preview resource cost");
    }
  };

  const handleCheckpointFile = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setCheckpointUpload("Uploading checkpoint: 0%");
    try {
      const created = await uploadCheckpoint(file, {
        taskId: mode,
        familyId: selectedModelFamily,
        displayName: file.name,
        role: "base",
      }, (ratio) => setCheckpointUpload(`Uploading checkpoint: ${Math.round(ratio * 100)}%`));
      setCheckpointUpload("Validating checkpoint…");
      for (let attempt = 0; attempt < 60; attempt += 1) {
        const status = await getCheckpoint(created.checkpoint_id);
        if (status.status === "READY") {
          setCheckpointUpload("Checkpoint ready.");
          await refreshBaseCheckpoints?.();
          return;
        }
        if (status.status === "REJECTED") {
          setCheckpointUpload(`Checkpoint rejected: ${status.validation_reason || "validation failed"}`);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
      setCheckpointUpload("Checkpoint is still validating; it will appear when READY.");
    } catch (error) { setCheckpointUpload(error.message || "Checkpoint upload failed."); }
    finally { event.target.value = ""; }
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
            {recipePresets.map((preset) => (
              <button key={preset.preset_id} type="button" className="attack-card" onClick={() => loadPreset?.(preset.preset_id)} style={{ flex: 1, minWidth: "120px" }}>
                {preset.name}
              </button>
            ))}
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
              <option key={item.id} value={item.id} disabled={item.status === "coming_later"}>
                {item.title}
                {item.status === "coming_later" ? " (coming later)" : ""}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* A perception task, its family, and a vendor/base weight are separate choices. */}
      <div className="config-panel__section">
        <label className="config-panel__label" htmlFor="model-family">Base Model Family</label>
        <div className="select-wrapper">
          <select id="model-family" className="select-field" value={selectedModelFamily || ""} onChange={(e) => setModelFamily(e.target.value)}>
            {modelFamilies.map((item) => <option key={item.id} value={item.id}>{item.display_name}{item.runnable ? "" : ` (${item.blocked_reason || "unavailable"})`}</option>)}
          </select>
        </div>
      </div>
      <div className="config-panel__section">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <label className="config-panel__label" htmlFor="model-version">Base Checkpoint</label>
          <button type="button" className="tab-bar__item" onClick={() => checkpointInputRef.current?.click()} disabled={!selectedModelFamily || isRunning}>Upload checkpoint</button>
          <input ref={checkpointInputRef} aria-label="Upload base checkpoint" type="file" accept={modelFamilies.find((item) => item.id === selectedModelFamily)?.checkpoint_extensions?.join(",") || ".pt,.pth"} style={{ display: "none" }} onChange={handleCheckpointFile} />
        </div>
        <div className="select-wrapper">
          <select id="model-version" className="select-field" value={selectedModelVersion || ""} onChange={(e) => setSelectedModelVersion(e.target.value)}>
            {checkpoints.map((item) => <option key={item.id} value={item.id} disabled={!item.runnable}>{item.model_name} — {item.id}</option>)}
          </select>
        </div>
        {blocked && (
          <small style={{ color: "var(--warning)", marginTop: "4px", display: "block" }}>
            A validated base checkpoint is required for Attack. Fine-tuned and repaired checkpoints are available only in Defence.
          </small>
        )}
        {checkpointUpload && <small role="status" style={{ display: "block", marginTop: 6 }}>{checkpointUpload}</small>}
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
              <option key={ds.id || ds.name} value={ds.id || ds.name}>
                {ds.title || ds.name || ds.id} — {ds.annotation_schema?.join(", ") || "declared contract"}
                {ds.benchmark_ready === false ? " (Quick inference only)" : ""}
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
        taskId={mode}
      />

      {/* Attacks Grid (Manual Selection) */}
      <div className="config-panel__section">
        <label className="config-panel__label">Selected Attacks ({selectedAttacks.length})</label>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "6px", marginBottom: "8px" }}>
          {[[threatFilter, setThreatFilter, "Threat model", "threat_model"], [typeFilter, setTypeFilter, "Attack type", "attack_type"], [scenarioFilter, setScenarioFilter, "Scenario", "scenario_kind"]].map(([value, setter, label, field]) => (
            <select key={field} aria-label={label} className="select-field" value={value} onChange={(event) => setter(event.target.value)}>
              <option value="">{label}</option>
              {[...new Set(attacks.map((attack) => attack[field]).filter(Boolean))].map((option) => <option key={option} value={option}>{option.replace(/_/g, " ")}</option>)}
            </select>
          ))}
        </div>
        <div className="attack-grid">
          {visibleAttacks.map((atk) => (
            <button
              key={atk.name}
              type="button"
              className={`attack-card ${selectedAttacks.includes(atk.name) ? "attack-card--selected" : ""}`}
              onClick={() => toggleAttack(atk.name)}
              disabled={atk.available === false}
              title={atk.available === false ? atk.reason : undefined}
            >
              <span className="attack-card__name">{atk.name.replace(/_/g, " ")}</span>
              <span className="attack-card__group">{atk.attack_type?.replace(/_/g, " ") || GROUP_LABELS[atk.group] || atk.group}</span>
              {atk.available === false && <span className="attack-card__group">Unavailable: {atk.reason}</span>}
            </button>
          ))}
        </div>
      </div>

      {/* Every ordered recipe step owns its severity. */}
      <div className="config-panel__section">
        <label className="config-panel__label">Selected Recipe</label>
        {recipe.steps.map((step) => (
          <label key={`${step.position}-${step.attack_name}`} className="recipe-step">
            <span>{step.attack_name.replace(/_/g, " ")}</span>
            <span>Severity {step.severity}</span>
            <input aria-label={`${step.attack_name} severity`} type="range" min="1" max="5" value={step.severity} onChange={(event) => updateAttackSeverity?.(step.position, Number(event.target.value))} />
          </label>
        ))}
        {!recipe.steps.length && <small className="text-secondary">Add an attack to create a recipe step.</small>}
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
            <div>Recipe steps: <strong>{estimatedCost.step_count ?? selectedAttacks.length}</strong></div>
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
            <label htmlFor="run-seed">Seed</label>
            <input id="run-seed" aria-label="Seed" type="number" min="0" value={runOptions?.seed ?? 42} onChange={(event) => setRunOptions?.({ seed: Number(event.target.value) })} />
            <label htmlFor="run-limit">Sample Limit</label>
            <input id="run-limit" aria-label="Sample Limit" type="number" min="1" value={runOptions?.limit ?? 8} onChange={(event) => setRunOptions?.({ limit: Number(event.target.value) })} />
            <label htmlFor="run-iou">IoU Threshold</label>
            <input id="run-iou" aria-label="IoU Threshold" type="number" min="0" max="1" step="0.01" value={runOptions?.iou ?? 0.5} onChange={(event) => setRunOptions?.({ iou: Number(event.target.value) })} />
            <label htmlFor="run-confidence">Confidence Threshold</label>
            <input id="run-confidence" aria-label="Confidence Threshold" type="number" min="0" max="1" step="0.01" value={runOptions?.confidence ?? 0.25} onChange={(event) => setRunOptions?.({ confidence: Number(event.target.value) })} />
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
