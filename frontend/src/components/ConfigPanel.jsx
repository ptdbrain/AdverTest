import React, { useRef, useState } from "react";
import UploadModal from "./UploadModal";
import { getCheckpoint, uploadCheckpoint } from "@/lib/api";

const ATTACK_TAB_DEFS = [
  ["white", "White-box"],
  ["gray", "Gray-box"],
  ["black", "Black-box"],
];

function attackSection(attack) {
  if (attack.threat_model === "white_box") return "white";
  if (attack.threat_model === "black_box" || attack.attack_type === "query") return "black";
  // Gray-box includes physical, transfer, patch, weather, sensor, corruption
  return "gray";
}

function checkpointLabel(item) {
  const path = item.checkpoint_path || "";
  const filename = path.split(/[\\/]/).pop();
  return filename ? `${item.id} — ${filename}` : item.id;
}

function formatDisabledReason(reason) {
  if (!reason) return "N/A";
  if (reason.includes("modality_mismatch:lidar")) return "LiDAR only";
  if (reason.includes("modality_mismatch:multi")) return "Multi-cam";
  if (reason.includes("missing_model_capability")) return "Incompatible";
  if (reason.includes("CHECKPOINT_UNKNOWN")) return "No Checkpoint";
  return reason.replace(/_/g, " ").slice(0, 12);
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
  progress = 0,
  progressDetail = "",
  runStatus,
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
    cancelRun: cancelRunAction,
  } = actions;

  const [attackTab, setAttackTab] = useState("white");
  const [expandedAttack, setExpandedAttack] = useState(null);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [checkpointUpload, setCheckpointUpload] = useState("");
  const checkpointInputRef = useRef(null);

  const checkpoints = baseCheckpoints.filter(
    (item) => item.task === mode && item.model_family_id === selectedModelFamily
  );
  const blocked = !checkpoints.find((item) => item.id === selectedModelVersion)?.runnable;

  const visibleAttacks = attacks.filter(
    (item) => attackSection(item) === attackTab
  );

  const handleCheckpointFile = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setCheckpointUpload("Uploading: 0%");
    try {
      const created = await uploadCheckpoint(file, {
        taskId: mode,
        familyId: selectedModelFamily,
        displayName: file.name,
        role: "base",
      }, (ratio) => setCheckpointUpload(`Uploading: ${Math.round(ratio * 100)}%`));
      setCheckpointUpload("Validating...");
      for (let attempt = 0; attempt < 60; attempt += 1) {
        const status = await getCheckpoint(created.checkpoint_id);
        if (status.status === "READY") {
          setCheckpointUpload("Ready");
          await refreshBaseCheckpoints?.();
          return;
        }
        if (status.status === "REJECTED") {
          setCheckpointUpload(`Rejected: ${status.validation_reason || "failed"}`);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
      setCheckpointUpload("Still validating...");
    } catch (error) {
      setCheckpointUpload(error.message || "Upload failed.");
    } finally {
      event.target.value = "";
    }
  };

  const handleAttackCardClick = (attackName) => {
    toggleAttack(attackName);
  };

  const handleSeveritySelect = (attackName, severity) => {
    const step = recipe.steps.find((s) => s.attack_name === attackName);
    if (step) {
      updateAttackSeverity?.(step.position, severity);
    }
  };

  const getAttackSeverity = (attackName) => {
    const step = recipe.steps.find((s) => s.attack_name === attackName);
    return step?.severity ?? 3;
  };

  return (
    <aside className="config-panel" aria-label="Configuration Panel">
      {/* 1. Task Selector (Segmented Control Pills) */}
      <div className="config-panel__section">
        <label className="config-panel__label">Task</label>
        <div className="segmented-control">
          {modes.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`segmented-control__item ${mode === item.id ? "segmented-control__item--active" : ""}`}
              onClick={() => setMode(item.id)}
              disabled={item.status === "coming_later"}
            >
              {item.title}
            </button>
          ))}
        </div>
      </div>

      {/* 2. Model / Checkpoint */}
      <div className="config-panel__section">
        <label className="config-panel__label" htmlFor="model-family">Model</label>
        <div className="select-wrapper">
          <select id="model-family" className="select-field" value={selectedModelFamily || ""} onChange={(e) => setModelFamily(e.target.value)}>
            {modelFamilies.map((item) => (
              <option key={item.id} value={item.id}>
                {item.display_name}{item.runnable ? "" : ` (${item.blocked_reason || "unavailable"})`}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="config-panel__section">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <label className="config-panel__label" htmlFor="model-version" style={{ margin: 0 }}>Checkpoint</label>
          <button
            type="button"
            className="action-button action-button--secondary"
            onClick={() => checkpointInputRef.current?.click()}
            disabled={!selectedModelFamily || isRunning}
            style={{ padding: "3px 8px", fontSize: "0.62rem" }}
          >
            Upload
          </button>
          <input
            ref={checkpointInputRef}
            aria-label="Upload base checkpoint"
            type="file"
            accept={modelFamilies.find((item) => item.id === selectedModelFamily)?.checkpoint_extensions?.join(",") || ".pt,.pth"}
            style={{ display: "none" }}
            onChange={handleCheckpointFile}
          />
        </div>
        <div className="select-wrapper">
          <select id="model-version" className="select-field" value={selectedModelVersion || ""} onChange={(e) => setSelectedModelVersion(e.target.value)}>
            {checkpoints.map((item) => (
              <option key={item.id} value={item.id} disabled={!item.runnable}>
                {item.model_name} — {item.id}
              </option>
            ))}
          </select>
        </div>
        {blocked && (
          <small style={{ color: "var(--warning)", marginTop: "2px", display: "block", fontSize: "0.62rem" }}>
            Validated checkpoint required for Attack.
          </small>
        )}
        {checkpointUpload && <small role="status" style={{ display: "block", marginTop: 4, fontSize: "0.62rem", color: "var(--text-tertiary)" }}>{checkpointUpload}</small>}
      </div>

      {/* 3. Dataset */}
      <div className="config-panel__section">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <label className="config-panel__label" style={{ margin: 0 }}>Dataset</label>
          <button
            type="button"
            className="action-button action-button--secondary"
            onClick={() => setShowUploadModal(true)}
            style={{ padding: "3px 8px", fontSize: "0.62rem" }}
          >
            Import
          </button>
        </div>
        <div className="select-wrapper">
          <select className="select-field" value={selectedDataset} onChange={(e) => setSelectedDataset(e.target.value)}>
            {datasets.map((ds) => (
              <option key={ds.id || ds.name} value={ds.id || ds.name}>
                {ds.title || ds.name || ds.id}
                {ds.benchmark_ready === false ? " (inference only)" : ""}
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
            if (addDataset) { addDataset(newDs); }
            else { setSelectedDataset(newDs.id || newDs.name); }
          }
        }}
        datasets={datasets}
        taskId={mode}
      />

      {/* 4. Attacks — 3 tabs + accordion cards */}
      <div className="config-panel__section">
        <label className="config-panel__label">Attacks ({selectedAttacks.length})</label>
        <div className="attack-tabs">
          {ATTACK_TAB_DEFS.map(([tabId, label]) => (
            <button
              key={tabId}
              type="button"
              className={`attack-tabs__item ${attackTab === tabId ? "attack-tabs__item--active" : ""}`}
              onClick={() => setAttackTab(tabId)}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="attack-grid stagger-children" style={{ marginTop: "6px" }}>
          {visibleAttacks.map((atk) => {
            const isSelected = selectedAttacks.includes(atk.name);
            const isExpanded = expandedAttack === atk.name;
            const severity = getAttackSeverity(atk.name);
            const isAvailable = atk.available !== false;

            return (
              <div
                key={atk.name}
                className={`attack-card ${isSelected ? "attack-card--selected" : ""} ${isExpanded ? "attack-card--expanded" : ""} ${!isAvailable ? "attack-card--disabled" : ""}`}
                title={!isAvailable ? `Không khả dụng: ${atk.reason || "Không tương thích với mô hình hoặc dữ liệu hiện tại"}` : undefined}
              >
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    width: "100%",
                    cursor: isAvailable ? "pointer" : "not-allowed",
                  }}
                  onClick={() => isAvailable && handleAttackCardClick(atk.name)}
                  role="button"
                  tabIndex={isAvailable ? 0 : -1}
                  aria-disabled={!isAvailable}
                >
                  <div className="attack-card__left">
                    <span className="attack-card__name">{atk.name.replace(/_/g, " ")}</span>
                  </div>
                  <div className="attack-card__right">
                    {!isAvailable && (
                      <span className="attack-card__disabled-badge">
                        {formatDisabledReason(atk.reason)}
                      </span>
                    )}
                    {isSelected && (
                      <button
                        type="button"
                        className="attack-card__remove-btn"
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleAttack(atk.name);
                        }}
                        title={`Bỏ chọn ${atk.name.replace(/_/g, " ")}`}
                        aria-label={`Remove ${atk.name}`}
                      >
                        ✕
                      </button>
                    )}
                  </div>
                </div>

                {/* Single unified severity level selector */}
                {isSelected && (
                  <div className="severity-expanded" style={{ marginTop: "4px" }}>
                    <span style={{ fontSize: "0.62rem", color: "var(--text-tertiary)", alignSelf: "center", marginRight: "2px" }}>
                      Sev:
                    </span>
                    {[1, 2, 3, 4, 5].map((n) => (
                      <button
                        key={n}
                        type="button"
                        className={`severity-expanded__btn ${severity === n ? "severity-expanded__btn--active" : ""}`}
                        onClick={(e) => { e.stopPropagation(); handleSeveritySelect(atk.name, n); }}
                        title={`Severity level ${n}`}
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
          {visibleAttacks.length === 0 && (
            <div className="text-xs text-secondary" style={{ padding: "8px 0" }}>No attacks in this category.</div>
          )}
        </div>
      </div>

      {/* 5. Progress Bar (inline, when running) */}
      {isRunning && (
        <div className="progress-inline">
          <div className="progress-inline__text">
            <strong>{progressDetail || "Processing..."}</strong>
            <span>{progress}%</span>
          </div>
          <div className="progress-inline__bar-track">
            <div className="progress-inline__bar-fill" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {/* Run / Cancel Button */}
      {isRunning ? (
        <button
          type="button"
          className="run-button"
          onClick={cancelRunAction}
          style={{
            background: "var(--danger-dim, rgba(239, 68, 68, 0.15))",
            borderColor: "var(--danger, #ef4444)",
            color: "var(--danger, #ef4444)",
            cursor: "pointer",
          }}
        >
          ✕ Cancel Run
        </button>
      ) : (
        <button
          type="button"
          className="run-button"
          onClick={handleRun}
          disabled={blocked || selectedAttacks.length === 0}
        >
          {blocked ? "Waiting for artifacts" : "Run Test"}
        </button>
      )}
    </aside>
  );
}
