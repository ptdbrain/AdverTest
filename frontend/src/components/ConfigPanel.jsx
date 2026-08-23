import React, { useRef, useState } from "react";
import UploadModal from "./UploadModal";
import { getCheckpoint, uploadCheckpoint } from "@/lib/api";
import { SHORT_ATTACK_LABELS, ATTACK_LABELS } from "@/lib/attackNaming";

const GROUP_LABELS = {
  A: "Corruption",
  B: "Weather",
  C: "Occlusion",
  D: "Adversarial",
  E: "Patch",
  F: "Blackbox",
};

const ATTACK_TAB_DEFS = [
  ["white", "White-box"],
  ["black", "Black-box / query"],
  ["gray", "Real-world / corruption / weather / sensor"],
];

function attackSection(attack) {
  if (
    attack.threat_model === "white_box" ||
    attack.group === "D" ||
    ["fgsm", "pgd", "dag", "cw_l2", "bim", "deepfool", "apgd"].includes(attack.name)
  ) {
    return "white";
  }
  if (
    attack.threat_model === "black_box" ||
    attack.attack_type === "query" ||
    attack.group === "F" ||
    ["square", "nes"].includes(attack.name)
  ) {
    return "black";
  }
  return "gray";
}

function SeveritySquares({ level = 3 }) {
  return (
    <div className="severity-squares" title={`Severity ${level}/5`}>
      {[1, 2, 3, 4, 5].map((i) => (
        <span
          key={i}
          className={`severity-squares__dot ${i <= level ? "severity-squares__dot--filled" : ""}`}
        />
      ))}
    </div>
  );
}

export default function ConfigPanel({
  datasets = [],
  attacks = [],
  modes = [],
  modelFamilies = [],
  baseCheckpoints = [],
  recipePresets = [],
  mode,
  selectedModelFamily,
  selectedModelVersion,
  selectedDataset,
  selectedAttacks = [],
  recipe = { steps: [] },
  runOptions = { seed: 42, limit: 8, split: "val", difficulty: "all", confidence: 0.25, iou: 0.5 },
  runId = null,
  activeRunMeta = null,
  isRunning = false,
  progress = 0,
  progressDetail = "",
  runStatus,
  actions = {},
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
    setRunOptions,
    refreshBaseCheckpoints,
    cancelRun: cancelRunAction,
  } = actions;

  const [attackTab, setAttackTab] = useState("white");
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [checkpointUpload, setCheckpointUpload] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const checkpointInputRef = useRef(null);

  const checkpoints = baseCheckpoints.filter(
    (item) => item.task === mode && (!selectedModelFamily || item.model_family_id === selectedModelFamily)
  );
  const blocked = !checkpoints.find((item) => item.id === selectedModelVersion)?.runnable;

  const handleCheckpointFile = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setCheckpointUpload("Uploading: 0%");
    try {
      const created = await uploadCheckpoint(
        file,
        {
          taskId: mode,
          familyId: selectedModelFamily,
          displayName: file.name,
          role: "base",
        },
        (ratio) => setCheckpointUpload(`Uploading: ${Math.round(ratio * 100)}%`)
      );
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

  const getAttackSeverity = (attackName) => {
    const step = recipe.steps.find((s) => s.attack_name === attackName);
    return step?.severity ?? 3;
  };

  return (
    <aside className="config-panel" aria-label="Configuration Panel">
      {/* 1. Task Selector */}
      <div className="config-panel__section">
        <label className="config-panel__label" htmlFor="perception-mode">Task</label>
        <div className="select-wrapper">
          <select
            id="perception-mode"
            aria-label="Task"
            className="select-field"
            value={mode || ""}
            onChange={(e) => setMode?.(e.target.value)}
          >
            {modes.map((item) => (
              <option key={item.id} value={item.id} disabled={item.status === "coming_later"}>
                {item.title}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* 2. Model / Base Checkpoint */}
      <div className="config-panel__section">
        <label className="config-panel__label" htmlFor="model-family">Model</label>
        <div className="select-wrapper">
          <select
            id="model-family"
            aria-label="Model"
            className="select-field"
            value={selectedModelFamily || ""}
            onChange={(e) => setModelFamily?.(e.target.value)}
          >
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
          <label className="config-panel__label" htmlFor="model-version" style={{ margin: 0 }}>Base Checkpoint</label>
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
          <select
            id="model-version"
            aria-label="Base Checkpoint"
            className="select-field"
            value={selectedModelVersion || ""}
            onChange={(e) => setSelectedModelVersion?.(e.target.value)}
          >
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
        {checkpointUpload && (
          <small role="status" style={{ display: "block", marginTop: 4, fontSize: "0.62rem", color: "var(--text-tertiary)" }}>
            {checkpointUpload}
          </small>
        )}
      </div>

      {/* 3. Dataset */}
      <div className="config-panel__section">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <label className="config-panel__label" htmlFor="dataset-select" style={{ margin: 0 }}>Dataset</label>
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
          <select
            id="dataset-select"
            aria-label="Dataset"
            className="select-field"
            value={selectedDataset || ""}
            onChange={(e) => setSelectedDataset?.(e.target.value)}
          >
            {datasets.map((ds) => (
              <option key={ds.id || ds.name} value={ds.id || ds.name}>
                {ds.title || ds.name || ds.id}
                {ds.benchmark_ready === false ? " (inference only)" : ""}
              </option>
            ))}
          </select>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "4px", fontSize: "0.62rem", color: "var(--success, #10B981)" }}>
          <span>🛡️</span>
          <span>Privacy Anonymized (Face & License Plates blurred)</span>
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
              setSelectedDataset?.(newDs.id || newDs.name);
            }
          }
        }}
        datasets={datasets}
        taskId={mode}
      />

      {/* 3b. Dataset Split */}
      <div className="config-panel__section">
        <label className="config-panel__label" htmlFor="dataset-split">
          Split
        </label>
        <div className="select-wrapper">
          <select
            id="dataset-split"
            aria-label="Dataset Split"
            className="select-field"
            value={runOptions?.split || "val"}
            onChange={(e) => setRunOptions?.({ split: e.target.value })}
          >
            <option value="val">Val (Validation)</option>
            <option value="train">Train (Training)</option>
            <option value="all">All (Full)</option>
          </select>
        </div>
      </div>

      {/* 4. Attack Tabs: White-box / Black-box / Gray-box */}
      <div className="config-panel__section">
        <label className="config-panel__label">Attacks ({selectedAttacks.length} selected)</label>
        <div className="attack-tabs" role="tablist">
          {ATTACK_TAB_DEFS.map(([tabId, label]) => (
            <button
              key={tabId}
              type="button"
              role="tab"
              aria-selected={attackTab === tabId}
              className={`attack-tabs__item ${attackTab === tabId ? "attack-tabs__item--active" : ""}`}
              onClick={() => setAttackTab(tabId)}
            >
              {label}
            </button>
          ))}
        </div>

        {ATTACK_TAB_DEFS.map(([tabId]) => {
          const sectionAttacks = attacks.filter((atk) => attackSection(atk) === tabId);
          const isCurrentTab = attackTab === tabId;

          return (
            <div
              key={tabId}
              className="attack-grid stagger-children"
              style={
                !isCurrentTab
                  ? { height: 0, overflow: "hidden", opacity: 0, margin: 0, padding: 0, pointerEvents: "none" }
                  : { marginTop: "6px" }
              }
            >
              {sectionAttacks.map((atk) => {
                const isSelected = selectedAttacks.includes(atk.name);
                const severity = getAttackSeverity(atk.name);
                const isAvailable = atk.available !== false;

                return (
                  <button
                    key={atk.name}
                    type="button"
                    className={`attack-card ${isSelected ? "attack-card--selected" : ""}`}
                    onClick={() => isAvailable && toggleAttack?.(atk.name)}
                    disabled={!isAvailable}
                    title={!isAvailable ? (atk.reason || "Unavailable") : undefined}
                  >
                    <div className="attack-card__left">
                      <span className="attack-card__name">{atk.name.replace(/_/g, " ")}</span>
                      <span className="attack-card__group">
                        {atk.attack_type?.replace(/_/g, " ") || GROUP_LABELS[atk.group] || atk.threat_model?.replace(/_/g, " ") || atk.group}
                      </span>
                      {!isAvailable && (
                        <span className="attack-card__group" style={{ color: "var(--warning)" }}>
                          Unavailable: {atk.reason}
                        </span>
                      )}
                    </div>
                    <div className="attack-card__right">
                      {isSelected && <SeveritySquares level={severity} />}
                    </div>
                  </button>
                );
              })}
              {sectionAttacks.length === 0 && isCurrentTab && (
                <div className="text-xs text-secondary" style={{ padding: "8px 0", gridColumn: "1 / -1" }}>
                  No attacks in this category.
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* 5. Recipe Steps with Direct Sliders */}
      {recipe.steps.length > 0 && (
        <div className="config-panel__section">
          <label className="config-panel__label">Recipe ({recipe.steps.length} steps)</label>
          <div className="stagger-children" style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
            {recipe.steps.map((step) => (
              <label
                key={`${step.position}-${step.attack_name}`}
                className="recipe-step"
                style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.72rem" }}
              >
                <span>{SHORT_ATTACK_LABELS[step.attack_name] || ATTACK_LABELS[step.attack_name] || step.attack_name.replace(/_/g, " ")}</span>
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <span style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem", color: "var(--accent-light)" }}>
                    Cấp {step.severity}
                  </span>
                  <input
                    aria-label={`${step.attack_name} severity`}
                    type="range"
                    min="1"
                    max="5"
                    value={step.severity}
                    onChange={(event) => updateAttackSeverity?.(step.position, Number(event.target.value))}
                    style={{ width: "60px" }}
                  />
                </div>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* 6. Advanced Settings Drawer */}
      <div className="config-panel__section">
        <button
          type="button"
          className="text-xs text-secondary"
          onClick={() => setShowAdvanced((prev) => !prev)}
          style={{ background: "none", border: "none", cursor: "pointer", textDecoration: "underline", padding: 0 }}
        >
          {showAdvanced ? "▼ Hide Advanced Drawer" : "▶ Show Advanced Settings Drawer"}
        </button>

        {showAdvanced && (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: "6px",
              marginTop: "6px",
              padding: "6px",
              background: "var(--bg-deep)",
              borderRadius: "var(--radius-xs)",
            }}
          >
            <div>
              <label className="config-panel__label" htmlFor="run-seed" style={{ fontSize: "0.58rem", margin: "0 0 2px 0" }}>
                Seed
              </label>
              <input
                id="run-seed"
                aria-label="Seed"
                type="number"
                min="0"
                className="select-field"
                value={runOptions?.seed ?? 42}
                onChange={(event) => setRunOptions?.({ seed: Number(event.target.value) })}
                style={{ padding: "4px 6px", fontSize: "0.68rem", width: "100%" }}
              />
            </div>
            <div>
              <label className="config-panel__label" htmlFor="run-limit" style={{ fontSize: "0.58rem", margin: "0 0 2px 0" }}>
                Sample Limit
              </label>
              <input
                id="run-limit"
                aria-label="Sample Limit"
                type="number"
                min="1"
                className="select-field"
                value={runOptions?.limit ?? 8}
                onChange={(event) => setRunOptions?.({ limit: Number(event.target.value) })}
                style={{ padding: "4px 6px", fontSize: "0.68rem", width: "100%" }}
              />
            </div>
            <div>
              <label className="config-panel__label" htmlFor="run-difficulty" style={{ fontSize: "0.58rem", margin: "0 0 2px 0" }}>
                Difficulty
              </label>
              <div className="select-wrapper">
                <select
                  id="run-difficulty"
                  aria-label="Difficulty"
                  className="select-field"
                  value={runOptions?.difficulty || "all"}
                  onChange={(e) => setRunOptions?.({ difficulty: e.target.value })}
                  style={{ padding: "4px 6px", fontSize: "0.68rem" }}
                >
                  <option value="all">All (All Boxes)</option>
                  <option value="moderate">Moderate (Benchmark)</option>
                  <option value="easy">Easy</option>
                  <option value="hard">Hard</option>
                </select>
              </div>
            </div>
            <div>
              <label className="config-panel__label" htmlFor="run-iou" style={{ fontSize: "0.58rem", margin: "0 0 2px 0" }}>
                IoU Threshold
              </label>
              <input
                id="run-iou"
                aria-label="IoU Threshold"
                type="number"
                min="0"
                max="1"
                step="0.01"
                className="select-field"
                value={runOptions?.iou ?? 0.5}
                onChange={(event) => setRunOptions?.({ iou: Number(event.target.value) })}
                style={{ padding: "4px 6px", fontSize: "0.68rem", width: "100%" }}
              />
            </div>
          </div>
        )}
      </div>

      {/* 7. Pre-flight Workload & Cost Complexity Card */}
      {!isRunning && selectedAttacks.length > 0 && (() => {
        const nSamples = runOptions?.limit ?? 8;
        const nAttacks = selectedAttacks.length;
        const hasExpensive = selectedAttacks.some((a) => ["cw_l2", "square_attack", "dag"].includes(a));
        const hasMedium = selectedAttacks.some((a) => ["pgd", "fgsm", "mi_fgsm", "tog", "sam2_pgd"].includes(a));
        const complexityTier = hasExpensive ? "🔥 Nặng (Iterative Optimization)" : hasMedium ? "⚙️ Trung bình (Gradient Steps)" : "⚡ Nhẹ (Single-pass Transform)";
        const complexityColor = hasExpensive ? "var(--danger)" : hasMedium ? "var(--warning)" : "var(--success)";
        const estForwardPasses = (nAttacks + 1) * nSamples;

        return (
          <div
            style={{
              background: "var(--bg-elevated, #1A2230)",
              padding: "8px 10px",
              borderRadius: "var(--radius-sm, 6px)",
              border: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.12))",
              display: "flex",
              flexDirection: "column",
              gap: "4px",
              fontSize: "0.68rem",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontWeight: 700, color: "var(--text-secondary)" }}>⚡ Khối lượng tính toán (Workload Budget)</span>
              <span style={{ color: complexityColor, fontWeight: 700, fontSize: "0.62rem" }}>
                {complexityTier}
              </span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px", color: "var(--text-muted)", marginTop: "2px" }}>
              <div>Số phép thử (Cells): <strong style={{ color: "var(--text-primary)" }}>{nAttacks} đòn × {nSamples} mẫu</strong></div>
              <div>Ước tính Forward Passes: <strong style={{ color: "var(--text-primary)" }}>{estForwardPasses} lần</strong></div>
              <div>Prediction Caching: <strong style={{ color: "var(--success)" }}>SQLite Enabled</strong></div>
              <div>Tài nguyên yêu cầu: <strong style={{ color: "var(--text-primary)" }}>Device Inferred</strong></div>
            </div>
          </div>
        );
      })()}

      {/* 8. Active Execution Monitor */}
      {isRunning && (
        <div
          style={{
            background: "rgba(56, 189, 248, 0.08)",
            border: "1px solid rgba(56, 189, 248, 0.25)",
            borderRadius: "8px",
            padding: "12px",
            display: "flex",
            flexDirection: "column",
            gap: "8px",
            margin: "4px 0",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.72rem", fontWeight: 800, color: "var(--accent)", display: "flex", alignItems: "center", gap: "6px" }}>
              <span className="status-beacon status-beacon--running" style={{ width: 8, height: 8 }} />
              ĐANG CHẠY PHIÊN #{runId ? runId.slice(0, 8) : "INITIALIZING"}
            </span>
            <span style={{ fontSize: "0.78rem", fontWeight: 800, color: "var(--text-primary)", fontFamily: "var(--font-mono)" }}>
              {progress}%
            </span>
          </div>

          <div className="progress-inline__bar-track" style={{ height: "6px", background: "var(--bg-primary)" }}>
            <div className="progress-inline__bar-fill" style={{ width: `${progress}%`, background: "var(--accent)" }} />
          </div>

          <div style={{ fontSize: "0.68rem", color: "var(--text-secondary)", lineHeight: 1.4 }}>
            <div>📌 <strong>Đòn tấn công:</strong> {activeRunMeta?.attacks?.length ? activeRunMeta.attacks.map((a) => SHORT_ATTACK_LABELS[a] || ATTACK_LABELS[a] || a).join(" + ") : (selectedAttacks.length ? selectedAttacks.map((a) => SHORT_ATTACK_LABELS[a] || a).join(", ") : "Tự chọn")}</div>
            <div>🤖 <strong>Mô hình:</strong> {activeRunMeta?.modelVersion || selectedModelVersion || "YOLO11"} | 📦 <strong>Dữ liệu:</strong> {activeRunMeta?.dataset || selectedDataset}</div>
            <div style={{ color: "var(--text-muted)", marginTop: "2px" }}>{progressDetail || "Đang xử lý gradient đối kháng..."}</div>
          </div>
        </div>
      )}

      {/* 9. Run & Control Buttons */}
      {isRunning ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <button
            type="button"
            className="run-button"
            onClick={cancelRunAction}
            style={{
              background: "rgba(239, 68, 68, 0.15)",
              borderColor: "#EF4444",
              color: "#EF4444",
              cursor: "pointer",
              fontWeight: 800,
              fontSize: "0.82rem",
              padding: "10px 14px",
            }}
          >
            🛑 HỦY & DỪNG PHIÊN NÀY NGAY
          </button>

          <button
            type="button"
            className="run-button"
            onClick={handleRun}
            style={{
              background: "rgba(56, 189, 248, 0.12)",
              borderColor: "var(--accent)",
              color: "var(--accent)",
              cursor: "pointer",
              fontWeight: 700,
              fontSize: "0.75rem",
              padding: "8px 12px",
            }}
            title="Hủy ngay phiên chạy cũ và bắt đầu chạy lại với cấu hình bạn vừa thay đổi"
          >
            ⚡ Hủy Cũ & Chạy Ngay Với Cấu Hình Mới
          </button>
        </div>
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
