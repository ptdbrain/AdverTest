import React from "react";

function degradationColor(d) {
  if (d < 15) return "var(--success)";
  if (d < 40) return "var(--warning)";
  return "var(--danger)";
}

function badgeClass(d) {
  if (d < 15) return "model-card__badge--good";
  if (d < 40) return "model-card__badge--moderate";
  return "model-card__badge--bad";
}

/**
 * ModelResultsPanel Component
 * Renders the right sidebar with RobustScore, model performance, and per-attack breakdown.
 */
export default function ModelResultsPanel({ report, models }) {
  return (
    <aside className="results-panel">
      <div className="results-panel__header">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 002 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0022 16z" />
          <path d="M3.27 6.96L12 12.01l8.73-5.05" />
          <path d="M12 22.08V12" />
        </svg>
        Model Performance
      </div>

      {!report ? (
        <>
          {models.map((m) => (
            <div key={m.name} className="model-card">
              <div className="model-card__header">
                <span className="model-card__name">{m.name}</span>
                <span className={`status-badge ${m.runnable ? "status-badge--completed" : "status-badge--queued"}`}>
                  {m.runnable ? "Ready" : "N/A"}
                </span>
              </div>
              <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>
                {m.task} · {m.modality}
              </div>
            </div>
          ))}
          <div className="empty-state" style={{ padding: "var(--space-lg)" }}>
            <div className="empty-state__message">Run an attack to see performance metrics</div>
          </div>
        </>
      ) : (
        <>
          {/* RobustScore — read from backend metrics.robustness */}
          {(() => {
            const robustness = report.metrics?.robustness;
            const scoreValue = robustness?.robust_score_normalized
              ?? robustness?.robust_score_plan
              ?? null;
            const displayScore = scoreValue != null
              ? Math.round(scoreValue)
              : Math.round(
                  (1 -
                    report.cells.reduce((sum, c) => sum + c.degradation, 0) /
                      Math.max(1, report.cells.length) /
                      100) *
                    100
                );
            return (
              <div className="robust-score">
                <span className="robust-score__title">
                  RobustScore
                  {robustness?.normalize_score && (
                    <span style={{ fontSize: "0.5rem", color: "var(--text-muted)", marginLeft: 4, fontWeight: 400 }}>
                      ({robustness.covered_categories?.join(", ") || "all"})
                    </span>
                  )}
                </span>
                <span className="robust-score__value">{displayScore}</span>
                <div className="robust-score__bar">
                  <div
                    className="robust-score__fill"
                    style={{ width: `${displayScore}%` }}
                  />
                </div>
                {robustness?.rpc != null && (
                  <div style={{ display: "flex", justifyContent: "space-between", width: "100%", fontSize: "0.55rem", color: "var(--text-muted)", marginTop: 2 }}>
                    <span>RPC {(robustness.rpc * 100).toFixed(0)}%</span>
                    <span>MPC {robustness.mpc?.toFixed(3)}</span>
                    <span>AP {robustness.ap_clean?.toFixed(3)}</span>
                  </div>
                )}
              </div>
            );
          })()}

          {/* Model result card */}
          <div className="model-card" style={{ borderColor: "rgba(99, 179, 237, 0.2)", borderWidth: "1px" }}>
            <div className="model-card__header">
              <span className="model-card__name">{report.model}</span>
              <span
                className={`model-card__badge ${badgeClass(
                  report.cells.reduce((sum, c) => sum + c.degradation, 0) /
                    Math.max(1, report.cells.length)
                )}`}
              >
                {report.model_version}
              </span>
            </div>
            <div className="model-card__metrics">
              <div className="model-card__metric">
                <span className="model-card__metric-label">AP Clean</span>
                <span className="model-card__metric-value text-success">
                  {report.ap_clean?.toFixed(3)}
                </span>
              </div>
              <div className="model-card__metric">
                <span className="model-card__metric-label">Avg. Degradation</span>
                <span className="model-card__metric-value text-danger">
                  {(
                    report.cells.reduce((sum, c) => sum + c.degradation, 0) /
                    Math.max(1, report.cells.length)
                  ).toFixed(1)}
                  %
                </span>
              </div>
            </div>
            <div className="model-card__degradation-bar">
              <div
                className="model-card__degradation-fill"
                style={{
                  width: `${Math.min(
                    100,
                    report.cells.reduce((sum, c) => sum + c.degradation, 0) /
                      Math.max(1, report.cells.length)
                  )}%`,
                  background: degradationColor(
                    report.cells.reduce((sum, c) => sum + c.degradation, 0) /
                      Math.max(1, report.cells.length)
                  ),
                }}
              />
            </div>
          </div>

          {/* Per-cell breakdown */}
          <div className="config-panel__section">
            <label className="config-panel__label">Per-Attack Results</label>
            {report.cells.map((cell, i) => (
              <div key={i} className="model-card" style={{ padding: "10px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div>
                    <div style={{ fontSize: "0.7rem", fontWeight: 600, textTransform: "capitalize" }}>
                      {cell.attack.replace(/_/g, " ")}
                    </div>
                    <div className="text-xs" style={{ color: "var(--text-muted)" }}>Severity {cell.severity}</div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <div
                      className="text-mono font-bold"
                      style={{ color: degradationColor(cell.degradation), fontSize: "0.85rem" }}
                    >
                      {cell.degradation > 0 ? "\u2193" : ""}
                      {cell.degradation.toFixed(1)}%
                    </div>
                    <div className="text-xs text-mono" style={{ color: "var(--text-muted)" }}>
                      AP {cell.ap.toFixed(3)}
                    </div>
                  </div>
                </div>
                <div className="model-card__degradation-bar">
                  <div
                    className="model-card__degradation-fill"
                    style={{
                      width: `${Math.min(100, cell.degradation)}%`,
                      background: degradationColor(cell.degradation),
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Actions */}
          <div className="action-buttons">
            <button className="action-button action-button--primary">Export</button>
            <button className="action-button action-button--warning">Flag Review</button>
          </div>
        </>
      )}
    </aside>
  );
}
