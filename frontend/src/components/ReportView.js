import React from "react";
import HeatmapMatrix from "@/components/HeatmapMatrix";
import RAChart from "@/components/RAChart";

/**
 * ReportView Component
 * Renders the robustness report with summary header, Heatmap, and RA Curve.
 */
export default function ReportView({ report }) {
  if (!report) {
    return (
      <div className="placeholder-view">
        <div className="placeholder-view__title">No report yet</div>
        <div className="placeholder-view__subtitle">
          Configure attacks on the left panel and run to see the Robustness Report with heatmap, RA curve, and per-attack metrics.
        </div>
      </div>
    );
  }

  const avgDegradation = report.cells.length > 0
    ? report.cells.reduce((sum, c) => sum + c.degradation, 0) / report.cells.length
    : 0;

  return (
    <>
      {/* Summary Header */}
      <div className="chart-container" style={{ padding: "var(--space-md)", display: "flex", gap: "var(--space-xl)", alignItems: "center", flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: "0.6rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 600 }}>Model</div>
          <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--cyan)", letterSpacing: "-0.02em" }}>
            {report.model} <span style={{ fontSize: "0.75rem", color: "var(--text-tertiary)", fontWeight: 400 }}>{report.model_version}</span>
          </div>
        </div>
        <div>
          <div style={{ fontSize: "0.6rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 600 }}>Dataset</div>
          <div style={{ fontSize: "0.85rem", fontWeight: 500 }}>{report.dataset}</div>
        </div>
        <div>
          <div style={{ fontSize: "0.6rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 600 }}>AP Clean</div>
          <div className="text-mono text-success" style={{ fontSize: "1rem", fontWeight: 600 }}>
            {report.ap_clean?.toFixed(3)}
          </div>
        </div>
        <div>
          <div style={{ fontSize: "0.6rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 600 }}>Avg Degradation</div>
          <div className="text-mono text-danger" style={{ fontSize: "1rem", fontWeight: 600 }}>
            {avgDegradation.toFixed(1)}%
          </div>
        </div>
        <div>
          <div style={{ fontSize: "0.6rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 600 }}>Duration</div>
          <div className="text-mono" style={{ fontSize: "0.85rem", fontWeight: 500, color: "var(--text-secondary)" }}>
            {report.seconds?.toFixed(1)}s
          </div>
        </div>
      </div>

      <HeatmapMatrix cells={report.cells} heatmap={report.heatmap} />
      <RAChart cells={report.cells} apClean={report.ap_clean} />
    </>
  );
}
