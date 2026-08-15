import React from "react";
import HeatmapMatrix from "@/components/HeatmapMatrix";
import RAChart from "@/components/RAChart";
import MetricsComparisonChart from "@/components/MetricsComparisonChart";

export default function ReportView({ report }) {
  if (!report) {
    return (
      <div className="placeholder-view">
        <div className="placeholder-view__title">No report yet</div>
        <div className="placeholder-view__subtitle">
          Configure and run attacks to generate the robustness report with heatmap and RA curve.
        </div>
      </div>
    );
  }

  const avgDegradation = report.cells.length > 0
    ? report.cells.reduce((sum, c) => sum + c.degradation, 0) / report.cells.length
    : 0;

  return (
    <div style={{ flex: 1, padding: "var(--space-md)", overflow: "auto", display: "flex", flexDirection: "column", gap: "var(--space-md)" }}>
      {/* Summary Header - Bold & Oversized Typography */}
      <div className="chart-container" style={{ padding: "var(--space-lg)", display: "flex", gap: "var(--space-2xl)", alignItems: "center", flexWrap: "wrap", borderTop: "3px solid var(--accent)" }}>
        
        {/* Model & Dataset Details */}
        <div style={{ flex: 1, minWidth: "200px" }}>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>Model</div>
          <div style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
            {report.model} <span style={{ fontSize: "0.85rem", color: "var(--text-tertiary)", fontWeight: 500 }}>{report.model_version}</span>
          </div>
          
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2, marginTop: 12 }}>Dataset</div>
          <div style={{ fontSize: "0.95rem", fontWeight: 600, color: "var(--text-secondary)" }}>{report.dataset}</div>
        </div>

        {/* AP Clean Oversized */}
        <div style={{ padding: "0 var(--space-lg)", borderLeft: "1px solid var(--border-subtle)" }}>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>AP Clean</div>
          <div className="text-mono" style={{ fontSize: "2.4rem", fontWeight: 800, color: "var(--success)", lineHeight: 1, letterSpacing: "-0.04em" }}>
            {report.benchmark_metrics_available === false ? "—" : report.ap_clean?.toFixed(3)}
          </div>
        </div>

        {/* Avg Degradation Gauge */}
        <div style={{ display: "flex", alignItems: "center", gap: "var(--space-md)", padding: "0 var(--space-lg)", borderLeft: "1px solid var(--border-subtle)" }}>
          <div>
            <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>Avg Degradation</div>
            <div className="text-mono" style={{ fontSize: "2.4rem", fontWeight: 800, color: "var(--danger)", lineHeight: 1, letterSpacing: "-0.04em" }}>
              {report.benchmark_metrics_available === false ? "—" : `-${avgDegradation.toFixed(1)}%`}
            </div>
          </div>
          {report.benchmark_metrics_available !== false && (
            <div style={{ position: "relative", width: 56, height: 56 }}>
              <svg viewBox="0 0 36 36" style={{ width: "100%", height: "100%", transform: "rotate(-90deg)" }}>
                <circle cx="18" cy="18" r="16" fill="none" stroke="var(--bg-elevated)" strokeWidth="4" />
                <circle 
                  cx="18" cy="18" r="16" fill="none" 
                  stroke={avgDegradation > 40 ? "var(--danger)" : avgDegradation > 15 ? "var(--warning)" : "var(--success)"} 
                  strokeWidth="4" 
                  strokeDasharray={`${Math.min(100, Math.max(0, avgDegradation))} 100`}
                  strokeLinecap="round"
                  style={{ transition: "stroke-dasharray 1s var(--ease-out)" }}
                />
              </svg>
            </div>
          )}
        </div>
        
      </div>

      {/* Bar Chart comparing metrics before and after attack */}
      <MetricsComparisonChart report={report} />

      {report.benchmark_metrics_available !== false && (
        <>
          <HeatmapMatrix cells={report.cells} heatmap={report.heatmap} />
          <RAChart cells={report.cells} apClean={report.ap_clean} />
        </>
      )}
    </div>
  );
}
