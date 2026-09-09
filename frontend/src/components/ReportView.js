"use client";

import { Download, FileCode, FileSpreadsheet, Printer } from "lucide-react";
import React from "react";
import AccumulatedMetricsTable from "@/components/AccumulatedMetricsTable";
import EvidenceBadge from "@/components/common/EvidenceBadge";
import HeatmapMatrix from "@/components/HeatmapMatrix";
import MethodComparisonTable from "@/components/MethodComparisonTable";
import MetricsComparisonChart from "@/components/MetricsComparisonChart";
import RAChart from "@/components/RAChart";
import { useLanguage } from "@/context/LanguageContext";
import { triggerAutoFlag } from "@/lib/api";
import { getDescriptiveAttackName } from "@/lib/attackNaming";
import { exportReportAsCsv, exportReportAsJson, printReportAsPdf } from "@/lib/exportReport";
import { buildRunDecisionView } from "@/lib/reportMetrics";

export default function ReportView({ report, onClearHistory }) {
  const { t } = useLanguage();
  const [flagging, setFlagging] = React.useState(false);
  const [flagResult, setFlagResult] = React.useState(null);

  if (!report) {
    return (
      <div className="placeholder-view">
        <div className="placeholder-view__title">{t("report.emptyTitle")}</div>
        <div className="placeholder-view__subtitle">{t("report.emptySubtitle")}</div>
      </div>
    );
  }

  const cells = report.cells || [];
  const decisionView = buildRunDecisionView(report);
  const avgDegradation = decisionView.degradationPercent;
  // ASR has attack-specific semantics. Only show it when the backend reports
  // an explicit measured value; never reinterpret a degradation threshold as ASR.
  const asrPct =
    report.metrics?.robustness?.attack_success_rate_percent ??
    (report.metrics?.robustness?.attack_success_rate != null
      ? report.metrics.robustness.attack_success_rate * 100
      : null);

  const cleanIoU = report.metrics?.clean?.mean_iou ?? report.metrics?.clean?.mean_bev_iou;
  const cleanIoUDisplay = cleanIoU != null ? cleanIoU.toFixed(3) : "—";
  const iouKey = report.metrics?.clean?.mean_bev_iou != null ? "mean_bev_iou" : "mean_iou";
  const attackedIoUValues = cells
    .map((cell) => cell.metrics?.[iouKey])
    .filter((value) => typeof value === "number" && Number.isFinite(value));
  const attackedIoU = attackedIoUValues.length
    ? attackedIoUValues.reduce((sum, value) => sum + value, 0) / attackedIoUValues.length
    : null;
  const attackedIoUDisplay = attackedIoU != null ? attackedIoU.toFixed(3) : "—";

  const handleAutoFlag = async () => {
    if (!report?.run_id) return;
    setFlagging(true);
    try {
      const res = await triggerAutoFlag(report.run_id, 20);
      setFlagResult(res);
    } catch (err) {
      console.error("Auto-flag failed:", err);
    } finally {
      setFlagging(false);
    }
  };

  const uniqueAttacks = Array.from(new Set(cells.map((c) => getDescriptiveAttackName(c, c.severity, report))));
  const isAccumulated = uniqueAttacks.length > 1 || (report.accumulated_runs && report.accumulated_runs.length > 1);

  return (
    <div
      style={{
        flex: 1,
        padding: "var(--space-md)",
        overflow: "auto",
        display: "flex",
        flexDirection: "column",
        gap: "var(--space-md)",
      }}
    >
      {/* Multi-Run Accumulation Status Header */}
      {isAccumulated && (
        <div
          style={{
            background: "rgba(56, 189, 248, 0.08)",
            border: "1px solid rgba(56, 189, 248, 0.25)",
            borderRadius: "var(--radius-md, 8px)",
            padding: "12px 18px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "10px",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <span style={{ fontSize: "1.4rem" }}>📊</span>
            <div>
              <div style={{ fontSize: "0.85rem", fontWeight: 800, color: "var(--accent, #38BDF8)" }}>
                {t("report.multiRunTitle")}
              </div>
              <div
                style={{ fontSize: "0.72rem", color: "var(--text-secondary)", marginTop: "2px" }}
                dangerouslySetInnerHTML={{
                  __html: t("report.multiRunDesc", {
                    count: uniqueAttacks.length,
                    attacks: uniqueAttacks.join(", "),
                    dataset: report.dataset,
                    runs: report.accumulated_runs?.length || 1,
                  }),
                }}
              />
            </div>
          </div>

          {onClearHistory && (
            <button
              type="button"
              onClick={onClearHistory}
              className="action-button action-button--secondary"
              style={{ fontSize: "0.72rem", padding: "5px 12px", color: "var(--text-muted)" }}
              title={t("report.clearHistory.title")}
            >
              {t("report.clearHistory")}
            </button>
          )}
        </div>
      )}

      {/* Export Toolbar & Evidence State Header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "12px",
          padding: "12px 16px",
          background: "#FFFFFF",
          borderRadius: "8px",
          border: "1px solid #E2E8F0",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <EvidenceBadge provenance={report.provenance} simulationOnly={report.simulation_only} />
          <span style={{ fontSize: "0.8rem", color: "#64748B", fontWeight: 600 }}>
            Run ID: <strong style={{ color: "#0F172A" }}>{report.run_id || "—"}</strong>
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <button
            type="button"
            onClick={() => exportReportAsJson(report)}
            className="action-button action-button--secondary"
            style={{
              fontSize: "0.75rem",
              padding: "6px 12px",
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
            }}
            title="Xuất dữ liệu gốc kèm toàn bộ Scientific Provenance"
          >
            <FileCode style={{ width: 14, height: 14 }} />
            <span>JSON</span>
          </button>
          <button
            type="button"
            onClick={() => exportReportAsCsv(report)}
            className="action-button action-button--secondary"
            style={{
              fontSize: "0.75rem",
              padding: "6px 12px",
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
            }}
            title="Xuất bảng số liệu CSV chuẩn hóa advertest-export-v1"
          >
            <FileSpreadsheet style={{ width: 14, height: 14 }} />
            <span>CSV</span>
          </button>
          <button
            type="button"
            onClick={() => printReportAsPdf(report)}
            className="action-button"
            style={{
              background: "#2563EB",
              color: "#FFF",
              fontSize: "0.75rem",
              fontWeight: 700,
              padding: "6px 14px",
              borderRadius: "6px",
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
            }}
            title="In hoặc Lưu Báo cáo Thử nghiệm dạng PDF"
          >
            <Printer style={{ width: 14, height: 14 }} />
            <span>In / PDF</span>
          </button>
        </div>
      </div>

      {/* Summary Header - Complete 5-Metric Suite from TOPIC.md */}
      <div
        className="chart-container"
        style={{
          padding: "var(--space-lg)",
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: "var(--space-md)",
          alignItems: "center",
          borderTop: "3px solid var(--accent)",
        }}
      >
        {/* Model & Dataset Details */}
        <div>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>
            Model & Task
          </div>
          <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
            {report.model}{" "}
            <span style={{ fontSize: "0.8rem", color: "var(--text-tertiary)", fontWeight: 500 }}>
              {report.model_version}
            </span>
          </div>
          <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-secondary)", marginTop: 4 }}>
            Dataset: {report.dataset}
          </div>
        </div>

        {/* Baseline mAP */}
        <div style={{ padding: "0 var(--space-md)", borderLeft: "1px solid var(--border-subtle)" }}>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>
            {decisionView.primary.label} clean
          </div>
          <div
            className="text-mono"
            style={{ fontSize: "2rem", fontWeight: 800, color: "var(--success)", lineHeight: 1 }}
          >
            {decisionView.primary.clean == null ? "—" : decisionView.primary.clean.toFixed(3)}
          </div>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: 4 }}>
            {t("report.cleanAccuracy")}
          </div>
        </div>

        {/* Avg Degradation Gauge */}
        <div style={{ padding: "0 var(--space-md)", borderLeft: "1px solid var(--border-subtle)" }}>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>
            Avg Degradation
          </div>
          <div
            className="text-mono"
            style={{ fontSize: "2rem", fontWeight: 800, color: "var(--danger)", lineHeight: 1 }}
          >
            {avgDegradation == null ? "—" : `-${avgDegradation.toFixed(1)}%`}
          </div>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: 4 }}>
            {t("report.avgDegradation", { count: cells.length })}
          </div>
        </div>

        {/* Attack Success Rate (ASR) */}
        <div style={{ padding: "0 var(--space-md)", borderLeft: "1px solid var(--border-subtle)" }}>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>
            Attack Success (ASR)
          </div>
          <div
            className="text-mono"
            style={{
              fontSize: "2rem",
              fontWeight: 800,
              color: asrPct != null && asrPct > 50 ? "var(--danger)" : "var(--warning)",
              lineHeight: 1,
            }}
          >
            {asrPct == null ? "—" : `${asrPct.toFixed(1)}%`}
          </div>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: 4 }}>{t("report.asrLabel")}</div>
        </div>

        {/* Mean IoU Before vs After */}
        <div style={{ padding: "0 var(--space-md)", borderLeft: "1px solid var(--border-subtle)" }}>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>
            {t("report.meanIoU")}
          </div>
          <div
            className="text-mono"
            style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text-primary)", lineHeight: 1, marginTop: 4 }}
          >
            <span style={{ color: "var(--success)" }}>{cleanIoUDisplay}</span>
            <span style={{ color: "var(--text-tertiary)", margin: "0 4px" }}>→</span>
            <span style={{ color: "var(--danger)" }}>{attackedIoUDisplay}</span>
          </div>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: 4 }}>{t("report.iouLabel")}</div>
        </div>
      </div>

      {/* ── Accumulated AP / mAP / Robustness columns — headline comparison ──
           Cột: baseline | tổ hợp 1 | tổ hợp 2 ... thêm dần sau mỗi lần chạy,
           tích lũy cho tới khi dataset bị đổi (scoped trong useAdverTest).
           Hàng: AP@50 | mAP@50-95 | Robustness.
      ────────────────────────────────────────────────────────────────── */}
      {report.cells?.length > 0 && <AccumulatedMetricsTable report={report} />}

      {/* ── Method × Severity pivot table — hiện ngay sau summary ────────────
           Cột: baseline | {attack}_{sev} ... tích lũy qua từng run.
           Hàng: tên attack (không có suffix severity).
      ────────────────────────────────────────────────────────────────── */}
      {report.cells?.length > 0 && <MethodComparisonTable report={report} />}

      {/* Human-in-the-Loop Review Queue Action Banner */}
      <div
        style={{
          background: "linear-gradient(90deg, rgba(245, 158, 11, 0.08) 0%, rgba(30, 41, 59, 0.6) 100%)",
          border: "1px solid rgba(245, 158, 11, 0.25)",
          borderRadius: "var(--radius-md, 8px)",
          padding: "12px 18px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "12px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span style={{ fontSize: "1.3rem" }}>⚖️</span>
          <div>
            <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#FBBF24" }}>{t("report.reviewQueueTitle")}</div>
            <div style={{ fontSize: "0.72rem", color: "var(--text-secondary)" }}>
              {flagResult ? t("report.flaggedMsg", { count: flagResult.count }) : t("report.flagHint")}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <button
            type="button"
            onClick={handleAutoFlag}
            disabled={flagging || !report.run_id}
            className="action-button action-button--secondary"
            style={{ fontSize: "0.75rem", padding: "6px 12px" }}
          >
            {flagging ? t("report.flagging") : t("report.autoFlag")}
          </button>
          <a
            href="/reviews"
            className="action-button"
            style={{
              background: "#D97706",
              color: "#FFF",
              textDecoration: "none",
              fontSize: "0.75rem",
              fontWeight: 700,
              padding: "6px 14px",
              borderRadius: "6px",
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
            }}
          >
            <span>{t("report.openReviewQueue")}</span>
            <span>→</span>
          </a>
        </div>
      </div>

      {/* Bar Chart comparing metrics before and after attack */}
      <MetricsComparisonChart report={report} />

      {/* Aggregate Robustness Accuracy Curve */}
      {decisionView.dataState === "MEASURED" && (
        <RAChart cells={report.cells} apClean={decisionView.primary.clean} report={report} />
      )}

      {/* Degradation Heatmap Matrix across attacks & severities */}
      {report.cells?.length > 0 && <HeatmapMatrix cells={report.cells} heatmap={report.heatmap} report={report} />}
    </div>
  );
}
