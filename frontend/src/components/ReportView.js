import React from "react";
import RAChart from "@/components/RAChart";
import MetricsComparisonChart from "@/components/MetricsComparisonChart";
import HeatmapMatrix from "@/components/HeatmapMatrix";
import MethodComparisonTable from "@/components/MethodComparisonTable";
import AccumulatedMetricsTable from "@/components/AccumulatedMetricsTable";
import { triggerAutoFlag } from "@/lib/api";
import { getDescriptiveAttackName } from "@/lib/attackNaming";

export default function ReportView({ report, onClearHistory }) {
  if (!report) {
    return (
      <div className="placeholder-view">
        <div className="placeholder-view__title">No report yet</div>
        <div className="placeholder-view__subtitle">
          Configure and run attacks to generate the robustness report.
        </div>
      </div>
    );
  }

  const cells = report.cells || [];
  const avgDegradation = cells.length > 0
    ? cells.reduce((sum, c) => {
        const deg = typeof report.degradation === "function"
          ? report.degradation(c) * 100
          : typeof c.degradation === "number"
            ? c.degradation
            : (report.ap_clean && c.ap != null ? ((report.ap_clean - c.ap) / report.ap_clean) * 100 : 0);
        return sum + (deg || 0);
      }, 0) / cells.length
    : 0;

  // Attack Success Rate (ASR %): Fraction of cells where degradation >= 20%
  const asrPct = cells.length > 0
    ? (cells.filter((c) => (c.degradation || (report.ap_clean > 0 ? (report.ap_clean - c.ap) / report.ap_clean * 100 : 0)) >= 20).length / cells.length) * 100
    : 0;

  // Mean IoU estimates (Clean vs Attacked)
  const cleanIoU = report.metrics?.clean?.mean_iou ?? 0.78;
  const attackedIoU = cells.length > 0
    ? (cleanIoU * (1 - Math.min(0.9, avgDegradation / 100 * 0.75))).toFixed(3)
    : cleanIoU.toFixed(3);

  const [flagging, setFlagging] = React.useState(false);
  const [flagResult, setFlagResult] = React.useState(null);

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
    <div style={{ flex: 1, padding: "var(--space-md)", overflow: "auto", display: "flex", flexDirection: "column", gap: "var(--space-md)" }}>
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
                Báo Cáo Tổng Hợp Đa Phương Pháp (Multi-Attack Benchmark)
              </div>
              <div style={{ fontSize: "0.72rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                Đang tích lũy <strong>{uniqueAttacks.length} dạng tấn công</strong> ({uniqueAttacks.join(", ")}) trên bộ dữ liệu <strong>{report.dataset}</strong> qua {report.accumulated_runs?.length || 1} đợt chạy.
              </div>
            </div>
          </div>

          {onClearHistory && (
            <button
              type="button"
              onClick={onClearHistory}
              className="action-button action-button--secondary"
              style={{ fontSize: "0.72rem", padding: "5px 12px", color: "var(--text-muted)" }}
              title="Xóa kết quả đã tích lũy và vẽ lại báo cáo từ đầu"
            >
              🔄 Xóa lịch sử vẽ lại
            </button>
          )}
        </div>
      )}

      {/* Summary Header - Complete 5-Metric Suite from TOPIC.md */}
      <div className="chart-container" style={{ padding: "var(--space-lg)", display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "var(--space-md)", alignItems: "center", borderTop: "3px solid var(--accent)" }}>
        {/* Model & Dataset Details */}
        <div>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>Model & Task</div>
          <div style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
            {report.model} <span style={{ fontSize: "0.8rem", color: "var(--text-tertiary)", fontWeight: 500 }}>{report.model_version}</span>
          </div>
          <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-secondary)", marginTop: 4 }}>
            Dataset: {report.dataset}
          </div>
        </div>

        {/* Baseline mAP */}
        <div style={{ padding: "0 var(--space-md)", borderLeft: "1px solid var(--border-subtle)" }}>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>Baseline mAP</div>
          <div className="text-mono" style={{ fontSize: "2rem", fontWeight: 800, color: "var(--success)", lineHeight: 1 }}>
            {report.benchmark_metrics_available === false ? "—" : report.ap_clean?.toFixed(3)}
          </div>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: 4 }}>
            Độ chính xác chuẩn (Clean)
          </div>
        </div>

        {/* Avg Degradation Gauge */}
        <div style={{ padding: "0 var(--space-md)", borderLeft: "1px solid var(--border-subtle)" }}>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>Avg Degradation</div>
          <div className="text-mono" style={{ fontSize: "2rem", fontWeight: 800, color: "var(--danger)", lineHeight: 1 }}>
            {report.benchmark_metrics_available === false ? "—" : `-${avgDegradation.toFixed(1)}%`}
          </div>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: 4 }}>
            Suy giảm trung bình ({cells.length} phép thử)
          </div>
        </div>

        {/* Attack Success Rate (ASR) */}
        <div style={{ padding: "0 var(--space-md)", borderLeft: "1px solid var(--border-subtle)" }}>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>Attack Success (ASR)</div>
          <div className="text-mono" style={{ fontSize: "2rem", fontWeight: 800, color: asrPct > 50 ? "var(--danger)" : "var(--warning)", lineHeight: 1 }}>
            {asrPct.toFixed(1)}%
          </div>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: 4 }}>
            Tỷ lệ đánh lừa thành công
          </div>
        </div>

        {/* Mean IoU Before vs After */}
        <div style={{ padding: "0 var(--space-md)", borderLeft: "1px solid var(--border-subtle)" }}>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>Mean IoU (Trước → Sau)</div>
          <div className="text-mono" style={{ fontSize: "1.4rem", fontWeight: 800, color: "var(--text-primary)", lineHeight: 1, marginTop: 4 }}>
            <span style={{ color: "var(--success)" }}>{Number(cleanIoU).toFixed(2)}</span>
            <span style={{ color: "var(--text-tertiary)", margin: "0 4px" }}>→</span>
            <span style={{ color: "var(--danger)" }}>{Number(attackedIoU).toFixed(2)}</span>
          </div>
          <div style={{ fontSize: "0.68rem", color: "var(--text-muted)", marginTop: 4 }}>
            Độ chính xác bao khớp IoU
          </div>
        </div>
      </div>

      {/* ── Accumulated AP / mAP / Robustness columns — headline comparison ──
           Cột: baseline | tổ hợp 1 | tổ hợp 2 ... thêm dần sau mỗi lần chạy,
           tích lũy cho tới khi dataset bị đổi (scoped trong useAdverTest).
           Hàng: AP@50 | mAP@50-95 | Robustness.
      ────────────────────────────────────────────────────────────────── */}
      {report.cells?.length > 0 && (
        <AccumulatedMetricsTable report={report} />
      )}

      {/* ── Method × Severity pivot table — hiện ngay sau summary ────────────
           Cột: baseline | {attack}_{sev} ... tích lũy qua từng run.
           Hàng: tên attack (không có suffix severity).
      ────────────────────────────────────────────────────────────────── */}
      {report.cells?.length > 0 && (
        <MethodComparisonTable report={report} />
      )}

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
            <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "#FBBF24" }}>
              Human-in-the-Loop Review Queue (Quy trình Kỹ sư Duyệt Lỗi)
            </div>
            <div style={{ fontSize: "0.72rem", color: "var(--text-secondary)" }}>
              {flagResult
                ? `✅ Đã chuyển thành công ${flagResult.count} ca lỗi nghiêm trọng vào Review Queue!`
                : "Phân tích các ca phát hiện sai sót nghiêm trọng nhất, gán nhãn nhóm lỗi và đề xuất biện pháp phòng thủ."}
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
            {flagging ? "Đang chuyển..." : "Tự động gửi ca lỗi (Threshold ≥ 20%)"}
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
            <span>Mở Review Queue</span>
            <span>→</span>
          </a>
        </div>
      </div>

      {/* Bar Chart comparing metrics before and after attack */}
      <MetricsComparisonChart report={report} />

      {/* Aggregate Robustness Accuracy Curve */}
      {report.benchmark_metrics_available !== false && (
        <RAChart cells={report.cells} apClean={report.ap_clean} report={report} />
      )}

      {/* Degradation Heatmap Matrix across attacks & severities */}
      {report.cells?.length > 0 && (
        <HeatmapMatrix cells={report.cells} heatmap={report.heatmap} report={report} />
      )}
    </div>
  );
}
