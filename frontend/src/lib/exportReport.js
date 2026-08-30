/**
 * P2.11: Production Report Exporters (JSON, CSV, PDF/Printable HTML).
 *
 * Guarantees:
 * - Direct ingestion from backend RunReport schema without fabricated numbers.
 * - JSON preserves 100% of scientific provenance.
 * - CSV uses flattened, versioned format (advertest-export-v1).
 * - PDF/Printable contains model/dataset/protocol, attacks, metrics, evidence state, limitations, and GPU gate.
 * - Missing metrics render as 'No data' / '—', never replaced by 0.
 */

import { getEvidenceState } from "@/components/common/EvidenceBadge";
import { buildRunDecisionView, primaryMetricForCell } from "@/lib/reportMetrics";

/**
 * Export RunReport as raw JSON with complete provenance.
 */
export function exportReportAsJson(report, filename = null) {
  if (!report) return;
  const jsonStr = JSON.stringify(report, null, 2);
  const blob = new Blob([jsonStr], { type: "application/json;charset=utf-8" });
  downloadBlob(blob, filename || `report_${report.run_id || "export"}.json`);
}

/**
 * Export RunReport cells and summary metrics as flattened CSV (version 1.0).
 */
export function exportReportAsCsv(report, filename = null) {
  if (!report) return;
  const decision = buildRunDecisionView(report);

  const rows = [];
  // CSV Header
  rows.push([
    "schema_version",
    "run_id",
    "model",
    "model_version",
    "dataset",
    "n_samples",
    "task_id",
    "primary_metric_key",
    "primary_metric_label",
    "clean_primary_metric",
    "attack_name",
    "severity",
    "attacked_primary_metric",
    "degradation_ratio",
    "degradation_percent",
    "evidence_state",
    "simulation_only",
    "seed",
    "benchmark_protocol_id",
    "dataset_version_id",
  ]);

  const evidenceState = getEvidenceState(report.provenance, report.simulation_only, undefined, report.evidence);
  const seed = report.provenance?.run_config?.seed ?? report.provenance?.seed ?? "—";
  const cells = report.cells || [];

  if (cells.length === 0) {
    // Single summary row if no cells
    rows.push([
      "advertest-export-v1",
      report.run_id || "—",
      report.model || "—",
      report.model_version || "—",
      report.dataset || "—",
      report.n_samples ?? "—",
      decision.taskId,
      decision.primary.key,
      decision.primary.label,
      decision.primary.clean ?? "No data",
      "—",
      "—",
      "—",
      "—",
      "—",
      evidenceState,
      report.simulation_only ?? true,
      seed,
      decision.protocol.benchmarkProtocolId ?? "—",
      decision.protocol.datasetVersionId ?? "—",
    ]);
  } else {
    cells.forEach((cell) => {
      const attackedPrimary = primaryMetricForCell(report, cell, decision.taskId, decision.primary.key);
      const degRatio = cell.degradation_ratio
        ?? (typeof cell.degradation_percent === "number" ? cell.degradation_percent / 100 : null)
        ?? (decision.primary.clean && attackedPrimary != null
          ? (decision.primary.clean - attackedPrimary) / decision.primary.clean
          : null);

      rows.push([
        "advertest-export-v1",
        report.run_id || "—",
        report.model || "—",
        report.model_version || "—",
        report.dataset || "—",
        report.n_samples ?? "—",
        decision.taskId,
        decision.primary.key,
        decision.primary.label,
        decision.primary.clean ?? "No data",
        cell.attack || cell.attack_name || "—",
        cell.severity ?? "—",
        attackedPrimary ?? "No data",
        degRatio != null ? degRatio.toFixed(6) : "No data",
        degRatio != null ? (degRatio * 100).toFixed(4) : "No data",
        evidenceState,
        report.simulation_only ?? true,
        seed,
        decision.protocol.benchmarkProtocolId ?? "—",
        decision.protocol.datasetVersionId ?? "—",
      ]);
    });
  }

  const csvContent = rows
    .map((row) =>
      row
        .map((val) => {
          const str = String(val ?? "");
          return str.includes(",") || str.includes('"') || str.includes("\n")
            ? `"${str.replace(/"/g, '""')}"`
            : str;
        })
        .join(",")
    )
    .join("\n");

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8" });
  downloadBlob(blob, filename || `report_${report.run_id || "export"}.csv`);
}

/**
 * Open a formatted printable PDF view in a new window.
 */
export function printReportAsPdf(report) {
  if (!report) return;
  const decision = buildRunDecisionView(report);

  const evidenceState = getEvidenceState(report.provenance, report.simulation_only, undefined, report.evidence);
  const printWindow = window.open("", "_blank");
  if (!printWindow) return;

  const cells = report.cells || [];
  const cellsHtml = cells.length > 0
    ? cells
        .map((c) => {
          const attackedPrimary = primaryMetricForCell(report, c, decision.taskId, decision.primary.key);
          const deg = c.degradation_ratio
            ?? (typeof c.degradation_percent === "number" ? c.degradation_percent / 100 : null)
            ?? (decision.primary.clean && attackedPrimary != null
              ? (decision.primary.clean - attackedPrimary) / decision.primary.clean
              : null);
          return `
          <tr>
            <td style="padding: 8px; border: 1px solid #ddd;">${c.attack || c.attack_name || "—"}</td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: center;">${c.severity ?? "—"}</td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">${attackedPrimary != null ? attackedPrimary.toFixed(4) : "No data"}</td>
            <td style="padding: 8px; border: 1px solid #ddd; text-align: right; color: ${deg && deg > 0.2 ? '#dc2626' : '#16a34a'};">
              ${deg != null ? (deg * 100).toFixed(2) + "%" : "No data"}
            </td>
          </tr>`;
        })
        .join("")
    : `<tr><td colspan="4" style="padding: 12px; text-align: center; color: #666;">Chưa có dữ liệu cells (No data)</td></tr>`;

  const html = `
    <!DOCTYPE html>
    <html>
      <head>
        <title>Báo cáo Đánh giá Độ bền vững AI — ${report.run_id || ""}</title>
        <meta charset="utf-8" />
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 40px; color: #1e293b; }
          h1 { color: #0f172a; margin-bottom: 4px; font-size: 24px; }
          .badge { display: inline-block; padding: 4px 10px; font-size: 12px; font-weight: 700; border-radius: 4px; background: #e0f2fe; color: #0369a1; }
          .meta-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; margin: 24px 0; background: #f8fafc; padding: 16px; border-radius: 8px; border: 1px solid #e2e8f0; }
          .meta-item { font-size: 13px; }
          .meta-item strong { color: #475569; }
          table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 13px; }
          th { background: #f1f5f9; padding: 10px; border: 1px solid #cbd5e1; text-align: left; font-weight: 600; }
          .footer { margin-top: 40px; font-size: 11px; color: #94a3b8; border-top: 1px solid #e2e8f0; padding-top: 12px; }
          @media print {
            body { padding: 0; }
            button { display: none; }
          }
        </style>
      </head>
      <body>
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
          <div>
            <h1>AdverTest — Báo cáo Thử nghiệm Độ bền vững AI</h1>
            <p style="margin: 0; color: #64748b; font-size: 13px;">Mã thử nghiệm: <strong>${report.run_id || "—"}</strong></p>
          </div>
          <div>
            <span class="badge">${evidenceState}</span>
          </div>
        </div>

        <div class="meta-grid">
          <div class="meta-item"><strong>Mô hình:</strong> ${report.model || "—"} (v${report.model_version || "1.0"})</div>
          <div class="meta-item"><strong>Tập dữ liệu:</strong> ${report.dataset || "—"} (${report.n_samples ?? "—"} mẫu)</div>
          <div class="meta-item"><strong>${decision.primary.label} Clean:</strong> ${decision.primary.clean != null ? decision.primary.clean.toFixed(4) : "No data"}</div>
          <div class="meta-item"><strong>Chế độ mô phỏng:</strong> ${report.simulation_only ? "Mô phỏng (Simulation)" : "Thực nghiệm (Real Dataset)"}</div>
          <div class="meta-item"><strong>Seed kiểm thử:</strong> ${decision.protocol.seed ?? "—"}</div>
          <div class="meta-item"><strong>Protocol ID:</strong> ${decision.protocol.benchmarkProtocolId || "—"}</div>
        </div>

        <h3>Chi tiết Ma trận Tấn công & Độ suy giảm (Degradation)</h3>
        <table>
          <thead>
            <tr>
              <th>Phương pháp tấn công</th>
              <th style="text-align: center;">Mức độ (Severity)</th>
              <th style="text-align: right;">${decision.primary.label} Sau tấn công</th>
              <th style="text-align: right;">Độ suy giảm (% Degradation)</th>
            </tr>
          </thead>
          <tbody>
            ${cellsHtml}
          </tbody>
        </table>

        <div class="footer">
          <p>Tạo tự động bởi Nền tảng AdverTest lúc ${new Date().toISOString()}. Provenance hash: ${report.provenance?.split_manifest_hash || "N/A"}</p>
        </div>

        <div style="margin-top: 24px; text-align: right;">
          <button onclick="window.print()" style="padding: 8px 16px; background: #2563eb; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-weight: 600;">In / Lưu PDF</button>
        </div>
      </body>
    </html>
  `;

  printWindow.document.open();
  printWindow.document.write(html);
  printWindow.document.close();
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
