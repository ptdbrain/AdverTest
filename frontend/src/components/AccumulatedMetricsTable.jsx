/**
 * AccumulatedMetricsTable
 *
 * The headline report-tab comparison the user asked for: after every new attack
 * combination run, a new *column* is appended so its AP / mAP / robustness can be
 * compared side-by-side against the clean baseline and every previously run
 * combination. Columns accumulate until the dataset changes (scoping is handled
 * upstream in useAdverTest, which keys its report cache by normalized dataset).
 *
 * Layout (transposed deliberately so "one combination = one column"):
 *
 *   | Chỉ số        | Baseline (Ảnh sạch) | combo_1 | combo_2 | ... |
 *   | AP@50         |       0.920        |  0.810  |  0.550  | ... |
 *   | mAP@50-95     |       0.662        |  0.598  |  0.401  | ... |
 *   | Robustness    |        100%        |  88.0%  |  59.8%  | ... |
 *
 * Per-combination "robustness" is the retained-AP fraction
 * ``AP_combo / AP_clean`` (equivalently ``1 - degradation``), matching the
 * backend's plan-§3 "resilience rate" notion at the single-cell granularity.
 *
 * The component is a pure pivot over the already-merged ``report.cells`` array:
 * it reads the accumulated cells (produced by useAdverTest's dataset-keyed merge)
 * and never mutates backend state.
 *
 * @param {{ report: object|null }} props
 */
"use client";

import { useMemo } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { getCanonicalAttackKey, getDescriptiveAttackName } from "@/lib/attackNaming";

// ─── colour helpers ────────────────────────────────────────────────────────

/**
 * Colour tokens for a cell given its *retained* performance (0..100), where 100
 * means "no loss vs its own baseline" and 0 means "collapsed". This is the
 * inverse of the degradation scale used in MethodComparisonTable.
 */
function retentionTokens(retainedPct) {
  if (retainedPct == null) return { bg: "var(--bg-elevated)", text: "var(--text-muted)" };
  const r = Math.max(0, Math.min(100, retainedPct));
  if (r >= 95) return { bg: "rgba(104,211,145,0.18)", text: "#68d391" };
  if (r >= 85) return { bg: "rgba(104,211,145,0.08)", text: "#9ae6b4" };
  if (r >= 70) return { bg: "rgba(246,173,85,0.14)", text: "#fbd38d" };
  if (r >= 50) return { bg: "rgba(246,173,85,0.22)", text: "#f6ad55" };
  if (r >= 25) return { bg: "rgba(252,129,129,0.16)", text: "#feb2b2" };
  return { bg: "rgba(252,129,129,0.28)", text: "#fc8181" };
}

// ─── data builder ─────────────────────────────────────────────────────────

/**
 * Builds the transposed metric × combination matrix from an accumulated report.
 *
 * @param {object|null} report - Accumulated RunReport (with merged cells).
 * @returns {{
 *   apClean: number,
 *   metricRows: Array<{ key: string; label: string; baseline: number|null,
 *                      baselinePct: number|null; format: (v:number|null)=>string }>,
 *   columns: Array<{ key: string; label: string; values: Record<string, number|null>; retained: number|null }>
 * }}
 */
function buildMatrix(report) {
  const apClean = typeof report?.ap_clean === "number" ? report.ap_clean : null;
  const cleanMetrics = report?.metrics?.clean ?? {};
  const is3D = cleanMetrics.kitti_3d_ap != null;
  const cleanAp50 = cleanMetrics.kitti_3d_ap ?? cleanMetrics.ap50 ?? apClean;
  const cleanMap = cleanMetrics.map50_95 ?? null;
  const rawCells = report?.cells ?? [];

  // Deduplicate by canonical key (latest wins, matching useAdverTest), while
  // preserving first-seen/run order so columns read left-to-right like history.
  const columnMap = new Map();
  rawCells.forEach((cell) => {
    const key = getCanonicalAttackKey(cell, cell.severity, report);
    const label = getDescriptiveAttackName(cell, cell.severity, report);
    const ap50 = cell.metrics?.kitti_3d_ap ?? cell.metrics?.ap50 ?? cell.ap ?? null;
    const map = cell.metrics?.map50_95 ?? null;
    const retained = apClean > 0 && ap50 != null ? Math.max(0, Math.min(100, (ap50 / apClean) * 100)) : null;
    columnMap.set(key, {
      key,
      label,
      values: { ap50, map, robustness: retained },
      retained,
    });
  });

  const columns = Array.from(columnMap.values());

  const metricRows = [
    { key: "ap50", label: is3D ? "BEV AP" : "AP@50", baselinePct: null },
    { key: "map", label: is3D ? "BEV mAP" : "mAP@50-95", baselinePct: null },
    { key: "robustness", label: "Robustness (giữ lại)", baselinePct: 100 },
  ];

  return { apClean, cleanAp50, cleanMap, columns, metricRows };
}

function formatMetric(key, value) {
  if (value == null) return "—";
  if (key === "robustness") return `${value.toFixed(1)}%`;
  return value.toFixed(3);
}

// ─── sub-components ────────────────────────────────────────────────────────

function MetricCell({ metricKey, value, retained, isBaseline }) {
  if (isBaseline) {
    return (
      <td
        style={{
          background: "rgba(22,163,74,0.10)",
          color: "#4ade80",
          fontFamily: "var(--font-mono)",
          fontWeight: 700,
          fontSize: "0.82rem",
          textAlign: "center",
          padding: "8px 12px",
          whiteSpace: "nowrap",
        }}
      >
        {formatMetric(metricKey, value)}
      </td>
    );
  }

  const tokens = retentionTokens(retained);
  return (
    <td
      style={{
        background: tokens.bg,
        color: tokens.text,
        fontFamily: "var(--font-mono)",
        fontWeight: 600,
        fontSize: "0.82rem",
        textAlign: "center",
        padding: "8px 12px",
        whiteSpace: "nowrap",
      }}
      title={value == null ? "Không khả dụng" : formatMetric(metricKey, value)}
    >
      {formatMetric(metricKey, value)}
    </td>
  );
}

// ─── main component ────────────────────────────────────────────────────────

export default function AccumulatedMetricsTable({ report }) {
  const { t } = useLanguage();
  const { apClean, cleanAp50, cleanMap, columns, metricRows = [] } = useMemo(() => buildMatrix(report), [report]);

  if (!report || columns.length === 0) {
    return (
      <div className="heatmap">
        <div className="heatmap__title">{t("accum.title")}</div>
        <div className="empty-state">
          <div className="empty-state__message">
            {t("accum.empty")}
          </div>
        </div>
      </div>
    );
  }

  const baselineValues = {
    ap50: cleanAp50,
    map: cleanMap,
    robustness: apClean > 0 ? 100 : null,
  };

  return (
    <div className="heatmap" style={{ overflowX: "auto" }}>
      {/* ── Header ── */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          flexWrap: "wrap",
          gap: "8px",
          marginBottom: "var(--space-sm)",
        }}
      >
        <div>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>
            {t("accum.title")}
          </div>
          <div style={{ fontSize: "0.85rem", fontWeight: 700, color: "var(--text-primary)" }} dangerouslySetInnerHTML={{ __html: t("accum.heading", { count: columns.length, dataset: report.dataset ?? "—" }) }} />
        </div>
        <span
          style={{
            fontSize: "0.68rem",
            color: "var(--text-muted)",
            fontFamily: "var(--font-mono)",
          }}
        >
          {t("accum.robustnessHint")}
        </span>
      </div>

      {/* ── Transposed pivot table ── */}
      <div style={{ overflowX: "auto" }}>
        <table
          className="heatmap__table"
          style={{ minWidth: "100%", borderCollapse: "separate", borderSpacing: 0 }}
        >
          <thead>
            <tr>
              <th
                style={{
                  textAlign: "left",
                  padding: "10px 14px",
                  fontSize: "0.72rem",
                  fontWeight: 700,
                  color: "var(--text-secondary)",
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  whiteSpace: "nowrap",
                  position: "sticky",
                  left: 0,
                  background: "var(--bg-elevated)",
                  zIndex: 2,
                  borderRight: "1px solid var(--border-subtle)",
                }}
              >
                Chỉ số
              </th>

              {/* Baseline column header */}
              <th
                style={{
                  padding: "8px 12px",
                  fontSize: "0.68rem",
                  fontWeight: 700,
                  color: "#4ade80",
                  textTransform: "uppercase",
                  letterSpacing: "0.04em",
                  textAlign: "center",
                  whiteSpace: "nowrap",
                  borderBottom: "2px solid rgba(22,163,74,0.5)",
                  minWidth: "130px",
                }}
                title="Baseline (Ảnh sạch)"
              >
                baseline
                <span
                  style={{
                    display: "block",
                    fontWeight: 400,
                    fontSize: "0.6rem",
                    color: "var(--text-muted)",
                  }}
                >
                  Ảnh sạch
                </span>
              </th>

              {/* One column per accumulated attack combination */}
              {columns.map((col) => (
                <th
                  key={col.key}
                  style={{
                    padding: "8px 12px",
                    fontSize: "0.68rem",
                    fontWeight: 700,
                    color: "var(--text-secondary)",
                    textTransform: "uppercase",
                    letterSpacing: "0.04em",
                    textAlign: "center",
                    whiteSpace: "nowrap",
                    borderBottom: "2px solid var(--border-subtle)",
                    minWidth: "130px",
                  }}
                  title={col.label}
                >
                  <span style={{ display: "block" }}>{col.key}</span>
                  <span
                    style={{
                      display: "block",
                      fontWeight: 400,
                      fontSize: "0.6rem",
                      color: "var(--text-muted)",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      maxWidth: "150px",
                      margin: "0 auto",
                    }}
                    title={col.label}
                  >
                    {col.label}
                  </span>
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {metricRows.map((row) => (
              <tr key={row.key}>
                <td
                  style={{
                    padding: "8px 14px",
                    fontWeight: 600,
                    fontSize: "0.8rem",
                    color: "var(--text-primary)",
                    whiteSpace: "nowrap",
                    position: "sticky",
                    left: 0,
                    background: "var(--bg-surface, var(--bg-base))",
                    zIndex: 1,
                    borderRight: "1px solid var(--border-subtle)",
                  }}
                >
                  {row.label}
                </td>

                <MetricCell
                  metricKey={row.key}
                  value={baselineValues[row.key]}
                  retained={row.baselinePct}
                  isBaseline
                />

                {columns.map((col) => (
                  <MetricCell
                    key={col.key}
                    metricKey={row.key}
                    value={col.values[row.key]}
                    retained={col.retained}
                  />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── Legend ── */}
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "12px",
          marginTop: "var(--space-sm)",
          padding: "8px 0",
          borderTop: "1px solid var(--border-subtle)",
          fontSize: "0.66rem",
          color: "var(--text-muted)",
        }}
      >
        {[
          { bg: "rgba(104,211,145,0.18)", label: "≥ 95% giữ lại" },
          { bg: "rgba(104,211,145,0.08)", label: "85–95%" },
          { bg: "rgba(246,173,85,0.14)", label: "70–85%" },
          { bg: "rgba(246,173,85,0.22)", label: "50–70%" },
          { bg: "rgba(252,129,129,0.16)", label: "25–50%" },
          { bg: "rgba(252,129,129,0.28)", label: "< 25%" },
        ].map(({ bg, label }) => (
          <span key={label} style={{ display: "flex", alignItems: "center", gap: "4px" }}>
            <span
              style={{
                display: "inline-block",
                width: 10,
                height: 10,
                borderRadius: 2,
                background: bg,
                border: "1px solid rgba(255,255,255,0.08)",
              }}
            />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
