/**
 * MethodComparisonTable
 *
 * Renders the accumulated benchmark report as a pivot table where:
 *   - Rows    = attack names (e.g. "gaussian_noise", "fgsm+brightness")
 *   - Columns = "baseline" (ap_clean) + one column per method–severity pair
 *               accumulated across successive runs, e.g. "gaussian_noise_3",
 *               "fgsm+blur_5", etc.
 *
 * The accumulation logic in useAdverTest already merges cells from different
 * runs into a single `report.cells` array. This component builds the pivot
 * view on top of that merged array without touching backend state.
 *
 * @param {{ report: object|null }} props
 */
"use client";

import { useMemo } from "react";
import { clsx } from "clsx";
import { getCanonicalAttackKey, getDescriptiveAttackName, SHORT_ATTACK_LABELS, ATTACK_LABELS } from "@/lib/attackNaming";

// ─── colour helpers ────────────────────────────────────────────────────────

/** Returns Tailwind-compatible inline colour tokens based on degradation %. */
function degradationTokens(degradationPct) {
  if (degradationPct == null) return { bg: "var(--bg-elevated)", text: "var(--text-muted)", badge: "—" };
  const d = Math.max(0, Math.min(100, degradationPct));
  if (d < 5)  return { bg: "rgba(104,211,145,0.12)", text: "#68d391" };
  if (d < 15) return { bg: "rgba(104,211,145,0.08)", text: "#9ae6b4" };
  if (d < 30) return { bg: "rgba(246,173,85,0.14)",  text: "#fbd38d" };
  if (d < 50) return { bg: "rgba(246,173,85,0.22)",  text: "#f6ad55" };
  if (d < 75) return { bg: "rgba(252,129,129,0.16)", text: "#feb2b2" };
  return        { bg: "rgba(252,129,129,0.28)", text: "#fc8181" };
}

// ─── column label helpers ──────────────────────────────────────────────────

/**
 * Given a cell, derives the short column header string used in the table.
 * Format: `{attackShortName}_{severity}`, e.g. "gaussian_noise_3".
 *
 * @param {object} cell - RunReport cell object.
 * @param {object|null} report - Full accumulated report for recipe-step lookup.
 * @returns {string}
 */
function columnKeyFromCell(cell, report) {
  return getCanonicalAttackKey(cell, cell.severity, report);
}

/**
 * Human-readable column header, e.g. "Nhiễu Gaussian (Cấp 3)".
 *
 * @param {object} cell
 * @param {object|null} report
 * @returns {string}
 */
function columnLabelFromCell(cell, report) {
  return getDescriptiveAttackName(cell, cell.severity, report);
}

/**
 * Short row-label for the attack axis.
 * For recipe (multi-step) attacks the steps are joined with " + ".
 *
 * @param {object} cell
 * @param {object|null} report
 * @returns {string}
 */
function rowLabelFromCell(cell, report) {
  // Strip the "(Cấp X)" suffix to keep row labels severity-agnostic
  return getDescriptiveAttackName(cell, null, report).replace(/\s*\(Cấp\s*\d+\)/i, "").trim();
}

// ─── data builder ─────────────────────────────────────────────────────────

/**
 * Builds the pivot structure from an accumulated RunReport.
 *
 * @param {object|null} report - Accumulated RunReport (with merged cells).
 * @returns {{
 *   columnDefs: Array<{ key: string; label: string; isBaseline: boolean }>,
 *   rows: Array<{ attackLabel: string; baseline: number; cells: Map<string, { ap: number; degradationPct: number }> }>
 * }}
 */
function buildPivotTable(report) {
  const apClean = report?.ap_clean ?? 0;
  const rawCells = report?.cells ?? [];

  if (rawCells.length === 0) {
    return { columnDefs: [], rows: [] };
  }

  // ── 1. Deduplicate cells by canonical key (latest wins, matching useAdverTest) ──
  const cellMap = new Map();
  rawCells.forEach((cell) => {
    const key = columnKeyFromCell(cell, report);
    cellMap.set(key, cell);
  });

  // ── 2. Build ordered column list (preserving insertion/run order) ──
  /** @type {Map<string, { key: string; label: string; cell: object }>} */
  const columnMap = new Map();
  rawCells.forEach((cell) => {
    const key = columnKeyFromCell(cell, report);
    if (!columnMap.has(key)) {
      columnMap.set(key, { key, label: columnLabelFromCell(cell, report), cell });
    }
  });

  const columnDefs = [
    { key: "baseline", label: "Baseline (Ảnh sạch)", isBaseline: true },
    ...Array.from(columnMap.values()).map(({ key, label }) => ({ key, label, isBaseline: false })),
  ];

  // ── 3. Group cells by attack row (attack name without severity) ──
  /** @type {Map<string, { rowLabel: string; byColKey: Map<string, object> }>} */
  const rowMap = new Map();
  Array.from(cellMap.values()).forEach((cell) => {
    const colKey = columnKeyFromCell(cell, report);
    const rowLabel = rowLabelFromCell(cell, report);

    if (!rowMap.has(rowLabel)) {
      rowMap.set(rowLabel, { rowLabel, byColKey: new Map() });
    }
    rowMap.get(rowLabel).byColKey.set(colKey, cell);
  });

  // ── 4. Compute degradation % for each cell relative to ap_clean ──
  const rows = Array.from(rowMap.values()).map(({ rowLabel, byColKey }) => ({
    attackLabel: rowLabel,
    baseline: apClean,
    byColKey,
  }));

  return { columnDefs, rows, apClean };
}

// ─── sub-components ────────────────────────────────────────────────────────

function BaselineCell({ ap }) {
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
      title={`AP clean: ${ap?.toFixed(4)}`}
    >
      {ap != null ? ap.toFixed(3) : "—"}
    </td>
  );
}

function AttackCell({ cell, apClean }) {
  if (!cell) {
    return (
      <td
        style={{
          background: "var(--bg-elevated)",
          color: "var(--text-muted)",
          textAlign: "center",
          padding: "8px 12px",
          fontSize: "0.8rem",
        }}
      >
        —
      </td>
    );
  }

  const ap = cell.ap ?? 0;
  const degradationPct = apClean > 0 ? Math.max(0, ((apClean - ap) / apClean) * 100) : 0;
  const tokens = degradationTokens(degradationPct);

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
      title={`AP: ${ap.toFixed(4)} | Degradation: ${degradationPct.toFixed(1)}%`}
    >
      <span style={{ display: "block" }}>{ap.toFixed(3)}</span>
      <span style={{ display: "block", fontSize: "0.65rem", opacity: 0.8, marginTop: "1px" }}>
        ↓{degradationPct.toFixed(1)}%
      </span>
    </td>
  );
}

// ─── main component ────────────────────────────────────────────────────────

export default function MethodComparisonTable({ report }) {
  const { columnDefs, rows, apClean } = useMemo(() => buildPivotTable(report), [report]);

  if (!report || columnDefs.length === 0) {
    return (
      <div className="heatmap">
        <div className="heatmap__title">Method × Severity Report</div>
        <div className="empty-state">
          <div className="empty-state__message">
            Chưa có dữ liệu — chạy ít nhất một method để hiển thị bảng.
          </div>
        </div>
      </div>
    );
  }

  const attackColumnCount = columnDefs.filter((c) => !c.isBaseline).length;

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
            Bảng so sánh tích lũy (Method × Severity)
          </div>
          <div
            style={{
              fontSize: "0.85rem",
              fontWeight: 700,
              color: "var(--text-primary)",
            }}
          >
            Baseline + {attackColumnCount} phương pháp tấn công tích lũy
          </div>
        </div>
        <span
          style={{
            fontSize: "0.68rem",
            color: "var(--text-muted)",
            fontFamily: "var(--font-mono)",
          }}
        >
          Cell = AP | ↓% = suy giảm so Baseline
        </span>
      </div>

      {/* ── Pivot Table ── */}
      <div style={{ overflowX: "auto" }}>
        <table
          className="heatmap__table"
          style={{ minWidth: "100%", borderCollapse: "separate", borderSpacing: 0 }}
        >
          <thead>
            <tr>
              {/* Row header: attack name */}
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
                Dạng tấn công
              </th>

              {/* Dynamic column headers */}
              {columnDefs.map((col) => (
                <th
                  key={col.key}
                  style={{
                    padding: "8px 12px",
                    fontSize: "0.68rem",
                    fontWeight: 700,
                    color: col.isBaseline ? "#4ade80" : "var(--text-secondary)",
                    textTransform: "uppercase",
                    letterSpacing: "0.04em",
                    textAlign: "center",
                    whiteSpace: "nowrap",
                    borderBottom: col.isBaseline
                      ? "2px solid rgba(22,163,74,0.5)"
                      : "2px solid var(--border-subtle)",
                    minWidth: "120px",
                  }}
                  title={col.label}
                >
                  {col.isBaseline ? (
                    "baseline"
                  ) : (
                    <>
                      <span style={{ display: "block" }}>{col.key}</span>
                      <span
                        style={{
                          display: "block",
                          fontWeight: 400,
                          fontSize: "0.6rem",
                          color: "var(--text-muted)",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          maxWidth: "140px",
                          margin: "0 auto",
                        }}
                        title={col.label}
                      >
                        {col.label}
                      </span>
                    </>
                  )}
                </th>
              ))}
            </tr>
          </thead>

          <tbody>
            {rows.map((row) => (
              <tr key={row.attackLabel}>
                {/* Attack name (sticky left column) */}
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
                  {row.attackLabel}
                </td>

                {/* Baseline column */}
                <BaselineCell ap={row.baseline} />

                {/* One cell per method–severity column */}
                {columnDefs
                  .filter((col) => !col.isBaseline)
                  .map((col) => (
                    <AttackCell
                      key={col.key}
                      cell={row.byColKey.get(col.key) ?? null}
                      apClean={apClean}
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
          { bg: "rgba(104,211,145,0.18)", label: "< 5% suy giảm" },
          { bg: "rgba(104,211,145,0.08)", label: "5–15%" },
          { bg: "rgba(246,173,85,0.14)",  label: "15–30%" },
          { bg: "rgba(246,173,85,0.22)",  label: "30–50%" },
          { bg: "rgba(252,129,129,0.16)", label: "50–75%" },
          { bg: "rgba(252,129,129,0.28)", label: "> 75%" },
        ].map(({ bg, label }) => (
          <span
            key={label}
            style={{ display: "flex", alignItems: "center", gap: "4px" }}
          >
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
