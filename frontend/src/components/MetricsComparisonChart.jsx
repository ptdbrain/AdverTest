"use client";

import React, { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { getCanonicalAttackKey, getDescriptiveAttackName } from "@/lib/attackNaming";
import { useLanguage } from "@/context/LanguageContext";

const ATTACK_PALETTE = [
  "#EF4444", // Red
  "#F59E0B", // Amber
  "#8B5CF6", // Purple
  "#EC4899", // Pink
  "#3B82F6", // Blue
  "#14B8A6", // Teal
  "#F97316", // Orange
  "#6366F1", // Indigo
  "#06B6D4", // Cyan
  "#84CC16", // Lime
];

function CustomComparisonTooltip({ active, payload, label, baselineValues }) {
  if (!active || !payload?.length) return null;

  return (
    <div
      style={{
        background: "var(--bg-elevated, #1A2230)",
        border: "1px solid var(--border-medium, rgba(148, 163, 184, 0.16))",
        borderRadius: "var(--radius-md, 8px)",
        padding: "12px 16px",
        backdropFilter: "blur(12px)",
        boxShadow: "var(--shadow-lg, 0 8px 24px rgba(0,0,0,0.3))",
        minWidth: "260px",
        color: "var(--text-primary, #E2E8F0)",
      }}
    >
      <div
        style={{
          fontSize: "0.72rem",
          fontWeight: 700,
          color: "var(--text-secondary, #94A3B8)",
          marginBottom: "8px",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          borderBottom: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.08))",
          paddingBottom: "4px",
        }}
      >
        {label}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "0.8rem" }}>
        {payload.map((p, idx) => {
          const val = Number(p.value);
          const isBaseline = p.dataKey === "Baseline";
          const baselineVal = baselineValues?.[label] ?? 1;
          const dropPct = !isBaseline && baselineVal > 0 ? ((baselineVal - val) / baselineVal) * 100 : 0;

          return (
            <div key={idx} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "12px" }}>
              <span style={{ display: "flex", alignItems: "center", gap: "6px", color: isBaseline ? "var(--success, #16A34A)" : "var(--text-secondary)" }}>
                <span style={{ width: 8, height: 8, borderRadius: "2px", background: p.color }} />
                {p.name}:
              </span>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", fontFamily: "var(--font-mono)" }}>
                <strong style={{ color: p.color }}>{val.toFixed(3)}</strong>
                {!isBaseline && dropPct > 0 && (
                  <span style={{ fontSize: "0.72rem", color: "var(--danger, #EF4444)", fontWeight: 700 }}>
                    (↓{dropPct.toFixed(1)}%)
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default function MetricsComparisonChart({ report }) {
  const { t } = useLanguage();
  const apClean = report?.ap_clean ?? 0;
  const cells = report?.cells ?? [];
  const cleanMetrics = report?.metrics?.clean ?? {};

  const cleanAp50 = Number((cleanMetrics.ap50 ?? apClean).toFixed(3));
  const cleanMap = Number((cleanMetrics.map50_95 ?? Math.max(0, apClean * 0.72)).toFixed(3));

  // Extract each unique attack cell with guaranteed unique keys and clean names
  const attackSeries = useMemo(() => {
    if (cells.length === 0) {
      return [{ key: "attack_default", label: "Attack (Avg)", color: ATTACK_PALETTE[0], ap50: 0, map50_95: 0 }];
    }

    // Traverse in reverse (or use Map) to keep the latest cell for each canonical attack+severity
    const cellMap = new Map();
    cells.forEach((cell) => {
      const key = getCanonicalAttackKey(cell, cell.severity, report);
      cellMap.set(key, cell);
    });

    const uniqueCells = Array.from(cellMap.values());

    return uniqueCells.map((cell, idx) => {
      const label = getDescriptiveAttackName(cell, cell.severity, report);
      const canonicalKey = getCanonicalAttackKey(cell, cell.severity, report);
      const uniqueKey = `series_${canonicalKey}`;

      return {
        key: uniqueKey,
        label,
        attack: cell.attack,
        severity: cell.severity,
        ap50: Number((cell.metrics?.ap50 ?? cell.ap ?? 0).toFixed(3)),
        map50_95: Number((cell.metrics?.map50_95 ?? (cell.ap ? cell.ap * 0.72 : 0)).toFixed(3)),
        color: ATTACK_PALETTE[idx % ATTACK_PALETTE.length],
      };
    });
  }, [cells, report]);

  // Build overview chart data: each metric category has Baseline + each attack series
  const { overviewData, seriesKeys } = useMemo(() => {
    const ap50Row = { category: "AP@50 (IoU 0.50)", Baseline: cleanAp50 };
    const mapRow = { category: "mAP@50-95 (IoU 0.50:0.95)", Baseline: cleanMap };

    attackSeries.forEach((series) => {
      ap50Row[series.key] = series.ap50;
      mapRow[series.key] = series.map50_95;
    });

    const series = [
      { key: "Baseline", label: t("metrics.baselineClean"), color: "var(--success, #16A34A)" },
      ...attackSeries.map((s) => ({ key: s.key, label: s.label, color: s.color })),
    ];

    return {
      overviewData: [ap50Row, mapRow],
      seriesKeys: series,
    };
  }, [cleanAp50, cleanMap, attackSeries]);

  const baselineLookup = useMemo(() => {
    return {
      "AP@50 (IoU 0.50)": cleanAp50,
      "mAP@50-95 (IoU 0.50:0.95)": cleanMap,
    };
  }, [cleanAp50, cleanMap]);

  return (
    <div className="chart-container" style={{ padding: "var(--space-lg)", display: "flex", flexDirection: "column", gap: "var(--space-md)" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: "8px" }}>
        <div>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>
            {t("metrics.title")}
          </div>
          <h3 style={{ fontSize: "1.05rem", fontWeight: 700, margin: 0, color: "var(--text-primary)" }}>
            {t("metrics.subtitle", { count: attackSeries.length })}
          </h3>
        </div>
        <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          {t("metrics.hint")}
        </span>
      </div>

      {/* Summary Stat Cards across Baseline and Attack Combinations */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
          gap: "var(--space-sm)",
          overflowX: "auto",
          paddingBottom: "4px",
        }}
      >
        {/* Baseline Card */}
        <div
          style={{
            background: "var(--bg-elevated)",
            padding: "12px 14px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
            borderLeft: "3px solid var(--success, #16A34A)",
          }}
        >
          <div style={{ fontSize: "0.68rem", color: "var(--success)", textTransform: "uppercase", fontWeight: 700 }}>
            {t("metrics.baselineClean")}
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginTop: "4px" }}>
            <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--success)", fontFamily: "var(--font-mono)" }}>
              {cleanAp50.toFixed(3)}
            </span>
            <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
              mAP: {cleanMap.toFixed(3)}
            </span>
          </div>
          <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "4px" }}>
            {t("metrics.baseline")}
          </div>
        </div>

        {/* Individual Attack Combination Cards */}
        {attackSeries.map((atk, idx) => {
          const dropAp50 = cleanAp50 > 0 ? ((cleanAp50 - atk.ap50) / cleanAp50) * 100 : 0;
          return (
            <div
              key={idx}
              style={{
                background: "var(--bg-elevated)",
                padding: "12px 14px",
                borderRadius: "var(--radius-sm)",
                border: "1px solid var(--border-subtle)",
                borderLeft: `3px solid ${atk.color}`,
              }}
            >
              <div style={{ fontSize: "0.68rem", color: atk.color, textTransform: "uppercase", fontWeight: 700, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                {atk.label}
              </div>
              <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginTop: "4px" }}>
                <span style={{ fontSize: "1.2rem", fontWeight: 800, color: atk.color, fontFamily: "var(--font-mono)" }}>
                  {atk.ap50.toFixed(3)}
                </span>
                <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--danger, #EF4444)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>
                  ↓ {dropAp50.toFixed(1)}%
                </span>
              </div>
              <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "4px" }}>
                mAP@50-95: {atk.map50_95.toFixed(3)}
              </div>
            </div>
          );
        })}
      </div>

      {/* Dynamic Multi-Column Bar Chart */}
      <div style={{ width: "100%", height: 320, marginTop: "var(--space-xs)" }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={overviewData}
            margin={{ top: 20, right: 30, left: 0, bottom: 10 }}
            barCategoryGap="25%"
            barGap={6}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} opacity={0.6} />
            <XAxis
              dataKey="category"
              stroke="var(--text-muted)"
              fontSize={12}
              fontWeight={600}
              tickLine={false}
              axisLine={{ stroke: "var(--border-subtle)" }}
            />
            <YAxis
              stroke="var(--text-muted)"
              fontSize={11}
              domain={[0, 1]}
              tickFormatter={(v) => Number(v).toFixed(2)}
              tickLine={false}
              axisLine={{ stroke: "var(--border-subtle)" }}
            />
            <Tooltip
              content={<CustomComparisonTooltip baselineValues={baselineLookup} />}
              cursor={{ fill: "var(--bg-hover)", opacity: 0.4 }}
            />
            <Legend
              verticalAlign="top"
              align="right"
              wrapperStyle={{ paddingBottom: 12, fontSize: "0.75rem" }}
              formatter={(value) => (
                <span style={{ color: "var(--text-secondary)", fontWeight: 600 }}>{value}</span>
              )}
            />
            {seriesKeys.map((s) => (
              <Bar
                key={s.key}
                dataKey={s.key}
                name={s.label}
                fill={s.color}
                radius={[4, 4, 0, 0]}
                maxBarSize={48}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
