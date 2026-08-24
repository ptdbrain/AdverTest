"use client";

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  LabelList,
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

function CustomBarTooltip({ active, payload }) {
  const { t } = useLanguage();
  if (!active || !payload?.length) return null;
  const data = payload[0]?.payload;
  if (!data) return null;

  const isBaseline = data.isBaseline;
  const raVal = Number(data.ra ?? 0);
  const degradation = Math.max(0, 100 - raVal);

  return (
    <div
      style={{
        background: "var(--bg-elevated, #1A2230)",
        border: "1px solid var(--border-medium, rgba(148, 163, 184, 0.16))",
        borderRadius: "var(--radius-md, 8px)",
        padding: "10px 14px",
        backdropFilter: "blur(12px)",
        boxShadow: "var(--shadow-lg, 0 8px 24px rgba(0,0,0,0.3))",
        minWidth: "220px",
        color: "var(--text-primary, #E2E8F0)",
      }}
    >
      <div
        style={{
          fontSize: "0.72rem",
          fontWeight: 700,
          color: data.color || "var(--text-primary)",
          marginBottom: "6px",
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          borderBottom: "1px solid var(--border-subtle, rgba(148, 163, 184, 0.08))",
          paddingBottom: "4px",
        }}
      >
        {data.name}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "6px", fontSize: "0.78rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ color: "var(--text-secondary)" }}>{t("ra.raLabel")}</span>
          <strong style={{ color: isBaseline ? "var(--success, #16A34A)" : raVal > 70 ? "var(--success, #16A34A)" : raVal > 40 ? "var(--warning, #F59E0B)" : "var(--danger, #EF4444)", fontFamily: "var(--font-mono)" }}>
            {raVal.toFixed(1)}%
          </strong>
        </div>
        {!isBaseline && (
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ color: "var(--text-secondary)" }}>{t("ra.degradation")}</span>
            <strong style={{ color: "var(--danger, #EF4444)", fontFamily: "var(--font-mono)" }}>
              ↓ {degradation.toFixed(1)}%
            </strong>
          </div>
        )}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "2px" }}>
          <span>{t("ra.mapActual")}</span>
          <span style={{ fontFamily: "var(--font-mono)" }}>{Number(data.ap ?? 0).toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
}

export default function RAChart({ cells = [], apClean = 1, report = null }) {
  const { t } = useLanguage();
  const chartData = useMemo(() => {
    if (cells.length === 0 || apClean === 0) return [];

    const baselineItem = {
      name: "Baseline",
      label: "Baseline",
      ra: 100.0,
      degradation: 0.0,
      ap: apClean,
      color: "var(--success, #16A34A)",
      isBaseline: true,
    };

    // Deduplicate cells by canonical attack key to prevent double rendering
    const cellMap = new Map();
    cells.forEach((cell) => {
      const key = getCanonicalAttackKey(cell, cell.severity, report);
      cellMap.set(key, cell);
    });

    const uniqueCells = Array.from(cellMap.values());

    const attackItems = uniqueCells.map((cell, idx) => {
      const label = getDescriptiveAttackName(cell, cell.severity, report);
      const raRatio = cell.ap / apClean;
      const raPct = Math.min(100, Math.max(0, raRatio * 100));
      return {
        name: label,
        label,
        attack: cell.attack,
        severity: cell.severity,
        ra: Number(raPct.toFixed(1)),
        degradation: Number((100 - raPct).toFixed(1)),
        ap: cell.ap,
        color: ATTACK_PALETTE[idx % ATTACK_PALETTE.length],
        isBaseline: false,
      };
    });

    return [baselineItem, ...attackItems];
  }, [cells, apClean, report]);

  if (chartData.length === 0) {
    return (
      <div className="chart-container">
        <div className="chart-container__title">{t("ra.chartTitle")}</div>
        <div className="empty-state">
          <div className="empty-state__message">{t("ra.empty")}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="chart-container" style={{ padding: "var(--space-lg)", display: "flex", flexDirection: "column", gap: "var(--space-md)" }}>
      {/* Header with Title and Description */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: "8px" }}>
        <div>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>
            {t("ra.axisLabel")}
          </div>
          <h3 style={{ fontSize: "1.05rem", fontWeight: 700, margin: 0, color: "var(--text-primary)" }}>
            {t("ra.subtitle")}
          </h3>
        </div>
        <span style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
          RA % = (AP / AP Baseline) × 100%
        </span>
      </div>

      {/* Percentage Multi-Column Bar Chart */}
      <div style={{ width: "100%", height: 300, marginTop: "var(--space-xs)" }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            margin={{ top: 24, right: 24, bottom: 24, left: 0 }}
            barCategoryGap="20%"
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} opacity={0.6} />
            <XAxis
              dataKey="name"
              stroke="var(--text-muted)"
              fontSize={11}
              fontWeight={600}
              interval={0}
              angle={-15}
              textAnchor="end"
              tickLine={false}
              axisLine={{ stroke: "var(--border-subtle)" }}
              height={50}
            />
            <YAxis
              stroke="var(--text-muted)"
              fontSize={11}
              domain={[0, 100]}
              tickFormatter={(v) => `${v}%`}
              tickLine={false}
              axisLine={{ stroke: "var(--border-subtle)" }}
            />
            <Tooltip
              content={<CustomBarTooltip />}
              cursor={{ fill: "var(--bg-hover)", opacity: 0.4 }}
            />
            <ReferenceLine y={100} stroke="var(--success)" strokeDasharray="3 3" opacity={0.5} label={{ value: "100% Clean", position: "insideTopRight", fill: "var(--success)", fontSize: 10 }} />
            <ReferenceLine y={50} stroke="var(--warning)" strokeDasharray="3 3" opacity={0.5} label={{ value: "50% Boundary", position: "insideTopRight", fill: "var(--warning)", fontSize: 10 }} />
            <Bar dataKey="ra" radius={[4, 4, 0, 0]} maxBarSize={48}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
              <LabelList
                dataKey="ra"
                position="top"
                formatter={(v) => `${Number(v).toFixed(0)}%`}
                style={{ fill: "var(--text-secondary)", fontSize: 10, fontWeight: 700 }}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Mini Legend */}
      <div style={{ display: "flex", gap: "16px", justifyContent: "center", flexWrap: "wrap", fontSize: "0.72rem", color: "var(--text-secondary)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ width: 10, height: 10, borderRadius: "2px", background: "var(--success)" }} />
          <span>{t("ra.legendGood")}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ width: 10, height: 10, borderRadius: "2px", background: "var(--warning)" }} />
          <span>{t("ra.avgRange")}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span style={{ width: 10, height: 10, borderRadius: "2px", background: "var(--danger)" }} />
          <span>{t("ra.legendSevere")}</span>
        </div>
      </div>
    </div>
  );
}
