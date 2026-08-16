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

function CustomComparisonTooltip({ active, payload, label }) {
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
        minWidth: "240px",
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
        {payload.map((p, idx) => (
          <div key={idx} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--text-secondary)" }}>
              <span style={{ width: 8, height: 8, borderRadius: "2px", background: p.color }} />
              {p.name}:
            </span>
            <strong style={{ fontFamily: "var(--font-mono)", color: p.color }}>
              {Number(p.value).toFixed(3)}
            </strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function MetricsComparisonChart({ report }) {
  const apClean = report?.ap_clean ?? 0;
  const cells = report?.cells ?? [];
  const cleanMetrics = report?.metrics?.clean ?? {};

  const cleanAp50 = Number((cleanMetrics.ap50 ?? apClean).toFixed(3));
  const cleanMap = Number((cleanMetrics.map50_95 ?? Math.max(0, apClean * 0.72)).toFixed(3));

  // Average attacked AP50 and mAP50-95
  const avgAttackedAp50 = cells.length > 0
    ? Number((cells.reduce((acc, c) => acc + (c.metrics?.ap50 ?? c.ap ?? 0), 0) / cells.length).toFixed(3))
    : 0;

  const avgAttackedMap = cells.length > 0
    ? Number((cells.reduce((acc, c) => acc + (c.metrics?.map50_95 ?? (c.ap ? c.ap * 0.72 : 0)), 0) / cells.length).toFixed(3))
    : 0;

  // Overview data: 2 metrics (AP50 & mAP50-95), each with Clean and Attacked
  const overviewData = useMemo(() => {
    return [
      {
        category: "AP@50 (IoU 0.50)",
        "Clean (Trước tấn công)": cleanAp50,
        "Attacked (Sau tấn công)": avgAttackedAp50,
        clean: cleanAp50,
        attacked: avgAttackedAp50,
      },
      {
        category: "mAP@50-95 (IoU 0.50:0.95)",
        "Clean (Trước tấn công)": cleanMap,
        "Attacked (Sau tấn công)": avgAttackedMap,
        clean: cleanMap,
        attacked: avgAttackedMap,
      },
    ];
  }, [cleanAp50, avgAttackedAp50, cleanMap, avgAttackedMap]);

  const ap50Degradation = cleanAp50 > 0 ? ((cleanAp50 - avgAttackedAp50) / cleanAp50) * 100 : 0;
  const mapDegradation = cleanMap > 0 ? ((cleanMap - avgAttackedMap) / cleanMap) * 100 : 0;

  return (
    <div className="chart-container" style={{ padding: "var(--space-lg)", display: "flex", flexDirection: "column", gap: "var(--space-md)" }}>
      {/* Header */}
      <div>
        <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>
          Biểu đồ so sánh cốt lõi
        </div>
        <h3 style={{ fontSize: "1.05rem", fontWeight: 700, margin: 0, color: "var(--text-primary)" }}>
          So sánh AP50 & mAP50-95 Trước và Sau Tấn công
        </h3>
      </div>

      {/* Summary Stat Cards */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: "var(--space-sm)",
        }}
      >
        {/* AP50 Box */}
        <div
          style={{
            background: "var(--bg-elevated)",
            padding: "12px 16px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
            borderLeft: "3px solid var(--accent)",
          }}
        >
          <div style={{ fontSize: "0.68rem", color: "var(--text-tertiary)", textTransform: "uppercase", fontWeight: 700 }}>
            AP@50 (IoU 0.50)
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginTop: "4px" }}>
            <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--success)", fontFamily: "var(--font-mono)" }}>
              {cleanAp50.toFixed(3)}
            </span>
            <span style={{ fontSize: "0.85rem", color: "var(--text-tertiary)" }}>→</span>
            <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--danger)", fontFamily: "var(--font-mono)" }}>
              {avgAttackedAp50.toFixed(3)}
            </span>
            <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--danger)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>
              ↓ {ap50Degradation.toFixed(1)}%
            </span>
          </div>
          <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "4px" }}>
            Clean: {cleanAp50.toFixed(3)} | Attacked: {avgAttackedAp50.toFixed(3)}
          </div>
        </div>

        {/* mAP50-95 Box */}
        <div
          style={{
            background: "var(--bg-elevated)",
            padding: "12px 16px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
            borderLeft: "3px solid #8B5CF6",
          }}
        >
          <div style={{ fontSize: "0.68rem", color: "var(--text-tertiary)", textTransform: "uppercase", fontWeight: 700 }}>
            mAP@50-95 (IoU 0.50:0.95)
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginTop: "4px" }}>
            <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--success)", fontFamily: "var(--font-mono)" }}>
              {cleanMap.toFixed(3)}
            </span>
            <span style={{ fontSize: "0.85rem", color: "var(--text-tertiary)" }}>→</span>
            <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "var(--danger)", fontFamily: "var(--font-mono)" }}>
              {avgAttackedMap.toFixed(3)}
            </span>
            <span style={{ fontSize: "0.8rem", fontWeight: 700, color: "var(--danger)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>
              ↓ {mapDegradation.toFixed(1)}%
            </span>
          </div>
          <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "4px" }}>
            Clean: {cleanMap.toFixed(3)} | Attacked: {avgAttackedMap.toFixed(3)}
          </div>
        </div>
      </div>

      {/* Bar Chart Overview (2 Pairs of Bars) */}
      <div style={{ width: "100%", height: 280, marginTop: "var(--space-xs)" }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={overviewData}
            margin={{ top: 20, right: 30, left: 0, bottom: 10 }}
            barCategoryGap="35%"
            barGap={10}
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
            <Tooltip content={<CustomComparisonTooltip />} cursor={{ fill: "var(--bg-hover)", opacity: 0.4 }} />
            <Legend
              verticalAlign="top"
              align="right"
              wrapperStyle={{ paddingBottom: 12, fontSize: "0.75rem" }}
              formatter={(value) => (
                <span style={{ color: "var(--text-secondary)", fontWeight: 600 }}>{value}</span>
              )}
            />
            <Bar
              dataKey="Clean (Trước tấn công)"
              fill="var(--success, #16A34A)"
              radius={[6, 6, 0, 0]}
              maxBarSize={64}
            />
            <Bar
              dataKey="Attacked (Sau tấn công)"
              fill="var(--danger, #EF6461)"
              radius={[6, 6, 0, 0]}
              maxBarSize={64}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
