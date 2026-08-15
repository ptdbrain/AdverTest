"use client";

import React, { useState, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
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
  const [viewMode, setViewMode] = useState("overview"); // "overview" | "per_attack"

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

  // 1. Overview data: 2 metrics (AP50 & mAP50-95), each with Clean and Attacked
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

  // 2. Per-attack data: Showing 4 bars per attack: AP50 Clean, AP50 Attacked, mAP50-95 Clean, mAP50-95 Attacked
  const perAttackData = useMemo(() => {
    if (!cells.length) return [];
    return cells.map((cell) => {
      const name = cell.attack ? cell.attack.replace(/_/g, " ").toUpperCase() : "Attack";
      const label = `${name} (Sev ${cell.severity})`;
      const cellAp50 = Number((cell.metrics?.ap50 ?? cell.ap ?? 0).toFixed(3));
      const cellMap = Number((cell.metrics?.map50_95 ?? (cell.ap ? cell.ap * 0.72 : 0)).toFixed(3));

      return {
        category: label,
        "AP50 (Clean)": cleanAp50,
        "AP50 (Attacked)": cellAp50,
        "mAP50-95 (Clean)": cleanMap,
        "mAP50-95 (Attacked)": cellMap,
        ap50Clean: cleanAp50,
        ap50Attacked: cellAp50,
        mapClean: cleanMap,
        mapAttacked: cellMap,
        dropAp50: cleanAp50 > 0 ? ((cleanAp50 - cellAp50) / cleanAp50) * 100 : 0,
        dropMap: cleanMap > 0 ? ((cleanMap - cellMap) / cleanMap) * 100 : 0,
      };
    });
  }, [cells, cleanAp50, cleanMap]);

  const ap50Degradation = cleanAp50 > 0 ? ((cleanAp50 - avgAttackedAp50) / cleanAp50) * 100 : 0;
  const mapDegradation = cleanMap > 0 ? ((cleanMap - avgAttackedMap) / cleanMap) * 100 : 0;

  return (
    <div className="chart-container" style={{ padding: "var(--space-lg)", display: "flex", flexDirection: "column", gap: "var(--space-md)" }}>
      {/* Header & Toggle */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "var(--space-sm)" }}>
        <div>
          <div className="config-panel__label" style={{ paddingBottom: 0, marginBottom: 2 }}>
            Biểu đồ so sánh cốt lõi
          </div>
          <h3 style={{ fontSize: "1.05rem", fontWeight: 700, margin: 0, color: "var(--text-primary)" }}>
            So sánh AP50 & mAP50-95 Trước và Sau Tấn công
          </h3>
        </div>

        {/* View Switcher */}
        <div
          style={{
            display: "inline-flex",
            background: "var(--bg-deep)",
            padding: "3px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
          }}
        >
          <button
            type="button"
            onClick={() => setViewMode("overview")}
            style={{
              padding: "4px 10px",
              fontSize: "0.68rem",
              fontWeight: 600,
              borderRadius: "var(--radius-xs)",
              border: "none",
              cursor: "pointer",
              transition: "all var(--transition-fast)",
              background: viewMode === "overview" ? "var(--bg-elevated)" : "transparent",
              color: viewMode === "overview" ? "var(--text-primary)" : "var(--text-tertiary)",
              boxShadow: viewMode === "overview" ? "var(--shadow-xs)" : "none",
            }}
          >
            Tổng quan (2 Cặp Cột)
          </button>
          <button
            type="button"
            onClick={() => setViewMode("per_attack")}
            style={{
              padding: "4px 10px",
              fontSize: "0.68rem",
              fontWeight: 600,
              borderRadius: "var(--radius-xs)",
              border: "none",
              cursor: "pointer",
              transition: "all var(--transition-fast)",
              background: viewMode === "per_attack" ? "var(--bg-elevated)" : "transparent",
              color: viewMode === "per_attack" ? "var(--text-primary)" : "var(--text-tertiary)",
              boxShadow: viewMode === "per_attack" ? "var(--shadow-xs)" : "none",
            }}
          >
            Chi tiết từng đòn tấn công ({cells.length})
          </button>
        </div>
      </div>

      {/* 4 Summary Stat Cards for 2 Metrics */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: "var(--space-sm)",
        }}
      >
        {/* AP50 Box */}
        <div
          style={{
            background: "var(--bg-elevated)",
            padding: "10px 14px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
            borderLeft: "3px solid var(--accent)",
          }}
        >
          <div style={{ fontSize: "0.68rem", color: "var(--text-tertiary)", textTransform: "uppercase", fontWeight: 700 }}>
            AP@50 (IoU 0.50)
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginTop: "4px" }}>
            <span style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--success)", fontFamily: "var(--font-mono)" }}>
              {cleanAp50.toFixed(3)}
            </span>
            <span style={{ fontSize: "0.85rem", color: "var(--text-tertiary)" }}>→</span>
            <span style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--danger)", fontFamily: "var(--font-mono)" }}>
              {avgAttackedAp50.toFixed(3)}
            </span>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--danger)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>
              ↓ {ap50Degradation.toFixed(1)}%
            </span>
          </div>
          <div style={{ fontSize: "0.62rem", color: "var(--text-muted)", marginTop: "2px" }}>
            Clean: {cleanAp50.toFixed(3)} | Attacked: {avgAttackedAp50.toFixed(3)}
          </div>
        </div>

        {/* mAP50-95 Box */}
        <div
          style={{
            background: "var(--bg-elevated)",
            padding: "10px 14px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
            borderLeft: "3px solid #8B5CF6",
          }}
        >
          <div style={{ fontSize: "0.68rem", color: "var(--text-tertiary)", textTransform: "uppercase", fontWeight: 700 }}>
            mAP@50-95 (COCO Standard)
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: "8px", marginTop: "4px" }}>
            <span style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--success)", fontFamily: "var(--font-mono)" }}>
              {cleanMap.toFixed(3)}
            </span>
            <span style={{ fontSize: "0.85rem", color: "var(--text-tertiary)" }}>→</span>
            <span style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--danger)", fontFamily: "var(--font-mono)" }}>
              {avgAttackedMap.toFixed(3)}
            </span>
            <span style={{ fontSize: "0.75rem", fontWeight: 700, color: "var(--danger)", marginLeft: "auto", fontFamily: "var(--font-mono)" }}>
              ↓ {mapDegradation.toFixed(1)}%
            </span>
          </div>
          <div style={{ fontSize: "0.62rem", color: "var(--text-muted)", marginTop: "2px" }}>
            Clean: {cleanMap.toFixed(3)} | Attacked: {avgAttackedMap.toFixed(3)}
          </div>
        </div>
      </div>

      {/* Bar Chart */}
      <div style={{ width: "100%", height: 270 }}>
        <ResponsiveContainer width="100%" height="100%">
          {viewMode === "overview" ? (
            /* Mode 1: 2 Metrics Overview (2 Bars per Metric) */
            <BarChart
              data={overviewData}
              margin={{ top: 20, right: 30, left: 0, bottom: 10 }}
              barCategoryGap="30%"
              barGap={8}
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
                wrapperStyle={{ paddingBottom: 10, fontSize: "0.75rem" }}
                formatter={(value) => (
                  <span style={{ color: "var(--text-secondary)", fontWeight: 600 }}>{value}</span>
                )}
              />
              <Bar
                dataKey="Clean (Trước tấn công)"
                fill="var(--success, #16A34A)"
                radius={[5, 5, 0, 0]}
                maxBarSize={60}
              />
              <Bar
                dataKey="Attacked (Sau tấn công)"
                fill="var(--danger, #EF6461)"
                radius={[5, 5, 0, 0]}
                maxBarSize={60}
              />
            </BarChart>
          ) : (
            /* Mode 2: Per-Attack Breakdown (4 Bars: AP50 Clean, AP50 Attacked, mAP Clean, mAP Attacked) */
            <BarChart
              data={perAttackData}
              margin={{ top: 20, right: 20, left: 0, bottom: 25 }}
              barCategoryGap="25%"
              barGap={4}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} opacity={0.6} />
              <XAxis
                dataKey="category"
                stroke="var(--text-muted)"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: "var(--border-subtle)" }}
                interval={0}
                angle={perAttackData.length > 3 ? -15 : 0}
                textAnchor={perAttackData.length > 3 ? "end" : "middle"}
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
                wrapperStyle={{ paddingBottom: 10, fontSize: "0.72rem" }}
              />
              <Bar dataKey="AP50 (Clean)" fill="#16A34A" radius={[3, 3, 0, 0]} maxBarSize={28} />
              <Bar dataKey="AP50 (Attacked)" fill="#EF6461" radius={[3, 3, 0, 0]} maxBarSize={28} />
              <Bar dataKey="mAP50-95 (Clean)" fill="#3B82F6" radius={[3, 3, 0, 0]} maxBarSize={28} />
              <Bar dataKey="mAP50-95 (Attacked)" fill="#F59E0B" radius={[3, 3, 0, 0]} maxBarSize={28} />
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>

      {/* Comparison Data Table */}
      <div style={{ overflowX: "auto", marginTop: "var(--space-xs)" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: "0.78rem",
            textAlign: "left",
            borderRadius: "var(--radius-sm)",
            overflow: "hidden",
          }}
        >
          <thead>
            <tr style={{ background: "var(--bg-elevated)", borderBottom: "1px solid var(--border-medium)" }}>
              <th style={{ padding: "8px 12px", color: "var(--text-secondary)", fontWeight: 600 }}>Phương pháp / Đòn tấn công</th>
              <th style={{ padding: "8px 12px", color: "var(--success)", fontWeight: 700 }}>AP50 Clean</th>
              <th style={{ padding: "8px 12px", color: "var(--danger)", fontWeight: 700 }}>AP50 Attacked</th>
              <th style={{ padding: "8px 12px", color: "#3B82F6", fontWeight: 700 }}>mAP50-95 Clean</th>
              <th style={{ padding: "8px 12px", color: "#F59E0B", fontWeight: 700 }}>mAP50-95 Attacked</th>
              <th style={{ padding: "8px 12px", color: "var(--text-tertiary)", fontWeight: 600 }}>Mức giảm AP50</th>
            </tr>
          </thead>
          <tbody>
            {perAttackData.map((row, i) => (
              <tr
                key={i}
                style={{
                  borderBottom: "1px solid var(--border-subtle)",
                  background: i % 2 === 0 ? "transparent" : "var(--bg-deep)",
                }}
              >
                <td style={{ padding: "8px 12px", fontWeight: 600, color: "var(--text-primary)" }}>{row.category}</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "var(--success)" }}>{row.ap50Clean.toFixed(3)}</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "var(--danger)", fontWeight: 600 }}>{row.ap50Attacked.toFixed(3)}</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#3B82F6" }}>{row.mapClean.toFixed(3)}</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: "#F59E0B", fontWeight: 600 }}>{row.mapAttacked.toFixed(3)}</td>
                <td style={{ padding: "8px 12px", fontFamily: "var(--font-mono)", color: row.dropAp50 > 30 ? "var(--danger)" : "var(--warning)", fontWeight: 700 }}>
                  ↓ {row.dropAp50.toFixed(1)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
