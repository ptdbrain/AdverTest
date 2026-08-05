"use client";

import { useMemo } from "react";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: "rgba(19, 19, 22, 0.95)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: "8px",
        padding: "10px 14px",
        backdropFilter: "blur(12px)",
        boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
      }}
    >
      <div style={{ fontSize: "0.6rem", color: "#718096", marginBottom: "6px", fontWeight: 600, letterSpacing: "0.06em", textTransform: "uppercase" }}>
        Severity {label}
      </div>
      {payload.map((p, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "0.75rem" }}>
          <span style={{ width: 6, height: 6, borderRadius: "50%", background: p.color, flexShrink: 0 }} />
          <span style={{ color: "#a0aec0" }}>{p.name}:</span>
          <span style={{ color: "#e2e8f0", fontFamily: "'JetBrains Mono', monospace", fontWeight: 600 }}>
            {(p.value * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

export default function RAChart({ cells = [], apClean = 1 }) {
  const chartData = useMemo(() => {
    if (cells.length === 0 || apClean === 0) return [];

    const bySeverity = {};
    cells.forEach((c) => {
      if (!bySeverity[c.severity]) bySeverity[c.severity] = [];
      bySeverity[c.severity].push(c.ap);
    });

    return Object.entries(bySeverity)
      .sort(([a], [b]) => Number(a) - Number(b))
      .map(([sev, aps]) => ({
        severity: Number(sev),
        ra: aps.reduce((s, v) => s + v, 0) / aps.length / apClean,
      }));
  }, [cells, apClean]);

  if (chartData.length === 0) {
    return (
      <div className="chart-container">
        <div className="chart-container__title">Robustness Accuracy Curve</div>
        <div className="empty-state">
          <div className="empty-state__message">No data yet</div>
        </div>
      </div>
    );
  }

  return (
    <div className="chart-container">
      <div className="chart-container__title">
        Robustness Accuracy — RA(s)
        <span style={{ fontSize: "0.55rem", color: "var(--text-muted)", fontWeight: 400, marginLeft: "auto" }}>
          mean AP(c,s) / AP_clean
        </span>
      </div>
      <ResponsiveContainer width="100%" height={240}>
        <AreaChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
          <defs>
            <linearGradient id="raGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#63b3ed" stopOpacity={0.25} />
              <stop offset="95%" stopColor="#63b3ed" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis
            dataKey="severity"
            tick={{ fill: "#718096", fontSize: 10 }}
            axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
            tickLine={{ stroke: "rgba(255,255,255,0.06)" }}
            label={{ value: "Severity", position: "insideBottom", offset: -2, fill: "#4a5568", fontSize: 9 }}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: "#718096", fontSize: 10 }}
            axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
            tickLine={{ stroke: "rgba(255,255,255,0.06)" }}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            label={{ value: "RA(s)", angle: -90, position: "insideLeft", offset: 10, fill: "#4a5568", fontSize: 9 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="ra"
            name="RA(s)"
            stroke="#63b3ed"
            strokeWidth={2}
            fill="url(#raGradient)"
            dot={{ fill: "#63b3ed", strokeWidth: 0, r: 3.5 }}
            activeDot={{ r: 5, fill: "#63b3ed", stroke: "#09090b", strokeWidth: 2 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
