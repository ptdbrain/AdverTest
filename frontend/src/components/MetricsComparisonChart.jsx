"use client";

import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { getCanonicalAttackKey, getDescriptiveAttackName } from "@/lib/attackNaming";
import { buildRunDecisionView } from "@/lib/reportMetrics";

const COLORS = ["#dc2626", "#d97706", "#7c3aed", "#2563eb", "#0f766e", "#c2410c"];

export default function MetricsComparisonChart({ report }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  const view = useMemo(() => buildRunDecisionView(report || {}), [report]);
  const cells = useMemo(() => report?.cells || [], [report]);
  const measuredMetrics = view.metrics.filter((metric) => metric.clean != null);
  const series = useMemo(() => {
    const unique = new Map();
    cells.forEach((cell) => unique.set(getCanonicalAttackKey(cell, cell.severity, report), cell));
    return [...unique.entries()].map(([key, cell], index) => ({
      key: `attack_${key}`,
      label: getDescriptiveAttackName(cell, cell.severity, report),
      cell,
      color: COLORS[index % COLORS.length],
    }));
  }, [cells, report]);

  if (!mounted || !report || view.dataState !== "MEASURED" || !measuredMetrics.length || !series.length) {
    return <div className="chart-container p-6 text-center text-sm text-slate-500">No data — cần metric clean và attacked do backend đo trực tiếp.</div>;
  }

  const data = measuredMetrics.map((metric) => {
    const row = { metric: metric.label, Clean: metric.clean };
    series.forEach(({ key, cell }) => {
      const value = cell.metrics?.[metric.key];
      row[key] = typeof value === "number" && Number.isFinite(value) ? value : null;
    });
    return row;
  });

  return (
    <div className="chart-container space-y-3 p-4">
      <div>
        <h3 className="text-sm font-semibold text-slate-900">Clean vs attacked metrics</h3>
        <p className="mt-1 text-xs text-slate-500">Không nội suy metric thiếu; mỗi cột dùng giá trị trực tiếp từ report cell.</p>
      </div>
      <div className="h-80 w-full" aria-label="Measured clean versus attacked metric chart">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 12, right: 16, left: 0, bottom: 8 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="metric" fontSize={12} />
            <YAxis domain={[0, 1]} fontSize={12} tickFormatter={(value) => value.toFixed(1)} />
            <Tooltip formatter={(value) => value == null ? "No data" : Number(value).toFixed(4)} />
            <Legend />
            <Bar dataKey="Clean" fill="#16a34a" radius={[4, 4, 0, 0]} />
            {series.map((item) => <Bar key={item.key} dataKey={item.key} name={item.label} fill={item.color} radius={[4, 4, 0, 0]} />)}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
