"use client";

import React, { useState, useEffect } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

export default function TrendLineChart({
  data = [],
  height = 260,
}) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div style={{ width: "100%", height }} className="min-h-[200px]" />;
  }

  if (!Array.isArray(data) || data.length === 0) {
    return <div style={{ width: "100%", height }} className="flex items-center justify-center rounded-lg border border-dashed border-slate-300 bg-slate-50 text-sm text-slate-500">— / No verified data</div>;
  }

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 10, right: 15, left: -15, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
          <XAxis
            dataKey="exp"
            tick={{ fill: "#64748B", fontSize: 11 }}
            axisLine={{ stroke: "#E2E8F0" }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: "#64748B", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#FFFFFF",
              borderColor: "#E2E8F0",
              borderRadius: "8px",
              fontSize: "12px",
              boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
            }}
          />
          <Legend
            verticalAlign="top"
            align="right"
            iconType="circle"
            wrapperStyle={{ fontSize: "11px", paddingBottom: "8px" }}
          />
          <Line
            type="monotone"
            dataKey="Clean_mAP"
            name="Mô hình gốc"
            stroke="#2563EB"
            strokeWidth={2}
            dot={{ r: 3, fill: "#2563EB" }}
          />
          <Line
            type="monotone"
            dataKey="Attacked_mAP"
            name="Bị tấn công"
            stroke="#EF4444"
            strokeWidth={2}
            dot={{ r: 3, fill: "#EF4444" }}
          />
          <Line
            type="monotone"
            dataKey="Defended_mAP"
            name="Đã phòng thủ"
            stroke="#16A34A"
            strokeWidth={2}
            dot={{ r: 3, fill: "#16A34A" }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
