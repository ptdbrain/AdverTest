"use client";

import React, { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const DEFAULT_TREND_DATA = [
  { exp: "EXP-001", Clean_mAP: 78.4, Attacked_mAP: 42.1, Defended_mAP: 68.5 },
  { exp: "EXP-002", Clean_mAP: 79.1, Attacked_mAP: 40.5, Defended_mAP: 70.2 },
  { exp: "EXP-003", Clean_mAP: 77.8, Attacked_mAP: 38.9, Defended_mAP: 69.8 },
  { exp: "EXP-004", Clean_mAP: 80.2, Attacked_mAP: 35.4, Defended_mAP: 72.1 },
  { exp: "EXP-005", Clean_mAP: 81.0, Attacked_mAP: 39.2, Defended_mAP: 74.5 },
  { exp: "EXP-006", Clean_mAP: 82.5, Attacked_mAP: 44.8, Defended_mAP: 76.8 },
];

export default function TrendLineChart({ data = DEFAULT_TREND_DATA, height = 260 }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div style={{ width: "100%", height }} className="min-h-[200px]" />;
  }

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 15, left: -15, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
          <XAxis
            dataKey="exp"
            tick={{ fill: "#64748B", fontSize: 11 }}
            axisLine={{ stroke: "#E2E8F0" }}
            tickLine={false}
          />
          <YAxis tick={{ fill: "#64748B", fontSize: 11 }} axisLine={false} tickLine={false} />
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
