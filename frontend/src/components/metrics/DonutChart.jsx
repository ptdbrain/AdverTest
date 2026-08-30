"use client";

import React, { useState, useEffect } from "react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
} from "recharts";

const COLORS = ["#2563EB", "#7C3AED", "#0EA5A8", "#F59E0B", "#EF4444", "#10B981"];

export default function DonutChart({
  data = [],
  centerValue,
  centerLabel,
  height = 200,
  innerRadius = 55,
  outerRadius = 75,
}) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div style={{ width: "100%", height }} className="min-h-[160px]" />;
  }

  return (
    <div style={{ width: "100%", height }} className="relative flex items-center justify-center">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={innerRadius}
            outerRadius={outerRadius}
            paddingAngle={3}
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.color || COLORS[index % COLORS.length]}
              />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              backgroundColor: "#FFFFFF",
              borderColor: "#E2E8F0",
              borderRadius: "8px",
              fontSize: "12px",
            }}
          />
          <Legend
            verticalAlign="bottom"
            iconType="circle"
            wrapperStyle={{ fontSize: "11px", paddingTop: "4px" }}
          />
        </PieChart>
      </ResponsiveContainer>

      {(centerValue || centerLabel) && (
        <div className="absolute top-[42%] left-1/2 -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none">
          {centerValue && (
            <div className="text-[16px] font-bold text-slate-900 leading-tight">
              {centerValue}
            </div>
          )}
          {centerLabel && (
            <div className="text-[10px] text-slate-500 font-medium leading-tight mt-0.5">
              {centerLabel}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
