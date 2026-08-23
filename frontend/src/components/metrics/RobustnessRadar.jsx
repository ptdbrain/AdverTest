"use client";

import React from "react";
import {
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
  Tooltip,
} from "recharts";
import { RADAR_BENCHMARK } from "@/data/mockData";

export default function RobustnessRadar({
  data = RADAR_BENCHMARK,
  height = 280,
  showDefended = true,
}) {
  return (
    <div style={{ width: "100%", height }} className="relative flex flex-col items-center">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke="#E2E8F0" />
          <PolarAngleAxis
            dataKey="subject"
            tick={{ fill: "#475569", fontSize: 11, fontWeight: 500 }}
          />
          <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#CBD5E1" tick={false} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#FFFFFF",
              borderColor: "#E2E8F0",
              borderRadius: "8px",
              fontSize: "12px",
              boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
            }}
          />
          <Radar
            name="Trước tấn công"
            dataKey="before"
            stroke="#2563EB"
            fill="#2563EB"
            fillOpacity={0.2}
          />
          <Radar
            name="Sau tấn công"
            dataKey="attacked"
            stroke="#EF4444"
            fill="#EF4444"
            fillOpacity={0.25}
          />
          {showDefended && (
            <Radar
              name="Sau phòng thủ"
              dataKey="defended"
              stroke="#16A34A"
              fill="#16A34A"
              fillOpacity={0.2}
            />
          )}
          <Legend
            verticalAlign="bottom"
            iconType="circle"
            wrapperStyle={{ fontSize: "11px", paddingTop: "6px" }}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
