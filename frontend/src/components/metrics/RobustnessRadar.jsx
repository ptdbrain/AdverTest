"use client";

import React, { useEffect, useState } from "react";
import {
  Legend,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";

const DEFAULT_RADAR_DATA = [
  { subject: "Weather", before: 85, attacked: 45, defended: 80 },
  { subject: "Noise", before: 90, attacked: 50, defended: 84 },
  { subject: "Lighting", before: 82, attacked: 40, defended: 78 },
  { subject: "Occlusion", before: 78, attacked: 35, defended: 72 },
  { subject: "Sensor Fault", before: 88, attacked: 55, defended: 82 },
];

export default function RobustnessRadar({ data = DEFAULT_RADAR_DATA, height = 280, showDefended = true }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div style={{ width: "100%", height }} className="min-h-[220px]" />;
  }

  return (
    <div style={{ width: "100%", height }} className="relative flex flex-col items-center">
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart cx="50%" cy="50%" outerRadius="70%" data={data}>
          <PolarGrid stroke="#E2E8F0" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: "#475569", fontSize: 11, fontWeight: 500 }} />
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
          <Radar name="Trước tấn công" dataKey="before" stroke="#2563EB" fill="#2563EB" fillOpacity={0.2} />
          <Radar name="Sau tấn công" dataKey="attacked" stroke="#EF4444" fill="#EF4444" fillOpacity={0.25} />
          {showDefended && (
            <Radar name="Sau phòng thủ" dataKey="defended" stroke="#16A34A" fill="#16A34A" fillOpacity={0.2} />
          )}
          <Legend verticalAlign="bottom" iconType="circle" wrapperStyle={{ fontSize: "11px", paddingTop: "6px" }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
